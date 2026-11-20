"""Run one queued generation job to completion and record how it ended.

Called by every dispatcher: a local pool thread, the inline dispatcher, or the
Cloud Tasks worker endpoint. The generation itself is
generate_routes.run_generation; this module is the bookkeeping around it —
status, storage, the token debit, and whether a failure is worth a retry.

Retries only exist in cloud mode (max_attempts > 1). A failure the caller
caused — bad input, moderation, not enough tokens, an expired token — is final
on the first attempt. Upstream trouble (FLUX down, rate limited, timed out) is
handed back to Cloud Tasks until the last attempt, then recorded as final.
"""

import json
import logging
import time

from flask import current_app

from . import generate_routes as gen
from .errors import AppError
from .job_store import CHARGING, DONE, FAILED, FINISHED, QUEUED, RUNNING
from .logctx import job_context, stage
from .storage import load_image, png_bytes

log = logging.getLogger(__name__)


def run_job(job_id, token, attempt=1, max_attempts=1):
    """Run `job_id`. True when the job has ended; False to ask for a retry."""
    with job_context(job_id):
        return _run_job(job_id, token, attempt, max_attempts)


def _run_job(job_id, token, attempt, max_attempts):
    started = time.perf_counter()
    jobs = current_app.extensions["jobs"]
    storage = current_app.extensions["storage"]

    job = jobs.get(job_id)
    if job is None:
        log.warning("job %s does not exist; dropping it", job_id)
        return True
    if job["status"] in FINISHED:
        log.info("job %s already %s; ignoring duplicate delivery", job_id, job["status"])
        return True

    waited = (time.time() * 1000 - job.get("createdAt", 0)) / 1000
    log.info("job %s started: attempt %d/%d, status was %s, waited %.1fs in queue",
             job_id, attempt, max_attempts, job["status"], waited)

    charging = job["status"] == CHARGING and job.get("resultKey")
    job = jobs.update(job_id, status=CHARGING if charging else RUNNING, attempts=attempt)
    params = json.loads(job["params"])

    try:
        if not charging:
            with stage(log, "load inputs"):
                rooms = [_image(storage, key) for key in job["inputs"]["rooms"]]
                uploads = [_image(storage, key) for key in job["inputs"]["references"]]
            outcome = gen.run_generation(params, rooms, uploads, token)

            key = f"{job_id}/result.png"
            with stage(log, "store result", key=key):
                storage.put(key, png_bytes(outcome["image"]), "image/png")
                if outcome.get("composite") is not None:
                    # What FLUX was given as image 1, for comparing with the result.
                    storage.put(f"{job_id}/composite.png", png_bytes(outcome["composite"]),
                                "image/png")
            # Recorded before the debit: if the debit fails transiently, the
            # retry finds the image here and does not pay FLUX a second time.
            job = jobs.update(job_id, status=CHARGING, resultKey=key, result={
                "referencesUsed": outcome["references_used"],
                "catalogueImages": outcome["catalogue_images"],
            })

        else:
            log.info("job %s resumes at the token debit; image already stored", job_id)

        cost = params["cost"]
        # Only now: a moderated, failed or timed-out generation costs nothing.
        with stage(log, "token debit", amount=cost):
            remaining = gen._client().spend(
                token, cost, reason=f"flux2:{job['result']['referencesUsed']}ref"
            )
        jobs.update(job_id, status=DONE, error=None,
                    result={**job["result"], "tokensSpent": cost, "balance": remaining})
        log.info("job %s done in %.1fs: %d tokens spent, balance now %s",
                 job_id, time.perf_counter() - started, cost, remaining)
        return True

    except AppError as exc:
        error, transient = exc.to_dict(), exc.status >= 500
        error["code"] = error.pop("error")
    except Exception as exc:  # noqa: BLE001 - unknown failure: record it, maybe retry
        log.exception("job %s crashed", job_id)
        error, transient = {"code": "internal_error", "message": str(exc) or repr(exc)}, True

    if transient and attempt < max_attempts:
        log.warning("job %s attempt %d/%d failed after %.1fs, will retry: %s: %s",
                    job_id, attempt, max_attempts, time.perf_counter() - started,
                    error["code"], error["message"])
        jobs.update(job_id, status=CHARGING if job.get("resultKey") else QUEUED, lastError=error)
        return False

    log.warning("job %s failed after %.1fs (attempt %d/%d, %s): %s: %s%s",
                job_id, time.perf_counter() - started, attempt, max_attempts,
                "transient" if transient else "final", error["code"], error["message"],
                f" details={error['details']}" if error.get("details") else "")
    jobs.update(job_id, status=FAILED, error=error)
    return True


def _image(storage, key):
    data = storage.get(key)
    if data is None:
        # Final, not retried: a file that is gone will not come back.
        raise AppError(f"job input {key} is missing from storage", status=410,
                       code="job_input_missing")
    return load_image(data)
