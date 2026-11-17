"""Hand a queued job to whatever runs it. Chosen by config.TASKS_MODE.

Every dispatcher passes the caller's access token along with the job id. The
worker needs it for the two calls made on the user's behalf — the catalogue
fetch and the token debit — and AuthAndSecurity has no service-to-service
credential yet. The token is never written to the job store.

That token is good for 15 minutes. A job that waits in the queue longer than
that, or a Cloud Tasks retry after it expires, fails with `unauthorized` and
costs the user nothing. A service account on AuthAndSecurity would remove the
limit; see the README.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor

from .errors import AppError
from .lazy import Lazy

log = logging.getLogger(__name__)


class DispatchUnavailable(AppError):
    status = 503
    code = "dispatch_unavailable"


class InlineDispatcher:
    """Run the job before returning. For tests and debugging."""

    def __init__(self, app):
        self.app = app

    def dispatch(self, job_id, token):
        from .job_runner import run_job

        run_job(job_id, token, attempt=1, max_attempts=1)


class LocalDispatcher:
    """A thread pool in this process. The localhost default.

    A restart loses whatever was queued or running; those jobs stay "queued" or
    "running" in the store. Fine on a laptop, and the reason cloud mode exists.
    """

    def __init__(self, app, threads):
        self.app = app
        self._pool = ThreadPoolExecutor(max_workers=max(1, threads), thread_name_prefix="job")

    def dispatch(self, job_id, token):
        self._pool.submit(self._run, job_id, token)

    def _run(self, job_id, token):
        from .job_runner import run_job

        with self.app.app_context():
            try:
                run_job(job_id, token, attempt=1, max_attempts=1)
            except Exception:  # noqa: BLE001 - a dead job must not kill the pool thread
                log.exception("job %s crashed outside the runner's own handling", job_id)


class CloudTasksDispatcher:
    """One Cloud Task per job, delivered as an authenticated POST to the worker."""

    def __init__(self, config):
        missing = [name for name in ("GCP_PROJECT", "WORKER_URL", "TASKS_SERVICE_ACCOUNT")
                   if not getattr(config, name)]
        if missing:
            raise DispatchUnavailable(f"TASKS_MODE=cloud needs {', '.join(missing)}")
        try:
            from google.cloud import tasks_v2
        except ImportError as exc:
            raise DispatchUnavailable(
                "google-cloud-tasks is not installed — pip install -r requirements.txt"
            ) from exc
        self.config = config
        self._tasks = tasks_v2
        self._client = tasks_v2.CloudTasksClient()
        self._queue = self._client.queue_path(
            config.GCP_PROJECT, config.GCP_LOCATION, config.TASKS_QUEUE
        )

    def dispatch(self, job_id, token):
        config = self.config
        task = {
            # Named after the job, so a double submit of the same job is
            # rejected by Cloud Tasks instead of generating (and charging) twice.
            "name": f"{self._queue}/tasks/{job_id}",
            "http_request": {
                "http_method": self._tasks.HttpMethod.POST,
                "url": f"{config.WORKER_URL}/internal/jobs/run",
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"jobId": job_id, "token": token}).encode(),
                "oidc_token": {
                    "service_account_email": config.TASKS_SERVICE_ACCOUNT,
                    "audience": config.TASKS_OIDC_AUDIENCE,
                },
            },
            "dispatch_deadline": {"seconds": config.TASKS_DISPATCH_DEADLINE},
        }
        try:
            self._client.create_task(parent=self._queue, task=task)
        except Exception as exc:  # noqa: BLE001 - google.api_core raises many kinds
            raise DispatchUnavailable(f"Could not queue the job: {exc}") from exc


def create_dispatcher(app, config):
    mode = config.TASKS_MODE
    if mode == "inline":
        return InlineDispatcher(app)
    if mode == "local":
        return LocalDispatcher(app, config.LOCAL_WORKER_THREADS)
    if mode == "cloud":
        return Lazy(lambda: CloudTasksDispatcher(config))
    raise ValueError(f"unknown TASKS_MODE {mode!r} (local | inline | cloud)")

