"""Job status for the frontend, and the endpoint Cloud Tasks delivers jobs to.

    GET  /api/jobs/<id>         status, and once done the cost and new balance
    GET  /api/jobs/<id>/image   the generated PNG
    POST /internal/jobs/run     Cloud Tasks only (TASKS_MODE=cloud)

A job is visible only to the user who created it. Anyone else gets the same
404 as for an id that does not exist, so ids cannot be probed.
"""

import functools
import io
import logging

from flask import Blueprint, current_app, g, jsonify, request, send_file

from .auth import require_auth
from .errors import AppError, AuthError, NotFoundError
from .job_runner import run_job
from .job_store import DONE, public_view

log = logging.getLogger(__name__)

bp = Blueprint("jobs", __name__)


def _jobs():
    return current_app.extensions["jobs"]


def _config():
    return current_app.config["APP_CONFIG"]


def _own_job(job_id):
    try:
        job = _jobs().get(job_id)
    except ValueError:
        job = None
    if job is None or job.get("userId") != (g.user or {}).get("id"):
        raise NotFoundError("No such job")
    return job


@bp.get("/api/jobs/<job_id>")
@require_auth
def job_status(job_id):
    return jsonify(public_view(_own_job(job_id)))


@bp.get("/api/jobs/<job_id>/image")
@require_auth
def job_image(job_id):
    job = _own_job(job_id)
    if job["status"] != DONE:
        raise AppError("The job has no image yet", status=409, code="job_not_done")
    data = current_app.extensions["storage"].get(job["resultKey"])
    if data is None:
        raise NotFoundError("The generated image is no longer stored")

    response = send_file(io.BytesIO(data), mimetype="image/png",
                         download_name=f"innigo-{job_id}.png")
    response.headers["Cache-Control"] = "private, max-age=86400"
    return response


def require_cloud_tasks(view):
    """Accept only an OIDC token Cloud Tasks minted as TASKS_SERVICE_ACCOUNT.

    Checked here, not only by Cloud Run IAM, because the API and the worker may
    be the same public service — and this route spends users' tokens.
    """

    @functools.wraps(view)
    def wrapper(*args, **kwargs):
        config = _config()
        if config.TASKS_MODE != "cloud":
            raise NotFoundError("Not found")  # does not exist outside cloud mode

        header = request.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise AuthError("Missing Cloud Tasks OIDC token")
        try:
            from google.auth.transport import requests as google_requests
            from google.oauth2 import id_token

            claims = id_token.verify_oauth2_token(
                token, google_requests.Request(), audience=config.TASKS_OIDC_AUDIENCE
            )
        except Exception as exc:  # noqa: BLE001 - any failure means "not Cloud Tasks"
            raise AuthError(f"Invalid Cloud Tasks OIDC token: {exc}") from exc
        if claims.get("email") != config.TASKS_SERVICE_ACCOUNT or not claims.get("email_verified"):
            raise AuthError("OIDC token is not from the Cloud Tasks service account")
        return view(*args, **kwargs)

    return wrapper


@bp.post("/internal/jobs/run")
@require_cloud_tasks
def run_job_endpoint():
    payload = request.get_json(silent=True) or {}
    job_id, token = payload.get("jobId"), payload.get("token")
    if not job_id or not token:
        # Malformed task: answer 200 so Cloud Tasks drops it instead of retrying.
        log.error("dropping a task without jobId/token")
        return jsonify(status="dropped"), 200

    # Cloud Tasks counts retries from 0.
    attempt = int(request.headers.get("X-CloudTasks-TaskRetryCount", "0")) + 1
    finished = run_job(job_id, token, attempt=attempt,
                       max_attempts=_config().TASKS_MAX_ATTEMPTS)
    if finished:
        return jsonify(status="finished"), 200
    # Non-2xx: Cloud Tasks retries with the queue's backoff.
    return jsonify(status="retry"), 503
