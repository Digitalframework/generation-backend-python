"""Job records: what was asked for, how far it got, and what it cost.

A job is a plain dict so it can go to Firestore or to JSON unchanged:

    id, userId, status, createdAt, updatedAt,
    params    everything the worker needs to rebuild the request
    inputs    storage keys of the uploaded rooms and references
    resultKey storage key of the generated image, once there is one
    result    tokensSpent, balance, referencesUsed, catalogueImages
    error     {code, message} when status is "failed"
    attempts  how many times a worker picked it up

status moves queued -> running -> (charging ->) done | failed. "charging"
means the image exists but the token debit has not gone through yet; a retry
that finds a job there skips straight to the debit instead of paying FLUX twice.
"""

import json
import os
import threading
import time
from pathlib import Path

from .errors import AppError
from .lazy import Lazy

QUEUED, RUNNING, CHARGING, DONE, FAILED = "queued", "running", "charging", "done", "failed"
FINISHED = {DONE, FAILED}


class JobStoreUnavailable(AppError):
    status = 503
    code = "job_store_unavailable"


def now_ms():
    return int(time.time() * 1000)


def public_view(job):
    """What the job's owner is shown. Params and storage keys stay internal."""
    view = {
        "jobId": job["id"],
        "status": job["status"],
        "createdAt": job.get("createdAt"),
        "updatedAt": job.get("updatedAt"),
    }
    if job["status"] == DONE:
        view["imageUrl"] = f"/api/jobs/{job['id']}/image"
        view.update(job.get("result") or {})
    if job["status"] == FAILED:
        view["error"] = job.get("error")
    return view


class FileJobStore:
    """One JSON file per job, next to the job's images.

    Files rather than a dict in memory because gunicorn runs more than one
    process: the POST that creates a job and the GET that polls it can land on
    different workers, and both have to see the same record.
    """

    def __init__(self, root):
        self.root = Path(root)
        self._lock = threading.Lock()

    def _path(self, job_id):
        if not job_id or not job_id.replace("-", "").isalnum():
            raise ValueError(f"bad job id {job_id!r}")
        return self.root / job_id / "job.json"

    def create(self, job):
        stamp = now_ms()
        job = {**job, "createdAt": stamp, "updatedAt": stamp}
        self._write(job)
        return job

    def get(self, job_id):
        try:
            return json.loads(self._path(job_id).read_text("utf-8"))
        except (FileNotFoundError, ValueError):
            return None

    def update(self, job_id, **fields):
        with self._lock:
            job = self.get(job_id)
            if job is None:
                raise KeyError(job_id)
            job.update(fields, updatedAt=now_ms())
            self._write(job)
            return job

    def _write(self, job):
        path = self._path(job["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"job.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(job), "utf-8")
        os.replace(tmp, path)


class FirestoreJobStore:
    """A Firestore collection, one document per job id."""

    def __init__(self, collection, project=None):
        try:
            from google.cloud import firestore
        except ImportError as exc:
            raise JobStoreUnavailable(
                "google-cloud-firestore is not installed — pip install -r requirements.txt"
            ) from exc
        self._collection = firestore.Client(project=project).collection(collection)

    def create(self, job):
        stamp = now_ms()
        job = {**job, "createdAt": stamp, "updatedAt": stamp}
        self._collection.document(job["id"]).set(job)
        return job

    def get(self, job_id):
        snapshot = self._collection.document(job_id).get()
        return snapshot.to_dict() if snapshot.exists else None

    def update(self, job_id, **fields):
        fields["updatedAt"] = now_ms()
        self._collection.document(job_id).update(fields)
        return self.get(job_id)


def create_job_store(config):
    kind = config.JOB_STORE
    if kind == "file":
        return FileJobStore(config.LOCAL_STORAGE_DIR)
    if kind == "firestore":
        return Lazy(lambda: FirestoreJobStore(config.FIRESTORE_COLLECTION, config.GCP_PROJECT))
    raise ValueError(f"unknown JOB_STORE {kind!r} (file | firestore)")

