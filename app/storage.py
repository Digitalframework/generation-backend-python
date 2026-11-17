"""Where a job's images live between the request that queues it and the worker.

The request that creates a job and the worker that runs it are not the same
call — in cloud mode not even the same machine — so the uploaded room photos
and references are written here first, and the result is written back here
for GET /api/jobs/<id>/image to serve.

Keys are plain relative paths: "<job_id>/rooms/0.png", "<job_id>/result.png".
"""

import io
import os
from pathlib import Path

from PIL import Image

from .errors import AppError
from .lazy import Lazy


class StorageUnavailable(AppError):
    status = 503
    code = "storage_unavailable"


class LocalStorage:
    """A directory on disk. Shared by every process on the machine."""

    def __init__(self, root):
        self.root = Path(root)

    def _path(self, key):
        path = (self.root / key).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError(f"storage key escapes the storage root: {key!r}")
        return path

    def put(self, key, data, content_type="application/octet-stream"):
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)  # a reader never sees half a file

    def get(self, key):
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            return None


class GcsStorage:
    """A Cloud Storage bucket. Credentials come from the environment (ADC)."""

    def __init__(self, bucket_name):
        if not bucket_name:
            raise StorageUnavailable("STORAGE_BACKEND=gcs needs GCS_BUCKET")
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise StorageUnavailable(
                "google-cloud-storage is not installed — pip install -r requirements.txt"
            ) from exc
        self._bucket = storage.Client().bucket(bucket_name)

    def put(self, key, data, content_type="application/octet-stream"):
        self._bucket.blob(key).upload_from_string(data, content_type=content_type)

    def get(self, key):
        from google.api_core.exceptions import NotFound

        try:
            return self._bucket.blob(key).download_as_bytes()
        except NotFound:
            return None


def create_storage(config):
    backend = config.STORAGE_BACKEND
    if backend == "local":
        return LocalStorage(config.LOCAL_STORAGE_DIR)
    if backend == "gcs":
        # Constructed lazily on first use: a misconfigured bucket should fail
        # the request that needs it, not stop the service from starting.
        return Lazy(lambda: GcsStorage(config.GCS_BUCKET))
    raise ValueError(f"unknown STORAGE_BACKEND {backend!r} (local | gcs)")


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def load_image(data):
    image = Image.open(io.BytesIO(data))
    image.load()
    return image
