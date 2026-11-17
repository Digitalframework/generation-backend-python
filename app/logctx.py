"""Logging helpers for the generation pipeline.

Every log record carries a `job` attribute: the first eight characters of the
job id while a job is being queued or run, "-" otherwise. With it in the format
(see create_app) all lines of one generation can be grepped together, even when
several jobs run interleaved in the thread pool.

`stage()` logs how long each step took, so a slow generation shows at a glance
whether VGGT, the catalogue or FLUX was the reason.
"""

import contextvars
import logging
import time
from contextlib import contextmanager

_current_job = contextvars.ContextVar("current_job", default="-")
_base_factory = logging.getLogRecordFactory()
_installed = False


def install():
    """Add the `job` attribute to every log record. Idempotent."""
    global _installed
    if _installed:
        return

    def factory(*args, **kwargs):
        record = _base_factory(*args, **kwargs)
        record.job = _current_job.get()
        return record

    logging.setLogRecordFactory(factory)
    _installed = True


@contextmanager
def job_context(job_id):
    """Tag log lines inside the block with `job_id`."""
    token = _current_job.set((job_id or "-")[:8])
    try:
        yield
    finally:
        _current_job.reset(token)


@contextmanager
def stage(log, name, **fields):
    """Log the start, duration and failure of one pipeline step."""
    extra = "".join(f" {key}={value}" for key, value in fields.items())
    log.info("%s started%s", name, extra)
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        log.warning("%s failed after %.2fs: %s: %s",
                    name, time.perf_counter() - start, type(exc).__name__, exc)
        raise
    log.info("%s done in %.2fs", name, time.perf_counter() - start)
