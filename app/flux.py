"""FLUX.2 generation via the hosted Black Forest Labs API.

Multi-reference works differently here than in the local pipeline: references go
in as numbered fields, `input_image` through `input_image_8`, rather than as one
list. Eight is the API's ceiling (the playground allows ten).

The API is asynchronous. Submitting returns a `polling_url`, which is polled
until the status turns `Ready`, at which point the result carries a signed URL
to the image — valid for ten minutes, so it is downloaded immediately rather
than handed to the caller.

Requests are still serialised behind a lock here only insofar as each call
blocks its own worker; there is no GPU to contend for, so concurrent requests
are fine.
"""

import base64
import io
import logging
import time

import requests
from PIL import Image

from .errors import AppError, UpstreamError

log = logging.getLogger(__name__)

# The API's ceiling. The playground allows ten; this is not that.
MAX_REFERENCES = 8

# Statuses that mean the request is still in flight.
PENDING_STATUSES = {"Pending", "Queued", "Processing", "Request Accepted"}


class ModelUnavailable(AppError):
    """Generation was asked for but the API is not configured."""

    status = 503
    code = "model_unavailable"


class ContentModerated(AppError):
    """The prompt or a reference image was refused by the API's filter."""

    status = 422
    code = "content_moderated"


class GenerationTimeout(AppError):
    status = 504
    code = "generation_timeout"


class FluxClient:
    def __init__(self, config):
        self.config = config
        self.session = requests.Session()

    @property
    def configured(self):
        return bool(self.config.FLUX_API_KEY)

    def generate(
        self,
        prompt,
        references=(),
        seed=None,
        aspect_ratio=None,
        output_format=None,
        safety_tolerance=None,
    ):
        """Submit, poll to completion, and return the image."""
        if not self.configured:
            raise ModelUnavailable(
                "FLUX_API_KEY is not set, so generation is unavailable. "
                "Get a key from https://dashboard.bfl.ai and put it in .env"
            )
        if len(references) > MAX_REFERENCES:
            raise AppError(
                f"at most {MAX_REFERENCES} reference images via the API, got {len(references)}",
                status=400,
                code="bad_request",
            )

        payload = {"prompt": prompt}
        # input_image, then input_image_2 .. input_image_8 — the API numbers them
        # rather than taking an array.
        for index, image in enumerate(references):
            field = "input_image" if index == 0 else f"input_image_{index + 1}"
            payload[field] = encode_reference(image, self.config.FLUX_REFERENCE_MAX_EDGE)

        if seed is not None:
            payload["seed"] = seed
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        payload["output_format"] = output_format or self.config.FLUX_OUTPUT_FORMAT
        if safety_tolerance is not None:
            payload["safety_tolerance"] = safety_tolerance

        polling_url = self._submit(payload)
        sample_url = self._poll(polling_url)
        return self._download(sample_url)

    # -- internals -------------------------------------------------------

    def _headers(self):
        return {"x-key": self.config.FLUX_API_KEY, "Accept": "application/json"}

    def _submit(self, payload):
        try:
            resp = self.session.post(
                self.config.FLUX_API_URL,
                headers=self._headers(),
                json=payload,
                timeout=self.config.FLUX_SUBMIT_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise UpstreamError(f"Could not reach the FLUX API: {exc}") from exc

        if resp.status_code in (401, 403):
            raise ModelUnavailable("The FLUX API rejected FLUX_API_KEY")
        if resp.status_code == 402:
            raise UpstreamError("The FLUX account is out of credit")
        if resp.status_code == 422:
            raise ContentModerated(_message(resp) or "The request was refused by the API")
        if resp.status_code == 429:
            raise UpstreamError("Rate limited by the FLUX API; retry shortly")
        if not resp.ok:
            raise UpstreamError(
                f"FLUX API returned {resp.status_code}", details=_message(resp)
            )

        body = _json(resp)
        polling_url = body.get("polling_url")
        if not polling_url:
            raise UpstreamError(f"FLUX API gave no polling_url: {body}")
        log.info("flux request %s submitted", body.get("id"))
        return polling_url

    def _poll(self, polling_url):
        deadline = time.monotonic() + self.config.FLUX_TIMEOUT
        interval = self.config.FLUX_POLL_INTERVAL

        while True:
            if time.monotonic() > deadline:
                raise GenerationTimeout(
                    f"FLUX did not finish within {self.config.FLUX_TIMEOUT}s"
                )
            try:
                resp = self.session.get(
                    polling_url, headers=self._headers(), timeout=self.config.FLUX_SUBMIT_TIMEOUT
                )
            except requests.RequestException as exc:
                raise UpstreamError(f"Polling the FLUX API failed: {exc}") from exc

            if not resp.ok:
                raise UpstreamError(
                    f"Polling returned {resp.status_code}", details=_message(resp)
                )

            body = _json(resp)
            status = body.get("status")

            if status == "Ready":
                sample = (body.get("result") or {}).get("sample")
                if not sample:
                    raise UpstreamError(f"FLUX reported Ready with no image: {body}")
                return sample
            if status in ("Error", "Failed"):
                raise UpstreamError(f"FLUX generation failed: {body.get('result') or body}")
            if status and "Moderated" in status:
                # Covers "Request Moderated" and "Content Moderated".
                raise ContentModerated(f"FLUX refused the request: {status}")
            if status and status not in PENDING_STATUSES:
                log.warning("unrecognised FLUX status %r, still waiting", status)

            time.sleep(interval)

    def _download(self, url):
        """Fetch the result. Its signed URL is only valid for ten minutes."""
        try:
            resp = self.session.get(url, timeout=self.config.FLUX_DOWNLOAD_TIMEOUT)
        except requests.RequestException as exc:
            raise UpstreamError(f"Could not download the generated image: {exc}") from exc
        if not resp.ok:
            raise UpstreamError(f"Downloading the generated image returned {resp.status_code}")

        try:
            image = Image.open(io.BytesIO(resp.content))
            image.load()
        except Exception as exc:  # noqa: BLE001 - whatever it is, it is not an image
            raise UpstreamError(f"FLUX returned something that is not an image: {exc}") from exc
        return image


def encode_reference(image, max_edge):
    """Base64 PNG, bounded in size.

    Every reference is inlined into the request body, so an unresized phone
    photo is megabytes of base64 on the wire for no benefit.
    """
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    if max(image.size) > max_edge:
        scale = max_edge / max(image.size)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.LANCZOS,
        )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _json(resp):
    try:
        return resp.json()
    except ValueError as exc:
        raise UpstreamError("FLUX API returned non-JSON") from exc


def _message(resp):
    try:
        body = resp.json()
    except ValueError:
        return (resp.text or "").strip()[:200] or None
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("message") or body.get("error")
        # FastAPI validation errors arrive as a list of objects.
        if isinstance(detail, list):
            return "; ".join(str(d.get("msg", d)) for d in detail)[:300]
        return detail
    return None
