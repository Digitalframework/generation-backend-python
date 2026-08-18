"""HTTP client for generation-backend's furniture catalogue.

`/api/images/**` lives in generation-backend (:8081), not in this Flask app and
not in AuthAndSecurity — see ImageController. Two routes are used here, both
behind `hasAnyRole('USER','ADMIN')`, so the caller's own token is forwarded
rather than any service credential:

    GET /api/images/batch?ids=1,3,5   metadata for a known list of ids
    GET /api/images/{id}/content      the PNG bytes

The metadata is what makes the composite possible: `widthCm`/`heightCm` are the
real-world dimensions the perspective fit needs, so an id is enough to place a
piece of furniture correctly without the caller measuring anything.

Ids with no matching row are simply absent from the batch response — the
controller says so explicitly — so the gap is detected here and reported as a
404 naming the missing ids, rather than silently generating a picture with one
sofa in it when two were asked for.
"""

import io
import logging

import requests
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .errors import AuthError, ForbiddenError, NotFoundError, UpstreamError

log = logging.getLogger(__name__)


class ImageServiceClient:
    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        retry = Retry(
            total=config.HTTP_RETRIES,
            backoff_factor=0.2,
            status_forcelist=(502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=20)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_batch(self, token, ids):
        """Metadata for `ids`, in the order asked for.

        One request for the whole list: the point of /batch is a single indexed
        `IN` query instead of an /api/images/{id} round trip each.
        """
        if not ids:
            return []

        url = f"{self.config.IMAGE_BASE_URL}{self.config.IMAGE_BATCH_PATH}"
        body = self._get(url, token, params={"ids": ",".join(str(i) for i in ids)})

        if not isinstance(body, list):
            raise UpstreamError(f"{self.config.IMAGE_BATCH_PATH} did not return a list")

        by_id = {item.get("id"): item for item in body if isinstance(item, dict)}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise NotFoundError(
                "no catalogue image with id "
                + ", ".join(str(i) for i in missing)
                + " — list them with GET /api/images"
            )
        return [by_id[i] for i in ids]

    def get_content(self, token, image_id):
        """The stored PNG for one id, as bytes.

        Bytes rather than a decoded image: the cutout loader reads a file object
        and wants the file as stored, alpha channel and all, which a decode and
        re-encode here would only put at risk.
        """
        url = f"{self.config.IMAGE_BASE_URL}/api/images/{image_id}/content"
        resp = self._request(url, token, accept="image/png")

        try:
            probe = Image.open(io.BytesIO(resp.content))
            probe.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise UpstreamError(
                f"image {image_id} content is not a readable image: {exc}"
            ) from exc
        return resp.content

    def fetch(self, token, ids):
        """`[(metadata, png_bytes), …]` for `ids`, in order."""
        metadata = self.get_batch(token, ids)
        return [(item, self.get_content(token, item["id"])) for item in metadata]

    # -- internals -------------------------------------------------------

    def _get(self, url, token, params=None):
        resp = self._request(url, token, params=params)
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError as exc:
            raise UpstreamError(f"generation-backend returned non-JSON for {url}") from exc

    def _request(self, url, token, params=None, accept="application/json"):
        try:
            resp = self.session.get(
                url,
                headers={"Accept": accept, "Authorization": f"Bearer {token}"},
                params=params,
                timeout=self.config.IMAGE_HTTP_TIMEOUT,
                # An unauthenticated call to a session-cookie chain answers with a
                # 302 to /login; following it would turn that into a 200 and an
                # HTML page, which is a far more confusing failure than the 302.
                allow_redirects=False,
            )
        except requests.Timeout as exc:
            raise UpstreamError(f"Timed out calling {url}") from exc
        except requests.RequestException as exc:
            log.warning("Call to %s failed: %s", url, exc)
            raise UpstreamError(
                f"Could not reach the image catalogue at {self.config.IMAGE_BASE_URL}"
            ) from exc

        return self._check(resp, url)

    @staticmethod
    def _check(resp, url):
        if resp.is_redirect:
            raise UpstreamError(
                f"{url} redirected to {resp.headers.get('Location')!r} instead of "
                "answering — it does not accept bearer tokens"
            )
        if resp.status_code == 401:
            raise AuthError("The image catalogue rejected the token")
        if resp.status_code == 403:
            raise ForbiddenError(
                "The token carries no USER or ADMIN role, which every "
                "/api/images route requires"
            )
        if resp.status_code == 404:
            raise NotFoundError(f"No such image resource: {url}")
        if resp.status_code >= 400:
            raise UpstreamError(
                f"The image catalogue returned {resp.status_code} for {url}",
                details=(resp.text or "").strip()[:200] or None,
            )
        return resp
