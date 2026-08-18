"""HTTP client for AuthAndSecurity's balance endpoints.

The balance lives in AuthAndSecurity's `app_user.token_balance` and is not in
the access token — deliberately, since a token is a 15-minute snapshot and a
balance changes on every spend. So it is read across the wire, forwarding the
caller's own token.

NOTE: the endpoints called here do not exist in AuthAndSecurity yet, and its
SecurityConfig has no `oauth2ResourceServer` chain, so bearer tokens are not
accepted there at all. See the README for the change needed.
"""

import logging

import requests
from cachetools import TTLCache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .errors import AuthError, ForbiddenError, InsufficientTokens, NotFoundError, UpstreamError

log = logging.getLogger(__name__)


class AuthServiceClient:
    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        retry = Retry(
            total=config.HTTP_RETRIES,
            backoff_factor=0.2,
            status_forcelist=(502, 503, 504),
            # GET only: retrying a debit would spend twice.
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=20)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self._cache = TTLCache(maxsize=1024, ttl=config.BALANCE_CACHE_TTL or 1)

    def get_balance(self, token):
        if self.config.BALANCE_CACHE_TTL and token in self._cache:
            return self._cache[token]

        data = self._request("GET", self.config.BALANCE_PATH, token)
        balance = self._extract_balance(data)

        if self.config.BALANCE_CACHE_TTL:
            self._cache[token] = balance
        return balance

    def spend(self, token, amount, reason=None):
        """Debit `amount` tokens, returning the resulting balance.

        Done by AuthAndSecurity so the check-and-decrement is one transaction
        against the row it owns. Reading the balance here and writing back a
        smaller number would race with any concurrent request and with the
        PayPal top-up in PaypalService.
        """
        if amount <= 0:
            raise ValueError("amount must be positive")

        payload = {"amount": amount}
        if reason:
            payload["reason"] = reason
        data = self._request("POST", self.config.SPEND_PATH, token, json=payload)

        self._cache.pop(token, None)
        return self._extract_balance(data)

    def health(self):
        """Whether the JWKS endpoint answers — it is the one path that is permitAll."""
        try:
            return self.session.get(
                self.config.JWKS_URI, timeout=self.config.HTTP_TIMEOUT
            ).ok
        except requests.RequestException:
            return False

    # -- internals -------------------------------------------------------

    def _request(self, method, path, token, json=None):
        url = f"{self.config.AUTH_BASE_URL}{path}"
        try:
            resp = self.session.request(
                method,
                url,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                json=json,
                timeout=self.config.HTTP_TIMEOUT,
                # A session-cookie chain answers an unauthenticated call with a
                # 302 to /login. Following it would turn that into a 200 and an
                # HTML page, which is a far more confusing failure than the 302.
                allow_redirects=False,
            )
        except requests.Timeout as exc:
            raise UpstreamError(f"Timed out calling {path}") from exc
        except requests.RequestException as exc:
            log.warning("Call to %s failed: %s", path, exc)
            raise UpstreamError(f"Could not reach AuthAndSecurity at {path}") from exc

        return self._handle(resp, path)

    def _handle(self, resp, path):
        if resp.is_redirect:
            raise UpstreamError(
                f"{path} redirected to {resp.headers.get('Location')!r} instead of "
                "answering — it is behind the session-cookie chain and does not "
                "accept bearer tokens"
            )
        if resp.status_code == 401:
            raise AuthError("AuthAndSecurity rejected the token")
        if resp.status_code == 403:
            raise ForbiddenError("AuthAndSecurity denied access to this resource")
        if resp.status_code == 404:
            raise NotFoundError(f"No such endpoint on AuthAndSecurity: {path}")
        if resp.status_code in (402, 409):
            raise InsufficientTokens(self._error_message(resp) or "Not enough tokens")
        if resp.status_code >= 400:
            raise UpstreamError(
                f"AuthAndSecurity returned {resp.status_code} for {path}",
                details=self._error_message(resp),
            )

        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError as exc:
            raise UpstreamError(f"AuthAndSecurity returned non-JSON for {path}") from exc

    @staticmethod
    def _error_message(resp):
        try:
            body = resp.json()
        except ValueError:
            return (resp.text or "").strip()[:200] or None
        if isinstance(body, dict):
            # ProblemDetail (spring.mvc.problemdetails.enabled=true) uses `detail`.
            return body.get("detail") or body.get("message") or body.get("error")
        return None

    @staticmethod
    def _extract_balance(data):
        if isinstance(data, int):
            return data
        # `tokenBalance` is the name CurrentUser uses; `balance` the likelier one
        # for a dedicated endpoint. Accept either rather than pin it now.
        if isinstance(data, dict):
            for key in ("balance", "tokenBalance"):
                if isinstance(data.get(key), int):
                    return data[key]
        raise UpstreamError("No token balance in the AuthAndSecurity response")
