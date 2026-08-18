"""Bearer-token authentication — the Python half of a Spring resource server.

AuthAndSecurity signs an RS256 token for a signed-in session and publishes the
matching public key at its JWKS endpoint. This module does what
`spring.security.oauth2.resourceserver.jwt.*` does over in generation-backend:
fetch the key by `kid`, check the signature, the issuer, the audience and the
expiry, then map the claims onto a principal.

Nothing here calls AuthAndSecurity per request. The token is self-contained and
that is the point of it — but it is also a 15-minute snapshot, so a permission
removed from the user table stays spendable until the token expires.
"""

import functools

import jwt
from flask import current_app, g, request
from jwt import PyJWKClient

from .errors import AuthError, ForbiddenError

# Claims minted by AuthAndSecurity's AccessTokenService.
ROLES_CLAIM = "roles"
EMAIL_CLAIM = "email"


class TokenVerifier:
    def __init__(self, config):
        self.config = config
        # Caches keys in memory and re-fetches when a token names an unknown
        # `kid`, which is what makes key rotation over there a non-event here.
        self._jwks = PyJWKClient(
            config.JWKS_URI,
            cache_keys=True,
            lifespan=3600,
            timeout=config.HTTP_TIMEOUT,
        )

    def verify(self, token):
        try:
            signing_key = self._jwks.get_signing_key_from_jwt(token)
        except jwt.PyJWKClientError as exc:
            # No usable key: either the issuer is unreachable or the token was
            # signed by something else entirely. Not the caller's fault to fix.
            raise AuthError(f"Could not obtain the signing key: {exc}") from exc
        except jwt.InvalidTokenError as exc:
            # Finding the key means reading the header, so a token too malformed
            # to parse fails here rather than at decode below.
            raise AuthError(f"Access token is not valid: {exc}") from exc

        try:
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=self.config.JWT_ALGORITHMS,
                issuer=self.config.JWT_ISSUER,
                audience=self.config.JWT_AUDIENCE,
                leeway=self.config.JWT_LEEWAY,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthError("Access token has expired; fetch a new one from /api/token") from exc
        except jwt.InvalidAudienceError as exc:
            raise AuthError(
                f"Token audience does not match '{self.config.JWT_AUDIENCE}'"
            ) from exc
        except jwt.InvalidIssuerError as exc:
            raise AuthError(f"Token issuer does not match '{self.config.JWT_ISSUER}'") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthError(f"Access token is not valid: {exc}") from exc

        return _principal(claims)


def _principal(claims):
    roles = claims.get(ROLES_CLAIM) or []
    if isinstance(roles, str):
        roles = [roles]
    return {
        # The user-table row id (a UUID), not the address — an email can be
        # corrected, and anything stored against a subject would be orphaned.
        "id": claims["sub"],
        "email": claims.get(EMAIL_CLAIM),
        # Bare names: "USER", "ADMIN". The ROLE_ prefix is Spring Security's own
        # convention and is deliberately not in the token.
        "roles": [str(r).upper() for r in roles],
        "claims": claims,
    }


def extract_bearer_token():
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("Missing or malformed Authorization header")
    return token.strip()


def require_auth(view):
    """Populate `g.user` and `g.token` from a verified token, or fail with 401."""

    @functools.wraps(view)
    def wrapper(*args, **kwargs):
        token = extract_bearer_token()
        g.token = token
        g.user = current_app.extensions["verifier"].verify(token)
        return view(*args, **kwargs)

    return wrapper


def require_role(*roles):
    """Require at least one of `roles`, named as they appear in the token."""
    wanted = {r.upper().removeprefix("ROLE_") for r in roles}

    def decorator(view):
        @functools.wraps(view)
        def wrapper(*args, **kwargs):
            held = {r.removeprefix("ROLE_") for r in g.user["roles"]}
            if not wanted & held:
                raise ForbiddenError(f"Requires one of: {', '.join(sorted(wanted))}")
            return view(*args, **kwargs)

        return wrapper

    return decorator
