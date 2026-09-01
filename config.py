import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return int(default)


def _float(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


class Config:
    """Mirrors generation-backend's application.properties.

    The values here must match `app.jwt.*` in AuthAndSecurity character for
    character — issuer especially, trailing slash and all — or every token is
    rejected.
    """

    # AuthAndSecurity. Signs the tokens, owns the user table, owns the balance.
    AUTH_BASE_URL = os.getenv("AUTH_BASE_URL", "http://localhost:8080").rstrip("/")

    # Where the public half of the signing key is fetched from: once, then cached,
    # and re-fetched when a token names a `kid` that is not in the cache. That is
    # what lets the key rotate over there without a change here.
    JWKS_URI = os.getenv("JWKS_URI") or f"{AUTH_BASE_URL}/.well-known/jwks.json"

    # `iss` must equal app.jwt.issuer; `aud` must equal app.jwt.audience, which
    # rejects a token minted for a different service.
    JWT_ISSUER = os.getenv("JWT_ISSUER") or AUTH_BASE_URL
    JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "generation-backend")
    JWT_ALGORITHMS = ["RS256"]

    # Tolerance for clock skew between this host and the issuer, in seconds.
    JWT_LEEWAY = _int("JWT_LEEWAY", 30)

    # Balance endpoints on AuthAndSecurity. See README: these do not exist yet.
    BALANCE_PATH = os.getenv("BALANCE_PATH", "/api/tokens/balance")
    SPEND_PATH = os.getenv("SPEND_PATH", "/api/tokens/spend")

    # What a composite costs, per reference image placed.
    TOKENS_PER_OBJECT = _int("TOKENS_PER_OBJECT", 1)

    # --- The furniture catalogue, served by generation-backend on :8081 ------
    # /api/generate fetches each `image_ids` entry from here: the metadata in
    # one batch call, then the PNG behind each id. The stored width_cm and
    # height_cm are what let a piece be placed at the right scale from an id.
    IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "http://localhost:8081").rstrip("/")
    IMAGE_BATCH_PATH = os.getenv("IMAGE_BATCH_PATH", "/api/images/batch")

    # Separate from HTTP_TIMEOUT: this one pulls PNGs, not small JSON bodies.
    IMAGE_HTTP_TIMEOUT = _int("IMAGE_HTTP_TIMEOUT", 10)

    # The composite becomes FLUX's "image 1", so the ceiling here is really
    # MAX_REFERENCES (8) — more pieces than that cannot be referred to.
    MAX_CATALOGUE_IMAGES = _int("MAX_CATALOGUE_IMAGES", 8)

    # The room a generation is composited into. Rooms are not uploaded anywhere
    # yet, so this is a file on disk; `scene`/`rooms` on the request override it.
    ROOM_IMAGE_PATH = Path(os.getenv("ROOM_IMAGE_PATH", "assets/room.png"))

    # --- FLUX.2, hosted at Black Forest Labs ---------------------------------
    # Key from https://dashboard.bfl.ai. Without it /api/generate answers 503.
    FLUX_API_KEY = os.getenv("FLUX_API_KEY") or None
    FLUX_API_URL = os.getenv("FLUX_API_URL", "https://api.bfl.ai/v1/flux-2-pro")

    FLUX_ASPECT_RATIO = os.getenv("FLUX_ASPECT_RATIO", "1:1")
    FLUX_OUTPUT_FORMAT = os.getenv("FLUX_OUTPUT_FORMAT", "png")

    # Every reference is inlined as base64 in the request body, so size matters
    # on the wire here, not just in memory.
    FLUX_REFERENCE_MAX_EDGE = _int("FLUX_REFERENCE_MAX_EDGE", 1024)

    # The API is async: submit, then poll until Ready.
    FLUX_POLL_INTERVAL = _float("FLUX_POLL_INTERVAL", 0.5)
    FLUX_TIMEOUT = _int("FLUX_TIMEOUT", 180)
    FLUX_SUBMIT_TIMEOUT = _int("FLUX_SUBMIT_TIMEOUT", 30)
    FLUX_DOWNLOAD_TIMEOUT = _int("FLUX_DOWNLOAD_TIMEOUT", 60)

    # A generation costs a base price plus a surcharge per reference image,
    # since each one adds real encoding work.
    TOKENS_PER_GENERATION = _int("TOKENS_PER_GENERATION", 5)
    TOKENS_PER_REFERENCE = _int("TOKENS_PER_REFERENCE", 1)

    # Scene plus cutouts in one request. Mirrors spring.servlet.multipart in
    # generation-backend, which caps a single upload at 16MB.
    MAX_UPLOAD_BYTES = _int("MAX_UPLOAD_MB", 40) * 1024 * 1024

    HTTP_TIMEOUT = _int("HTTP_TIMEOUT", 5)
    HTTP_RETRIES = _int("HTTP_RETRIES", 2)

    # A balance changes on every spend, so cache it barely at all — this only
    # collapses the burst of reads a single page load makes.
    BALANCE_CACHE_TTL = _int("BALANCE_CACHE_TTL", 3)


class TestConfig(Config):
    AUTH_BASE_URL = "http://auth.test"
    IMAGE_BASE_URL = "http://images.test"
    JWKS_URI = "http://auth.test/.well-known/jwks.json"
    JWT_ISSUER = "http://auth.test"
    JWT_AUDIENCE = "generation-backend"
    BALANCE_CACHE_TTL = 0
    HTTP_RETRIES = 0
    FLUX_API_KEY = "test-key"
    FLUX_API_URL = "https://flux.test/v1/flux-2-pro"
    FLUX_POLL_INTERVAL = 0.01  # no real waiting in tests
    FLUX_TIMEOUT = 1
