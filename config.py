import os
import tempfile
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
    # Reconstruct room dimensions for the FLUX prompt, even while furniture
    # compositing is bypassed. Set 0 to run the API without local ML inference.
    ROOM_GEOMETRY_ENABLED = os.getenv("ROOM_GEOMETRY_ENABLED", "1").lower() in ("1", "true", "yes")
    PERSON_HEIGHT_M = _float("PERSON_HEIGHT_M", 1.75)
    CHEST_HEIGHT_RATIO = _float("CHEST_HEIGHT_RATIO", 0.73)

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

    # --- Generation jobs ------------------------------------------------------
    # POST /api/generate answers 202 with a job id; the generation itself runs
    # elsewhere and GET /api/jobs/<id> reports on it. TASKS_MODE picks where:
    #
    #   local   a thread pool inside this process. The default, and all that
    #           localhost needs: no Google account, no extra services.
    #   inline  runs before the 202 is sent. Tests and step-through debugging.
    #   cloud   Google Cloud Tasks POSTs each job to WORKER_URL/internal/jobs/run.
    #
    # The job store and file storage default to match: on disk for local and
    # inline, Firestore and Cloud Storage for cloud. Either can be overridden,
    # e.g. TASKS_MODE=local with STORAGE_BACKEND=gcs to try the bucket locally.
    TASKS_MODE = os.getenv("TASKS_MODE", "local").strip().lower()
    _CLOUD = TASKS_MODE == "cloud"

    # How many generations run at once in local mode. VGGT itself still runs
    # one at a time (vggt_scene._INFERENCE_LOCK); the second slot lets one job
    # wait on FLUX while the next one reconstructs its room.
    LOCAL_WORKER_THREADS = _int("LOCAL_WORKER_THREADS", 2)

    # file | firestore. The file store keeps one JSON per job next to its
    # images, so every gunicorn worker on the machine sees the same jobs.
    JOB_STORE = (os.getenv("JOB_STORE") or ("firestore" if _CLOUD else "file")).lower()
    FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "generation_jobs")

    # local | gcs. Uploaded rooms, references and the result image.
    STORAGE_BACKEND = (os.getenv("STORAGE_BACKEND") or ("gcs" if _CLOUD else "local")).lower()
    LOCAL_STORAGE_DIR = Path(os.getenv("LOCAL_STORAGE_DIR", "var/jobs"))
    GCS_BUCKET = os.getenv("GCS_BUCKET") or None

    # --- Google Cloud Tasks (TASKS_MODE=cloud only) ---------------------------
    GCP_PROJECT = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or None
    GCP_LOCATION = os.getenv("GCP_LOCATION", "europe-west3")
    TASKS_QUEUE = os.getenv("TASKS_QUEUE", "generate-jobs")
    # Base URL of the service that runs jobs — this same app, deployed as the
    # worker. The task is sent to WORKER_URL + /internal/jobs/run.
    WORKER_URL = (os.getenv("WORKER_URL") or "").rstrip("/") or None
    # Service account Cloud Tasks signs its OIDC token as. The worker endpoint
    # accepts only tokens from this account.
    TASKS_SERVICE_ACCOUNT = os.getenv("TASKS_SERVICE_ACCOUNT") or None
    # The `aud` of that token. Defaults to WORKER_URL, which is what Cloud Run
    # expects when it checks the token itself.
    TASKS_OIDC_AUDIENCE = os.getenv("TASKS_OIDC_AUDIENCE") or WORKER_URL
    # Must equal the queue's --max-attempts: on the last attempt a transient
    # failure is recorded as final instead of being handed back for a retry.
    TASKS_MAX_ATTEMPTS = _int("TASKS_MAX_ATTEMPTS", 3)
    # How long Cloud Tasks waits for one attempt. VGGT + FLUX (FLUX_TIMEOUT)
    # has to fit inside it; Cloud Tasks allows up to 1800s.
    TASKS_DISPATCH_DEADLINE = _int("TASKS_DISPATCH_DEADLINE", 900)


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
    # Jobs finish before POST /api/generate returns, so a test can assert on
    # the result straight away. Storage goes to a per-test directory (conftest).
    TASKS_MODE = "inline"
    _CLOUD = False
    JOB_STORE = "file"
    STORAGE_BACKEND = "local"
    LOCAL_STORAGE_DIR = Path(tempfile.mkdtemp(prefix="innigo-test-jobs-"))
