# ImageGenBackend

A Flask service that verifies the access tokens AuthAndSecurity issues and checks
the caller's token balance before doing paid work.

## How it fits together

### Local AMD GPU setup (Windows)

VGGT room reconstruction supports AMD ROCm through PyTorch's existing
`torch.cuda` interface; no `import rocm` or separate VGGT backend is needed.
The Windows setup uses Python **3.12**, PyTorch **2.9.1 + ROCm 7.2.1**,
and matching torchvision/ROCm SDK wheels. RX 9070 and RX 9070 XT are supported.
AMD specifies Windows 11 and the compatible 26.2.2 graphics driver for this
release. See the [AMD installation guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/windows/install-pytorch.html).

For a fresh environment (preserve an existing Python 3.10 environment first):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-rocm-windows.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.hip); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Always use `requirements-rocm-windows.txt` when reinstalling this AMD setup;
the generic requirements alone do not install AMD's Windows runtime. The
ROCm requirements are Windows/Python 3.12 only; Linux/NVIDIA deployments use
their platform-appropriate PyTorch build. Background removal remains on CPU.

`app/vggt_scene.py` selects `cuda` for AMD or NVIDIA and falls back to CPU
when GPU support is unavailable. AMD uses FP16 autocast; NVIDIA uses BF16
when supported, otherwise FP16. Model weights download on first inference.
Test scene reconstruction directly with an actual room photo:

```powershell
.\.venv\Scripts\python.exe -c "from app.vggt_scene import solve_scene; solve_scene('assets/room.png', device='cuda', verbose=True)"
```

The generation route currently bypasses VGGT/compositing for testing in
`app/generate_routes.py`; installing ROCm does not remove that bypass.

### Service connections

```
browser --- session cookie ---> AuthAndSecurity  (:8080)   signs RS256 tokens,
   |                                  |                     owns app_user.token_balance
   |  GET /api/token                  |
   |<---- JWT (15 min) ---------------+
   |                                  ^
   +---- Authorization: Bearer -----> ImageGenBackend (:5000)
                                      verifies via /.well-known/jwks.json
```

This service is a resource server, exactly like generation-backend on :8081. It
does the same job [app/auth.py](app/auth.py) that
`spring.security.oauth2.resourceserver.jwt.*` does over there: fetch the public
key by `kid`, check signature, `iss`, `aud` and `exp`, then map claims onto a
principal. The JWKS is cached and re-fetched only when a token names an unknown
`kid`, so rotating the signing key needs no change here.

The token carries `sub` (the `app_user` row UUID), `email` and `roles` — bare
names like `USER` and `ADMIN`, since the `ROLE_` prefix is Spring Security's own
convention and deliberately absent from the token.

## Two things are missing on the AuthAndSecurity side

**This service cannot read a balance yet.** Both gaps are in AuthAndSecurity, not
here — the Flask side is written and tested against the endpoints as configured
in [.env.example](.env.example), so it starts working the moment they exist.

### 1. There is no bearer-token filter chain

`SecurityConfig.securityFilterChain` configures `oneTimeTokenLogin` and
`oauth2Login` only. There is no `oauth2ResourceServer { jwt { } }`, so
`Authorization: Bearer …` is not an authentication mechanism there at all —
`anyRequest authenticated` bounces such a call to `/login` with a 302. That is
why [auth_client.py](app/auth_client.py) sends `allow_redirects=False` and reports
the redirect as a clear error instead of parsing the login page as JSON.

AuthAndSecurity needs a second `SecurityFilterChain`, ordered ahead of the
session one and matched to the API paths:

```kotlin
@Bean
@Order(1)
fun apiFilterChain(http: HttpSecurity, decoder: JwtDecoder): SecurityFilterChain {
    http {
        securityMatcher("/api/tokens/**")
        authorizeHttpRequests { authorize(anyRequest, authenticated) }
        oauth2ResourceServer { jwt { jwtDecoder = decoder } }
        // Stateless: nothing here reads a cookie, so there is no CSRF to defend.
        csrf { disable() }
        sessionManagement { sessionCreationPolicy = SessionCreationPolicy.STATELESS }
    }
    return http.build()
}
```

Note this makes AuthAndSecurity verify tokens it signed itself, so the `JwtDecoder`
can be built straight from the public key in `JwtConfig` rather than over HTTP.

### 2. There are no balance endpoints

`token_balance` is only ever read through `CurrentUser`, and only written by
`PaypalService` on a completed purchase. Nothing spends it. Two endpoints are
needed — the paths are configurable here via `BALANCE_PATH` and `SPEND_PATH`:

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/api/tokens/balance` | `{"balance": 250}` |
| POST | `/api/tokens/spend` | `{"balance": 245}` for a body of `{"amount": 5, "reason": "image"}` |

**The debit must happen there, not here.** A read-then-write across the wire races
with any concurrent spend and with the `user.tokenBalance += purchase.tokens` in
`PaypalService`. It wants to be one transaction on the row AuthAndSecurity owns —
a guarded `UPDATE app_user SET token_balance = token_balance - :amount WHERE
user_id = :id AND token_balance >= :amount`, answering 409 when it touches no
rows. This service already maps that 409 onto a 402.

### A third thing to decide: the audience

`app.jwt.audience` is a single `String`, currently `generation-backend`, and the
`aud` check is what stops one service replaying another's token. So either:

- this service accepts `generation-backend` (the default in
  [.env.example](.env.example)) — fine if it is replacing that service, wrong if
  both run, since then either can replay at the other; or
- `JwtProperties.audience` becomes a `List<String>`, or `issueFor` takes the
  audience as a parameter, and this service gets one of its own.

The second is the right answer once both services are running. Only `JWT_AUDIENCE`
changes here.

## Layout

| File | Purpose |
| --- | --- |
| [config.py](config.py) | Env-driven config, mirroring generation-backend's properties |
| [app/auth.py](app/auth.py) | JWKS fetch, RS256 verification, `require_auth` / `require_role` |
| [app/auth_client.py](app/auth_client.py) | Balance reads and spends against AuthAndSecurity |
| [app/routes.py](app/routes.py) | The API blueprint |
| [app/errors.py](app/errors.py) | Error types mapped to HTTP statuses |

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env
python wsgi.py
```

Production: `waitress-serve --port=5000 wsgi:app` on Windows, `gunicorn -w 1 --threads 8 --timeout 900 wsgi:app`
elsewhere.

## Endpoints

All but `/api/health` need `Authorization: Bearer <token from :8080/api/token>`.

| Method | URL | Description |
| --- | --- | --- |
| GET | `http://localhost:5000/api/health` | Liveness, plus whether the JWKS endpoint answers |
| GET | `http://localhost:5000/api/me` | Identity from the token — no network call |
| GET | `http://localhost:5000/api/tokens/balance` | The caller's balance |
| GET | `http://localhost:5000/api/tokens/check?amount=1` | Whether the caller can afford N |
| POST | `http://localhost:5000/api/tokens/spend` | Debit `{"amount": N, "reason": "…"}` |
| GET | `http://localhost:5000/api/generate/quote?references=0&images=1` | What that generation would cost |
| POST | `http://localhost:5000/api/generate` | Queues a FLUX.2 generation; answers 202 `{jobId}` |
| GET | `http://localhost:5000/api/jobs/<jobId>` | Job status; cost and new balance once `done` |
| GET | `http://localhost:5000/api/jobs/<jobId>/image` | The generated PNG |
| GET | `http://localhost:5000/api/admin/health` | Example role gate, `ADMIN` only |

Port 5000 is the default and what `docker-compose.yml` publishes. From another
container on the compose network the host is the service name, `imagegen:5000`;
from a container outside it, `host.docker.internal:5000`.

```bash
TOKEN=$(curl -s -b cookies.txt http://localhost:8080/api/token | jq -r .accessToken)
curl -H "Authorization: Bearer $TOKEN" http://localhost:5000/api/tokens/balance
```

## Generating with FLUX.2

[app/flux.py](app/flux.py) calls the hosted Black Forest Labs API — no GPU here,
billed per image. Put a key from [dashboard.bfl.ai](https://dashboard.bfl.ai) in
`FLUX_API_KEY`; without one the endpoint answers 503 `model_unavailable` without
calling out.

**Multi-reference is numbered fields, not a list**: `input_image`, then
`input_image_2` … `input_image_8`. Eight is the API ceiling (the playground
allows ten — this is not that). The prompt addresses them by index or descriptively, and combining both
works best:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  -F prompt="the sofa from image 1 and the lamp from image 2, in the room from image 3, photorealistic interior" \
  -F references=@sofa.png \
  -F references=@lamp.png \
  -F references=@room.png \
  -F aspect_ratio=16:9 -F seed=7 \
  http://localhost:5000/api/generate
# -> 202 {"jobId": "…", "status": "queued", "statusUrl": "/api/jobs/…", "cost": 9}

curl -H "Authorization: Bearer $TOKEN" http://localhost:5000/api/jobs/<jobId>
# -> {"status": "done", "imageUrl": "/api/jobs/<jobId>/image",
#     "tokensSpent": 9, "balance": 41, "referencesUsed": 4, "catalogueImages": 1}
curl -H "Authorization: Bearer $TOKEN" http://localhost:5000/api/jobs/<jobId>/image -o generated.png
```

The generation runs as a job — see [Generation jobs](#generation-jobs). Status
goes `queued` → `running` → `done` or `failed`; a failed job carries
`error.code` and `error.message` (e.g. `content_moderated`, `placement_conflict`).

| Field | | |
| --- | --- | --- |
| `prompt` | text, required | may refer to "image 1", "image 2", … |
| `references` | file, repeated | 0–8 images, inlined as base64 |
| `aspect_ratio` | e.g. `16:9` | default `1:1`. The API has no width/height |
| `output_format` | `png` or `jpeg` | |
| `seed` | int | for reproducibility |
| `safety_tolerance` | 0–6 | how strict the content filter is |

Costs `TOKENS_PER_GENERATION` (5) plus `TOKENS_PER_REFERENCE` (1) each — three
references is 8 tokens. Checked before the API call, debited only once an image
comes back, so a moderated, failed or timed-out generation is free.

Practical notes:

- **The API is asynchronous**: submit, get a `polling_url`, poll every
  `FLUX_POLL_INTERVAL` until the status is `Ready`, then download. That round
  trip happens in the job, not the request — `FLUX_TIMEOUT` (180s) bounds it.
- **The result URL is signed and expires after ten minutes**, so the image is
  downloaded immediately and served from here rather than the URL being passed
  back to the caller.
- **References are downscaled to `FLUX_REFERENCE_MAX_EDGE` (1024)** before being
  base64'd into the request body, where an unresized phone photo is megabytes on
  the wire for no benefit.
- Moderation comes back as a status (`Content Moderated` / `Request Moderated`),
  not an HTTP error, and maps to a failed job with `content_moderated` here.

## Generation jobs

`POST /api/generate` validates the request, checks the balance, stores the room
photos and references, and answers **202** straight away. The generation itself
(catalogue fetch → VGGT → composite → FLUX → debit) runs in a job. Where it runs
is one setting, `TASKS_MODE`:

| `TASKS_MODE` | Runs jobs in | Job records | Images | Use for |
| --- | --- | --- | --- | --- |
| `local` (default) | thread pool in this process | `var/jobs/<id>/job.json` | `var/jobs/<id>/…` | localhost, docker compose |
| `inline` | the request itself, before the 202 | same | same | tests, debugging |
| `cloud` | Google Cloud Tasks → `POST /internal/jobs/run` | Firestore | Cloud Storage | production |

Nothing besides `.env` changes between them. `JOB_STORE` (`file` | `firestore`)
and `STORAGE_BACKEND` (`local` | `gcs`) follow `TASKS_MODE` but can be set on
their own, e.g. to try the bucket from localhost.

**Local mode** needs nothing installed beyond `requirements.txt`'s web part.
`LOCAL_WORKER_THREADS` (2) jobs run at once; VGGT itself runs one forward pass
at a time per process (`vggt_scene._INFERENCE_LOCK`), so the second slot is for
a job waiting on FLUX. A restart drops jobs that were queued or running.

**Cloud mode**, one-time setup (region and names are examples):

```bash
gcloud tasks queues create generate-jobs --location=europe-west3 \
  --max-concurrent-dispatches=2 --max-attempts=3 --min-backoff=10s
gcloud firestore databases create --location=europe-west3
gcloud storage buckets create gs://innigo-generation-jobs --location=europe-west3
gcloud iam service-accounts create tasks-invoker
# the API's runtime account needs roles/cloudtasks.enqueuer, roles/datastore.user,
# roles/storage.objectAdmin and roles/iam.serviceAccountUser on tasks-invoker
```

Then set `TASKS_MODE=cloud`, `GCP_PROJECT`, `GCP_LOCATION`, `WORKER_URL`,
`TASKS_SERVICE_ACCOUNT` and `GCS_BUCKET` (see `.env.example`). The worker can be
the same Cloud Run service as the API or a second one with a GPU and
`--concurrency 1`; either way give it `--timeout 900`. `/internal/jobs/run`
accepts only an OIDC token from `TASKS_SERVICE_ACCOUNT` with the expected
audience, and answers 404 in any other mode.

Retries (cloud only): bad input, moderation, insufficient tokens and an expired
user token fail the job at once. FLUX or AuthAndSecurity trouble answers 503 so
Cloud Tasks retries, until attempt `TASKS_MAX_ATTEMPTS`, which is recorded as
final. If the image was generated but the debit failed, the retry only redoes
the debit, so FLUX is never paid twice for one job.

**Limitation — the user's token travels with the job.** The worker fetches the
catalogue and debits the balance with the caller's access token, which lives 15
minutes. It goes in the task body, never into Firestore. A job that waits longer
fails with `unauthorized` and costs nothing. A service credential on
AuthAndSecurity (debit on behalf of a user id) would remove the limit.

## Compositing

[app/composite.py](app/composite.py) places product cutouts into a room photo at
the right scale and perspective. Everything follows from one relation:

```
px_per_metre(x) = (floor_y(x) - horizon_y) / camera_height_m
```

The horizon comes from where the baseboard and crown molding meet, since they are
parallel in 3D. Camera height is measured by hand — `auto_geometry` can estimate
both with GeoCalib and metric depth, but expect 10–20% scale error, so it is for
batch throughput rather than precision.

`POST /api/compose` is multipart:

| Field | | |
| --- | --- | --- |
| `scene` | file | the room photograph |
| `objects` | file, repeated | one or more cutouts, up to 8 |
| `placements` | JSON array | one entry per object, **matched by position** |
| `geometry` | JSON object | `floor_line`, and `ceiling_line` or `horizon_y`, plus `camera_height_m` |

```bash
curl -H "Authorization: Bearer $TOKEN" \
  -F scene=@room.png \
  -F objects=@console.png -F objects=@lamp.png \
  -F 'placements=[
        {"center_x":615,"width_m":1.65,"height_m":0.76,"forward_offset_px":14},
        {"center_x":300,"width_m":0.40,"height_m":1.55,"forward_offset_px":40}]' \
  -F 'geometry={"floor_line":[[180,610],[890,655]],
                "ceiling_line":[[180,160],[890,100]],
                "camera_height_m":1.10}' \
  http://localhost:5000/api/compose -o composite.png
```

The response is the PNG, with `X-Tokens-Spent` and `X-Token-Balance` headers.

**Multiple objects share the room's geometry** — the horizon and camera height
are properties of the photograph, not of any object — but each gets its own
position, real-world size and forward offset. They are drawn far-to-near, sorted
by where their base meets the floor, with each object's shadow laid down
immediately before it. That ordering is what lets a nearer object occlude a
farther one and cast onto it.

Costs `TOKENS_PER_OBJECT` per cutout, checked before the work starts and debited
only after it succeeds, so a failed composite is free. `GET /api/compose/quote`
prices a request without doing any of it.

## Gating expensive work

```python
@bp.post("/generate")
@require_auth
@require_tokens(5)
def generate():
    image = run_model(request.json["prompt"])
    _client().spend(g.token, g.token_cost, reason="image")
    return jsonify(image=image)
```

`require_tokens` is a pre-check that fails fast with 402; the debit still goes
through `spend`, which is where the atomic decrement lives. Whether to charge
before or after the model runs is a real decision — charging first loses the
user's tokens on a model failure, charging after gives away work on a crash.

## Wall-based generation placement

FLUX receives image 1 as the rough room/layout guide, followed by the original
catalogue product photos in `image_ids` order (images 2 onward), followed by
optional uploaded `references`. The prompt explicitly maps each product to its
image number and treats the originals as the authority for product design.
The flat cutouts guide placement rather than defining the product's silhouette.
All transmitted images count toward reference pricing and the configured limit
of eight: one composite plus products plus uploads. Requests exceeding that limit
are rejected instead of dropping product references. `/generate/quote` treats
`references` as uploaded extras and returns `totalReferences` including products
and the composite.

`/api/generate` now follows VGGT camera/depth -> fitted floor and wall planes ->
metric furniture transforms -> warped rough composite -> FLUX. Objects follow
the chosen wall tangent instead of facing the camera. Automatic screen-column
spacing has been removed from generation.

Send `placements` in catalogue-id order. Width, height and **full footprint
depth** default to `widthCm`, `heightCm`, `depthCm`. For an L sofa, depth must
include the extended chaise; the planner currently uses a conservative rectangular
bounding footprint, not an L-shaped footprint. Missing depth blocks placement.

```json
[{"wall_id":"wall_1","wall_offset_m":0.0,"wall_gap_m":0.05,
  "width_m":3.0,"height_m":0.3,"depth_m":0.31,"mount_height_m":0.4}]
```

`wall_offset_m` is the signed distance along the returned wall's `right_axis`
from its `origin_m`. Default wall gap is 5 cm. `mount_height_m` is the bottom
edge's height above the floor (default 0). Generation rejects legacy `center_x`
and `forward_offset_px`; these cannot express a physical wall placement.

Both generation endpoints accept `user_height` as a **Text** multipart field in
metres, e.g. `1.70`. The estimated camera height passed to VGGT is
`user_height * CHEST_HEIGHT_RATIO` (default ratio `0.73`): 1.70 m gives 1.241 m.
This assumes the first room photo was taken with the camera at chest height;
it is a scale estimate, not a camera height measured by VGGT. Omitted height
uses `PERSON_HEIGHT_M` (default `1.75`, giving 1.2775 m). Both defaults are
environment settings. Do not send `geometry.camera_height_m`; the generation
endpoints reject the old geometry field. Internal VGGT geometry still uses
the calculated camera height in metres.

`POST /api/generate/plan` accepts the same room uploads, user height, image ids and
placements as generation, without requiring a prompt. It returns supported walls
and planned transforms, without calling FLUX or debiting tokens. Both endpoints
return HTTP 422 `placement_conflict` with `details.issues` if a product cannot be
placed by the planner. The frontend can then offer another wall, corrected
measurements/views, or exclusion from this generation. Exclusion means removing
that id from the next request, not deleting its catalogue record. No item is
silently removed or resized. Planning is deterministic for the same reconstruction
and inputs; this endpoint does not save an approved plan or implement a UI dialog.

The current planner tries large pieces first on observed wall support segments,
checks rectangular footprint collisions and requires a fully visible guide in
the first camera view. Failure means **no supported placement was found**, not
proof that the furniture cannot fit anywhere in the room. Door/window segmentation,
walkway constraints, semantic wall identification and complete 3D furniture rendering
are not implemented. Large furniture may resemble a wall geometrically. Scale
depends on the chest-height estimate, and a homography cannot reconstruct unseen
product sides. Wall-oriented cutouts are rough guides for FLUX, not exact final
geometry. The same planned transforms are included in the text prompt.

## Tests

```bash
.venv/Scripts/python.exe -m pytest -q
```

25 tests. A real RSA key pair is generated per session and published as a JWK set,
so tokens are signed and verified exactly as in the running system — expiry, wrong
audience, wrong issuer, tampering and unknown `kid` are all covered against real
crypto. Only the JWKS fetch and the AuthAndSecurity HTTP calls are stubbed.
