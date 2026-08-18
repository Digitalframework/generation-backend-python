# ImageGenBackend

A Flask service that verifies the access tokens AuthAndSecurity issues and checks
the caller's token balance before doing paid work.

## How it fits together

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

Production: `waitress-serve --port=5000 wsgi:app` on Windows, `gunicorn -w 4 wsgi:app`
elsewhere.

## Endpoints

All but `/api/health` need `Authorization: Bearer <token from :8080/api/token>`.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/health` | Liveness, plus whether the JWKS endpoint answers |
| GET | `/api/me` | Identity from the token — no network call |
| GET | `/api/tokens/balance` | The caller's balance |
| GET | `/api/tokens/check?amount=N` | Whether the caller can afford N |
| POST | `/api/tokens/spend` | Debit `{"amount": N, "reason": "…"}` |
| GET | `/api/admin/health` | Example role gate, `ADMIN` only |
| GET | `/api/generate/quote?references=N` | What a generation with N references would cost |
| POST | `/api/generate` | FLUX.2 generation from a prompt + N reference images |
| GET | `/api/compose/quote?objects=N` | What a composite of N objects would cost |
| POST | `/api/compose` | Place N cutouts into a room photo geometrically |

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
allows ten, and the local `diffusers` pipeline takes a list of ten — this is not
that). The prompt addresses them by index or descriptively, and combining both
works best:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  -F prompt="the sofa from image 1 and the lamp from image 2, in the room from image 3, photorealistic interior" \
  -F references=@sofa.png \
  -F references=@lamp.png \
  -F references=@room.png \
  -F aspect_ratio=16:9 -F seed=7 \
  http://localhost:5000/api/generate -o generated.png
```

Response is the PNG, with `X-Tokens-Spent`, `X-Token-Balance` and
`X-References-Used` headers.

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
  `FLUX_POLL_INTERVAL` until the status is `Ready`, then download. The whole
  round trip happens inside the request, so a worker is tied up for its
  duration — `FLUX_TIMEOUT` (180s) bounds that. Moving generation onto a queue
  and returning a job id is the fix if throughput matters.
- **The result URL is signed and expires after ten minutes**, so the image is
  downloaded immediately and served from here rather than the URL being passed
  back to the caller.
- **References are downscaled to `FLUX_REFERENCE_MAX_EDGE` (1024)** before being
  base64'd into the request body, where an unresized phone photo is megabytes on
  the wire for no benefit.
- Moderation comes back as a status (`Content Moderated` / `Request Moderated`),
  not an HTTP error, and maps to **422 `content_moderated`** here.

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

`harmonise` is unchanged and still single-region: with several objects the union
of their bounding boxes can span most of the frame, at which point the img2img
pass is re-generating the room rather than blending an edit into it. Harmonising
each object's own box would be the fix, and it is not done here.

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

## Tests

```bash
.venv/Scripts/python.exe -m pytest -q
```

25 tests. A real RSA key pair is generated per session and published as a JWK set,
so tokens are signed and verified exactly as in the running system — expiry, wrong
audience, wrong issuer, tampering and unknown `kid` are all covered against real
crypto. Only the JWKS fetch and the AuthAndSecurity HTTP calls are stubbed.
