"""FLUX.2 generation, over a composite of catalogue furniture in a room.

The endpoint runs four steps, in this order and for a reason:

    1. price the request and check the balance, so nothing expensive starts
       that the caller cannot pay for;
    2. fetch the furniture by id from generation-backend's catalogue —
       GET /api/images/batch for the metadata, then /{id}/content for the PNGs.
       The stored width_cm/height_cm are what let the pieces be placed at the
       right scale from an id alone;
    3. composite them into the room photograph with the perspective fit in
       app/composite.py — locally, no API call, so a bad geometry costs nothing;
    4. send that composite to FLUX.2 as its reference image, which is what
       turns a pasted-in cutout into a photograph.

Multipart:

    prompt          text; the composite goes in as "image 1"
    image_ids       catalogue ids, "3,7" or a JSON array. Required — this
                    endpoint always composites at least one catalogue piece
                    into the room before calling FLUX.
    placements      JSON array, one entry per id in the same order, all keys
                    optional — wall_id, wall_offset_m, wall_gap_m,
                    mount_height_m, width_m, height_m, depth_m, light_from,
                    shadow_core, shadow_ambient, glossy_floor.
                    Sizes default to catalogue measurements. Supported walls
                    determine positions and rotation, with footprint checks.
                    Shadow and reflection
                    default OFF: FLUX lights the composite, and a baked-in
                    ellipse under a fixed light direction competes with it.
                    Pass a positive shadow_core/shadow_ambient to draw one.
    user_height     person's height in metres; defaults to PERSON_HEIGHT_M
                    (1.75). Estimated camera height = user_height * 0.73,
                    assuming the first room photo was taken at chest height.
    rooms           up to three files of the same room; first is the main view
    scene           a single room photograph, accepted instead of rooms
    references      extra reference images, up to the API's ceiling

Room photos come from the multipart rooms/scene uploads, with
config.ROOM_IMAGE_PATH (assets/room.png) as the fallback when none are uploaded.

Tokens are checked before the API call and debited only once an image comes
back, so a failed, moderated or timed-out generation is free.

Asynchronous: POST /api/generate validates, checks the balance, stores the
uploads and answers 202 {jobId}. Steps 2-4 run later in run_generation(), on
whatever config.TASKS_MODE dispatches to; GET /api/jobs/<jobId> reports the
outcome (app/job_routes.py). The generation used to hold an HTTP request open
for up to several minutes, which timed out behind proxies and let concurrent
requests pile onto VGGT.
"""

import io
import json
import logging
import math
import re
import time
import uuid
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from flask import Blueprint, current_app, g, jsonify, request

from .auth import require_auth
from .composite import Placement, compose, load_cutout
from .errors import AppError, InsufficientTokens
from .flux import MAX_REFERENCES
from .job_store import QUEUED, public_view
from .logctx import job_context, stage
from .prompt_builder import build_prompt, reference_from_catalogue
from .storage import png_bytes

from .vggt_scene import solve_scene
from .placement_plan import plan_placements, PlacementConflict
from .scene_validation import bad_request as _bad
from .scene_validation import json_field, validate_placement

log = logging.getLogger(__name__)

bp = Blueprint("generate", __name__, url_prefix="/api")

ASPECT_RATIO = re.compile(r"^\d{1,2}:\d{1,2}$")
OUTPUT_FORMATS = ("png", "jpeg")


def _client():
    return current_app.extensions["auth_client"]


def _flux():
    return current_app.extensions["flux"]


def _images():
    return current_app.extensions["images"]


def _config():
    return current_app.config["APP_CONFIG"]


def _cost(reference_count, catalogue_count=0):
    """Base price, plus every reference that reaches FLUX, plus each piece placed.

    `reference_count` counts the composite when there is one: it is a reference
    image on the wire like any other, and the per-object part pays for the
    fetch, the cutout and the warp that produced it.
    """
    config = _config()
    return (
        config.TOKENS_PER_GENERATION
        + reference_count * config.TOKENS_PER_REFERENCE
        + catalogue_count * config.TOKENS_PER_OBJECT
    )


def _int_field(name, minimum, maximum):
    raw = request.form.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        raise _bad(f"`{name}` must be an integer") from None
    if not minimum <= value <= maximum:
        raise _bad(f"`{name}` must be between {minimum} and {maximum}")
    return value


def _image_ids():
    """`image_ids` as a list of ints — "3,7", "3 7" or a JSON array all work."""
    raw = request.form.get("image_ids")
    if raw is None or not raw.strip():
        raise _bad("`image_ids` is required — at least one catalogue id to place")

    try:
        # A JSON array is accepted because `placements` next to it is JSON too;
        # a bare "3,7" is not JSON at all, and is the form a query string wants.
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    values = parsed if isinstance(parsed, list) else re.split(r"[,\s]+", raw.strip())

    ids = []
    for value in values:
        try:
            image_id = int(value)
        except (TypeError, ValueError):
            raise _bad(f"`image_ids` must be whole numbers, got {value!r}") from None
        if image_id <= 0:
            raise _bad("`image_ids` must be positive")
        if image_id not in ids:  # placing the same piece twice is a placements job
            ids.append(image_id)

    if not ids:
        raise _bad("`image_ids` is required — at least one catalogue id to place")

    limit = _config().MAX_CATALOGUE_IMAGES
    if len(ids) > limit:
        raise _bad(f"at most {limit} catalogue images per generation, got {len(ids)}")
    return ids


MAX_ROOMS = 3


def _scenes():
    """Every view of the room, first one first.

    `rooms` takes up to three photographs of the *same* room. All of them go to
    VGGT, which reconstructs far better from several viewpoints than from one;
    only the first is composited into and sent to FLUX. The names are positional
    - room1.png, room2.png, room3.png - so the prompt and the geometry refer to
    the same frame regardless of what the client called its uploads.

    `scene` remains accepted as the single-image form.
    """
    uploads = request.files.getlist("rooms")
    if len(uploads) > MAX_ROOMS:
        raise _bad(
            f"at most {MAX_ROOMS} room images (room1.png … room{MAX_ROOMS}.png), "
            f"got {len(uploads)}"
        )

    if uploads:
        images = []
        for index, upload in enumerate(uploads, start=1):
            try:
                image = Image.open(upload.stream)
                image.load()
            except UnidentifiedImageError:
                raise _bad(
                    f"rooms[{index - 1}] ({upload.filename}) is not a readable image"
                ) from None
            images.append(image)
        return images, [f"room{i}.png" for i in range(1, len(images) + 1)]

    upload = request.files.get("scene")
    if upload is not None:
        try:
            image = Image.open(upload.stream)
            image.load()
        except UnidentifiedImageError:
            raise _bad("`scene` is not a readable image") from None
        return [image], ["room1.png"]

    path = getattr(_config(), "ROOM_IMAGE_PATH", Path("assets/room.png"))
    try:
        image = Image.open(path)
        image.load()
    except FileNotFoundError:
        raise AppError(
            f"no room image at {path}. Put a room photograph there, point "
            "ROOM_IMAGE_PATH at one, or send `rooms` files with the request.",
            status=503,
            code="room_unavailable",
        ) from None
    except (UnidentifiedImageError, OSError) as exc:
        raise AppError(
            f"the room image at {path} could not be read: {exc}",
            status=503,
            code="room_unavailable",
        ) from None
    return [image], ["room1.png"]


def _size_m(spec, metadata, key, cm_key, label):
    """Real-world size in metres: the caller's override, else the catalogue's."""
    if spec.get(key) is not None:
        return float(spec[key])

    centimetres = metadata.get(cm_key)
    try:
        metres = float(centimetres) / 100.0
    except (TypeError, ValueError):
        metres = 0.0
    if not math.isfinite(metres) or metres <= 0:
        raise _bad(
            f"catalogue image {label} has no usable {cm_key}, so its size is "
            f"unknown — pass placements[].{key} for it"
        )
    return metres


def _catalogue_placements(fetched, specs, geometry):
    placements = []
    for index, (metadata, content) in enumerate(fetched):
        spec = specs[index] if index < len(specs) else {}
        label = f"{metadata.get('name') or 'image'} (id {metadata.get('id')})"

        try:
            cutout = load_cutout(io.BytesIO(content))
        except ValueError as exc:
            # The catalogue's picture, not the caller's request: the piece has
            # no separable foreground, which no change to the call would fix.
            raise AppError(
                f"could not cut {label} out of its background: {exc}",
                status=422,
                code="cutout_failed",
            ) from None
        except ImportError:
            raise AppError(
                f"{label} is a fully opaque image and background removal is not "
                "installed here — `pip install rembg onnxruntime`",
                status=503,
                code="cutout_unavailable",
            ) from None

        if spec.get("center_x") is not None or spec.get("forward_offset_px") is not None:
            raise _bad("Generation now uses wall-relative placement; replace center_x/"
                       "forward_offset_px with wall_id, wall_offset_m and wall_gap_m.")
        depth = spec.get("depth_m", metadata.get("depthCm"))
        if depth is not None:
            try:
                depth = float(depth) if spec.get("depth_m") is not None else float(depth) / 100
            except (TypeError, ValueError):
                raise _bad(f"{label} needs a positive footprint depth") from None
            if not math.isfinite(depth) or depth <= 0:
                raise _bad(f"{label} needs a positive footprint depth")

        # Only the keys the caller actually sent. Anything omitted keeps the
        # Placement default, and those leave shadow and reflection off: FLUX
        # relights the composite, so baking lighting in only fights it.
        lighting = {}
        if spec.get("light_from") is not None:
            lighting["light_from"] = spec["light_from"]
        for key in ("shadow_core", "shadow_ambient"):
            if spec.get(key) is not None:
                lighting[key] = float(spec[key])
        if spec.get("glossy_floor") is not None:
            lighting["glossy_floor"] = bool(spec["glossy_floor"])

        placements.append(
            Placement(
                image=cutout,
                center_x=0.0,  # unused: the wall planner supplies a 3D baseline
                width_m=_size_m(spec, metadata, "width_m", "widthCm", label),
                height_m=_size_m(spec, metadata, "height_m", "heightCm", label),
                depth_size_m=depth,
                label=label,
                **lighting,
            )
        )
    return plan_placements(placements, specs, geometry)


# The ratios the API accepts, as (w, h). An output shaped differently from the
# room photograph forces the model to invent wall, window or door to fill the
# edges the photograph does not cover, so the nearest of these is picked.
ASPECT_CHOICES = ((21, 9), (16, 9), (3, 2), (4, 3), (5, 4), (1, 1),
                  (4, 5), (3, 4), (2, 3), (9, 16), (9, 21))


def _aspect_ratio_for(image):
    """The listed ratio closest to `image`'s own shape, as "16:9"."""
    ratio = image.width / image.height
    w, h = min(ASPECT_CHOICES, key=lambda c: abs(c[0] / c[1] - ratio))
    return f"{w}:{h}"


def _user_camera_height():
    """First-photo scale assumes a chest-held camera; same rule in both routes."""
    geometry_spec = json_field("geometry", required=False)
    if geometry_spec is not None:
        if not isinstance(geometry_spec, dict):
            raise _bad("`geometry` must be an object")
        if geometry_spec:
            raise _bad("Generation now uses `user_height` in metres instead of `geometry`; "
                       "camera height is estimated for a chest-held camera.")
    raw = request.form.get("user_height")
    try:
        height = float(raw if raw is not None else getattr(_config(), "PERSON_HEIGHT_M", 1.75))
    except (TypeError, ValueError):
        raise _bad("`user_height` must be a number in metres") from None
    if not math.isfinite(height) or not 0 < height <= 3:
        raise _bad("`user_height` must be between 0 and 3 metres")
    ratio = getattr(_config(), "CHEST_HEIGHT_RATIO", 0.73)
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise AppError("CHEST_HEIGHT_RATIO must be between 0 and 1", status=503,
                       code="geometry_unavailable")
    return height * ratio


def _composite(ids, specs, scenes, names, camera_height, token):
    """Fetch `ids` from the catalogue and paste them into the room.

    Returns `(result, geometry, names, fetched)`: the prompt builder needs the
    same geometry and the same catalogue rows the composite was built from, and
    re-fetching them would risk the prompt and the picture disagreeing.

    All room views condition VGGT. Catalogue cutouts are warped into the
    first view, and that composite becomes FLUX's first reference image.

    Takes everything as arguments rather than from `request`: it runs inside
    the plan request, but for /generate in a job worker with no request at all.
    """
    with stage(log, "catalogue fetch", ids=ids):
        fetched = _images().fetch(token, ids)
    try:
        with stage(log, "room reconstruction (VGGT)", views=len(scenes),
                   camera_height_m=round(camera_height, 3)):
            geometry = solve_scene(scenes, camera_height_m=camera_height)
    except ValueError as exc:
        raise AppError(
            f"the room could not be reconstructed: {exc}",
            status=422, code="scene_unsolvable",
        ) from None
    except ImportError as exc:
        raise AppError(
            "room reconstruction dependencies are unavailable",
            status=503, code="geometry_unavailable",
        ) from exc

    try:
        with stage(log, "placement planning", pieces=len(fetched)):
            placements = _catalogue_placements(fetched, specs, geometry)
    except PlacementConflict as exc:
        raise AppError(str(exc), status=422, code="placement_conflict",
                       details={"issues": exc.issues}) from None
    try:
        with stage(log, "composite"):
            result = compose(scenes[0], placements, geometry)
    except ValueError as exc:
        raise _bad(str(exc)) from None
    return result, geometry, names, fetched


def _placement_specs(ids):
    specs = json_field("placements", required=False) or []
    if not isinstance(specs, list) or len(specs) > len(ids):
        raise _bad("`placements` must be an array with at most one entry per image_id")
    for index, spec in enumerate(specs):
        validate_placement(spec, index, require_size=False)
    return specs


@bp.post("/generate/plan")
@require_auth
def placement_preview():
    """Reconstruct and compose without a FLUX call or token debit."""
    ids = _image_ids()
    camera_height = _user_camera_height()
    specs = _placement_specs(ids)
    scenes, names = _scenes()
    result, geometry, _names, _fetched = _composite(
        ids, specs, scenes, names, camera_height, g.token
    )
    return jsonify(status="ready", camera_height_m=camera_height,
                   placements=result.placement_plan,
                   walls=[{"id": w.id, "origin_m": w.origin.tolist(),
                           "right_axis": w.right.tolist(), "inward_axis": w.inward.tolist(),
                           "start_m": w.start_m, "end_m": w.end_m,
                           "residual_m": w.residual_m} for w in geometry.walls],
                   limitations=["Geometric wall candidates; doors/windows are not segmented.",
                                "Scale assumes a chest-held camera at user_height * CHEST_HEIGHT_RATIO; cutouts remain flat guides."])


@bp.post("/generate")
@require_auth
def generate():
    """Validate, price, store the uploads and queue the job. Answers 202.

    Everything that can be wrong with the request is caught here, before the
    202, so the caller gets a 400 straight away rather than a failed job later.
    """
    started = time.perf_counter()
    user_id = (g.user or {}).get("id")
    room_files = request.files.getlist("rooms") or request.files.getlist("scene")
    log.info(
        "generate request: user=%s image_ids=%r rooms=%d references=%d "
        "user_height=%s prompt=%d chars",
        user_id, request.form.get("image_ids"), len(room_files),
        len(request.files.getlist("references")), request.form.get("user_height"),
        len((request.form.get("prompt") or "").strip()),
    )

    prompt = (request.form.get("prompt") or "").strip()
    if not prompt:
        raise _bad("`prompt` is required")

    ids = _image_ids()

    camera_height = _user_camera_height()  # validate before account lookups

    specs = _placement_specs(ids)

    uploads = request.files.getlist("references")
    # Image 1 guides layout; original catalogue photos preserve product identity.
    reference_count = 1 + len(ids) + len(uploads)
    if reference_count > MAX_REFERENCES:
        raise _bad(
            f"at most {MAX_REFERENCES} reference images via the API, got "
            f"{reference_count} — the composite, {len(ids)} product photo(s), "
            f"and {len(uploads)} upload(s)"
        )

    aspect_ratio = request.form.get("aspect_ratio") or None
    if aspect_ratio and not ASPECT_RATIO.match(aspect_ratio):
        raise _bad("`aspect_ratio` must look like '16:9'")

    output_format = (request.form.get("output_format") or "").lower() or None
    if output_format and output_format not in OUTPUT_FORMATS:
        raise _bad(f"`output_format` must be one of {', '.join(OUTPUT_FORMATS)}")

    seed = _int_field("seed", 0, 2**32 - 1)
    safety_tolerance = _int_field("safety_tolerance", 0, 6)

    cost = _cost(reference_count, len(ids))
    with stage(log, "balance check", user=user_id):
        balance = _client().get_balance(g.token)
    log.info("balance %d, cost %d (%d catalogue, %d reference upload(s))",
             balance, cost, len(ids), len(uploads))
    if balance < cost:
        raise InsufficientTokens(
            f"This generation costs {cost} tokens "
            f"({len(ids)} catalogue images, {len(uploads)} reference images); "
            f"balance is {balance}"
        )

    scenes, scene_names = _scenes()
    references = []
    for index, upload in enumerate(uploads):
        try:
            image = Image.open(upload.stream)
            image.load()
        except UnidentifiedImageError:
            raise _bad(
                f"references[{index}] ({upload.filename}) is not a readable image"
            ) from None
        references.append(image)

    job_id = uuid.uuid4().hex
    storage = current_app.extensions["storage"]
    inputs = {"rooms": [], "references": []}
    for kind, images in (("rooms", scenes), ("references", references)):
        for index, image in enumerate(images):
            key = f"{job_id}/{kind}/{index}.png"
            storage.put(key, png_bytes(image), "image/png")
            inputs[kind].append(key)

    params = {
        "prompt": prompt,
        "image_ids": ids,
        "placements": specs,
        "camera_height_m": camera_height,
        "scene_names": scene_names,
        "aspect_ratio": aspect_ratio,
        "output_format": output_format,
        "seed": seed,
        "safety_tolerance": safety_tolerance,
        "cost": cost,
    }
    jobs = current_app.extensions["jobs"]
    job = jobs.create({
        "id": job_id,
        "userId": (g.user or {}).get("id"),
        "status": QUEUED,
        # A JSON string, not a map: Firestore rejects nested arrays, and
        # placements can carry them.
        "params": json.dumps(params),
        "inputs": inputs,
        "attempts": 0,
    })
    with job_context(job_id):
        log.info(
            "job %s queued: mode=%s user=%s catalogue=%s rooms=%d references=%d "
            "cost=%d balance_before=%d",
            job_id, _config().TASKS_MODE, user_id, ids, len(scenes), len(references),
            cost, balance,
        )
        current_app.extensions["dispatcher"].dispatch(job_id, g.token)
    log.info("generate request answered 202 in %.2fs", time.perf_counter() - started)

    # Inline mode has already finished the job; report where it actually is.
    job = jobs.get(job_id) or job
    body = public_view(job)
    body["statusUrl"] = f"/api/jobs/{job_id}"
    body["cost"] = cost
    response = jsonify(body)
    response.status_code = 202
    response.headers["Location"] = body["statusUrl"]
    return response


def run_generation(params, scenes, uploads, token):
    """Steps 2-4 for one queued job: catalogue, composite, FLUX.

    Runs in a job worker, not a request. `params` is what generate() stored;
    `scenes` and `uploads` are the images it stored. Returns the image and the
    counts the job record reports. The token debit is the caller's job
    (app/job_runner.py), so a retry can redo it without regenerating.
    """
    ids, specs = params["image_ids"], params["placements"]

    # First, so the prompt can call it "image 1" — the room is what the
    # generation is conditioned on.
    composited, geometry, scene_names, fetched = _composite(
        ids, specs, scenes, params["scene_names"], params["camera_height_m"], token
    )
    references = [composited.image]

    for metadata, content in fetched:
        try:
            product = Image.open(io.BytesIO(content))
            product.load()
        except (UnidentifiedImageError, OSError):
            raise AppError(f"catalogue image {metadata.get('id')} is not readable",
                           status=422, code="reference_unreadable") from None
        references.append(product.convert("RGB"))

    references.extend(uploads)

    # Exact originals are images 2..N+1, in catalogue-id order.
    catalogue_refs = [
        reference_from_catalogue(metadata, specs[i] if i < len(specs) else None,
                                 image_index=i + 2)
        for i, (metadata, _content) in enumerate(fetched)
    ]
    full_prompt = build_prompt(
        params["prompt"],
        catalogue_refs,
        geometry=geometry,
        scene_name=scene_names[0],
    )

    identity_prompt = (
        "# IMAGE ROLES AND PRODUCT IDENTITY\n"
        "Image 1 is a rough placement guide: preserve its room architecture, camera "
        "and planned furniture positions. Its cutouts are flat approximations. "
        "Use the original product images listed below as the authority for furniture "
        "identity, silhouette, configuration, colour, materials and details. "
        "Reproduce those exact products at the planned positions, adapting only "
        "perspective and room lighting. Preserve each sofa's L-shaped or straight "
        "configuration and chaise side and proportions as shown in its original image.\n"
        + "\n".join(f"Image {i + 2}: {ref.filename} — {ref.furniture_type.label}"
                    for i, ref in enumerate(catalogue_refs)) + "\n\n"
    )

    if composited.placement_plan:
        full_prompt = ("# PLANNED COMPOSITION IN IMAGE 1\n"
                       "Preserve each planned position and orientation. Furniture width axes "
                       "follow their assigned walls; the floor supplies the vertical axis. "
                       "Keep each supplied footprint and full chaise projection. "
                       "Wall-mounted furniture uses the planned elevation.\n"
                       + json.dumps(composited.placement_plan) + "\n\n" + full_prompt)

    full_prompt = identity_prompt + full_prompt

    # Log the assembled prompt and a rough token estimate for diagnosis.
    log.info(
        "prompt for %s: %d chars, ~%d tokens, %d reference image(s)",
        scene_names[0],
        len(full_prompt),
        len(full_prompt) // 4,
        len(references),
    )
    log.info("=============== PROMPT START ===============")
    log.info(full_prompt)
    log.info("================ PROMPT END ================")

    aspect_ratio = params["aspect_ratio"] or _aspect_ratio_for(composited.image)
    with stage(log, "FLUX generation", references=len(references),
               aspect_ratio=aspect_ratio, seed=params["seed"]):
        image = _flux().generate(
            full_prompt,
            references=references,
            seed=params["seed"],
            aspect_ratio=aspect_ratio,
            output_format=params["output_format"],
            safety_tolerance=params["safety_tolerance"],
        )
    log.info("FLUX result %dx%d", image.width, image.height)

    return {
        "image": image,
        "references_used": len(references),
        "catalogue_images": len(ids),
    }


@bp.get("/generate/quote")
@require_auth
def quote():
    """Price a generation without running it."""

    def _count(name, maximum):
        raw = request.args.get(name, "0")
        try:
            value = int(raw)
        except ValueError:
            raise _bad(f"`{name}` must be an integer") from None
        if not 0 <= value <= maximum:
            raise _bad(f"`{name}` must be between 0 and {maximum}")
        return value

    references = _count("references", MAX_REFERENCES)
    catalogue = _count("images", _config().MAX_CATALOGUE_IMAGES)
    if catalogue < 1:
        raise _bad("`images` must be at least 1 — /api/generate always requires image_ids")

    total_references = 1 + catalogue + references
    if total_references > MAX_REFERENCES:
        raise _bad(f"composite + product photos + uploaded references exceed {MAX_REFERENCES} images")
    cost = _cost(total_references, catalogue)
    balance = _client().get_balance(g.token)
    return jsonify(
        references=references,
        totalReferences=total_references,
        images=catalogue,
        cost=cost,
        balance=balance,
        sufficient=balance >= cost,
        apiConfigured=_flux().configured,
    )
