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
                    optional — center_x, forward_offset_px, width_m, height_m,
                    light_from, shadow_core, shadow_ambient, glossy_floor.
                    Sizes default to the catalogue's measurements and positions
                    to an even spread along the floor. Shadow and reflection
                    default OFF: FLUX lights the composite, and a baked-in
                    ellipse under a fixed light direction competes with it.
                    Pass a positive shadow_core/shadow_ambient to draw one.
    geometry        JSON room description; defaults to the config's ROOM_*
                    fractions, measured for the bundled room.png
    scene           a room photograph, overriding room.png for this request
    references      extra reference images, up to the API's ceiling

The room is a file on disk (config.ROOM_IMAGE_PATH, assets/room.png) rather
than a catalogue entry: rooms are not uploaded anywhere yet, and this keeps the
endpoint testable end to end until they are.

Tokens are checked before the API call and debited only once an image comes
back, so a failed, moderated or timed-out generation is free.
"""

import io
import json
import re
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from flask import Blueprint, current_app, g, jsonify, request, send_file

from .auth import require_auth
from .composite import Placement, compose, load_cutout
from .errors import AppError, InsufficientTokens
from .flux import MAX_REFERENCES
from .prompt_builder import build_prompt, reference_from_catalogue
from .vggt_scene import solve_scene
from .scene_validation import bad_request as _bad
from .scene_validation import build_geometry, json_field, validate_placement

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


def _auto_center_x(index, count, geometry):
    """Spread the pieces evenly across the reconstructed floor, left to right.

    n objects go at 1/(n+1), 2/(n+1) … of the span, which keeps them off both
    walls and apart from each other without the caller positioning anything.
    The span is where VGGT actually found floor, not the whole image width.
    """
    x_left, x_right = geometry.floor_span
    return x_left + (x_right - x_left) * (index + 1) / (count + 1)


def _size_m(spec, metadata, key, cm_key, label):
    """Real-world size in metres: the caller's override, else the catalogue's."""
    if spec.get(key) is not None:
        return float(spec[key])

    centimetres = metadata.get(cm_key)
    try:
        metres = float(centimetres) / 100.0
    except (TypeError, ValueError):
        metres = 0.0
    if metres <= 0:
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

        center_x = spec.get("center_x")
        if center_x is None:
            center_x = _auto_center_x(index, len(fetched), geometry)

        forward_offset = spec.get("forward_offset_px")
        if forward_offset is None:
            forward_offset = getattr(_config(), "ROOM_FORWARD_OFFSET", 0.0) * geometry.size[1]

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
                center_x=float(center_x),
                width_m=_size_m(spec, metadata, "width_m", "widthCm", label),
                height_m=_size_m(spec, metadata, "height_m", "heightCm", label),
                forward_offset_px=float(forward_offset),
                label=label,
                **lighting,
            )
        )
    return placements


def _composite(ids, specs):
    """Fetch `ids` from the catalogue and paste them into the room.

    Returns `(result, geometry, fetched)`: the prompt builder needs the same
    geometry and the same catalogue rows the composite was built from, and
    re-fetching them would risk the prompt and the picture disagreeing.
    """
    fetched = _images().fetch(g.token, ids)

    scenes, names = _scenes()
    geometry_spec = json_field("geometry", required=False)
    camera_height_m = build_geometry(geometry_spec) if geometry_spec else None
    if camera_height_m is None:
        camera_height_m = getattr(_config(), "ROOM_CAMERA_HEIGHT_M", 1.10)

    # Every view goes in; only the first comes back out.
    try:
        geometry = solve_scene(scenes, camera_height_m=camera_height_m)
    except ValueError as exc:
        raise AppError(
            f"the room could not be reconstructed: {exc}",
            status=422,
            code="scene_unsolvable",
        ) from None
    except ImportError as exc:
        raise AppError(
            f"room reconstruction is unavailable here: {exc}",
            status=503,
            code="geometry_unavailable",
        ) from None

    placements = _catalogue_placements(fetched, specs, geometry)
    try:
        return compose(scenes[0], placements, geometry), geometry, names, fetched
    except ValueError as exc:
        # An object that cannot be placed on the reconstructed floor —
        # above the horizon, or behind the camera. The caller's numbers.
        raise _bad(str(exc)) from None


@bp.post("/generate")
@require_auth
def generate():
    prompt = (request.form.get("prompt") or "").strip()
    if not prompt:
        raise _bad("`prompt` is required")

    ids = _image_ids()

    specs = json_field("placements", required=False) or []
    if not isinstance(specs, list):
        raise _bad("`placements` must be a JSON array")
    if len(specs) > len(ids):
        raise _bad(
            f"{len(ids)} image_ids but {len(specs)} placements — they are "
            "matched by position, so there cannot be more placements than ids"
        )
    for index, spec in enumerate(specs):
        validate_placement(spec, index, require_size=False)

    uploads = request.files.getlist("references")
    # The composite itself goes to FLUX as a reference, so it takes one slot.
    reference_count = len(uploads) + 1
    if reference_count > MAX_REFERENCES:
        raise _bad(
            f"at most {MAX_REFERENCES} reference images via the API, got "
            f"{reference_count} including the composite"
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
    balance = _client().get_balance(g.token)
    if balance < cost:
        raise InsufficientTokens(
            f"This generation costs {cost} tokens "
            f"({len(ids)} catalogue images, {len(uploads)} reference images); "
            f"balance is {balance}"
        )

    # First, so the prompt can call it "image 1" — the room with the
    # furniture already in place is what the generation is conditioned on.
    composited, geometry, scene_names, fetched = _composite(ids, specs)
    references = [composited.image]

    for index, upload in enumerate(uploads):
        try:
            image = Image.open(upload.stream)
            image.load()
        except UnidentifiedImageError:
            raise _bad(
                f"references[{index}] ({upload.filename}) is not a readable image"
            ) from None
        references.append(image)

    # The catalogue pieces are composited into image 1 rather than sent as
    # separate references, so they carry no image index of their own.
    catalogue_refs = [
        reference_from_catalogue(metadata, specs[i] if i < len(specs) else None)
        for i, (metadata, _content) in enumerate(fetched)
    ]
    full_prompt = build_prompt(
        prompt,
        catalogue_refs,
        geometry=geometry,
        scene_name=scene_names[0],
    )

    image = _flux().generate(
        full_prompt,
        references=references,
        seed=seed,
        aspect_ratio=aspect_ratio or _config().FLUX_ASPECT_RATIO,
        output_format=output_format,
        safety_tolerance=safety_tolerance,
    )

    # Only now: a moderated, failed or timed-out generation costs nothing.
    remaining = _client().spend(g.token, cost, reason=f"flux2:{len(references)}ref")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = send_file(buffer, mimetype="image/png", download_name="generated.png")
    response.headers["X-Tokens-Spent"] = str(cost)
    response.headers["X-Token-Balance"] = str(remaining)
    response.headers["X-References-Used"] = str(len(references))
    response.headers["X-Catalogue-Images"] = str(len(ids))
    return response


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

    # The composite the catalogue images become is itself a reference.
    cost = _cost(references + 1, catalogue)
    balance = _client().get_balance(g.token)
    return jsonify(
        references=references,
        images=catalogue,
        cost=cost,
        balance=balance,
        sufficient=balance >= cost,
        apiConfigured=_flux().configured,
    )
