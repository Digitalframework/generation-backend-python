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
    edits           optional JSON array of positioned placements, any number up
                    to MAX_EDITS, several may use the same catalogue piece:
                    {"image_id": 7, "position": [x, y],
                     "remove": [[[x, y], ...], ...]}.
                    `image_id` must be one of image_ids. `position` is where
                    that piece's front edge meets the floor, normalised 0..1
                    on the first room photo — one copy of the piece stands
                    there. `remove` are outlines (from /api/segment_selection)
                    of what it replaces; they are erased from the photo before
                    compositing. Pieces with at least one edit are not wall-
                    planned; image_ids without any edit are, as before. This is
                    how "replace these two armchairs with the selected chair,
                    one moved over there" reaches the backend.
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
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Blueprint, current_app, g, jsonify, request

from .auth import require_auth
from .composite import Placement, compose, load_cutout
from .errors import AppError, InsufficientTokens
from .flux import MAX_REFERENCES
from .job_store import QUEUED, public_view
from .logctx import job_context, stage
from .prompt_builder import (build_prompt, intro_block, reference_from_catalogue,
                             resolve_type)
from .storage import png_bytes

from .vggt_scene import solve_scene
from . import sam_selection
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


def _catalogue_placements(fetched, specs, geometry, edits=()):
    placements = []
    for index, (metadata, content) in enumerate(fetched):
        spec = specs[index] if index < len(specs) else {}
        label = f"{metadata.get('name') or 'image'} (id {metadata.get('id')})"
        # The furniture type drives the layout rules in placement_plan.py: a
        # coffee table goes in front of the sofa, not against a wall.
        try:
            kind = resolve_type(spec.get("type"), metadata,
                                spec.get("name") or metadata.get("name")).slug
        except ValueError as exc:
            raise _bad(f"placements[{index}].type: {exc}") from None

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
                kind=kind,
                **lighting,
            )
        )
    # Every edit is one copy of its piece, standing where the user put it.
    # Pieces nobody positioned go to the wall planner as before.
    # Ids compared as text: the catalogue's JSON may carry them either way.
    index_of = {str(metadata.get("id")): i for i, (metadata, _content) in enumerate(fetched)}
    positioned = []
    for edit in edits:
        index = index_of.get(str(edit["image_id"]))
        if index is None:
            raise AppError(f"catalogue image {edit['image_id']} was not returned by the "
                           "catalogue", status=422, code="reference_unreadable")
        item = replace(placements[index])  # shares the cutout, own position
        _place_at(item, edit["position"], geometry)
        positioned.append(item)

    used = {index_of.get(str(edit["image_id"])) for edit in edits}
    planned = [i for i in range(len(placements)) if i not in used]
    if planned:
        try:
            plan_placements([placements[i] for i in planned],
                            [specs[i] if i < len(specs) else {} for i in planned], geometry)
        except PlacementConflict as exc:
            for issue in exc.issues:  # report indices into image_ids, not the subset
                if "index" in issue:
                    issue["index"] = planned[issue["index"]]
            raise
    return [placements[i] for i in planned] + positioned


def _place_at(item, position, geometry):
    """Stand `item` on the floor at an image point, facing the camera's way.

    `position` is normalised (x, y) on the first photo: the bottom centre of
    the moved selection, i.e. where the piece's front edge meets the floor.
    The ray through that pixel hits the reconstructed floor plane; that point
    is the front baseline the warp builds the upright rectangle from. The
    width axis follows the nearest wall, so a chair placed near a wall lines up
    with it; with no walls, it follows the camera's horizontal axis.
    """
    x = position[0] * geometry.size[0]
    y = position[1] * geometry.size[1]
    try:
        base = geometry.floor_point(x, y) / geometry.units_per_metre
    except ValueError:
        raise AppError(
            f"{item.label} was moved to a spot that is not on the floor — "
            "drag it lower in the picture",
            status=422, code="position_off_floor",
        ) from None

    up = np.asarray(geometry.normal, dtype=float)
    if geometry.walls:
        nearest = min(geometry.walls, key=lambda w: abs((base - w.origin) @ w.inward))
        right = np.asarray(nearest.right, dtype=float)
    else:
        right = np.array([1.0, 0.0, 0.0])
    right = right - (right @ up) * up
    right /= np.linalg.norm(right)

    item.base_m, item.right_axis = base, right
    item.wall_id, item.mount_height_m = None, 0.0


# Longest side the old furniture is erased at. Telea inpainting costs grow with
# the hole's area; FLUX repaints the region anyway, so a smaller pass suffices.
ERASE_LONG_SIDE = 1024


def _erase(scene, edits):
    """The room with the replaced furniture painted out, or `scene` unchanged.

    Fills each `remove` outline (grown by ~1% so the old piece's edge goes
    too) from its surroundings with OpenCV's Telea inpainting. Crude, but
    local and instant; the result goes to FLUX as image 1, which is told to
    render those areas as empty room.
    """
    polygons = [poly for edit in edits or () for poly in edit.get("remove") or ()]
    if not polygons:
        return scene

    import cv2

    rgb = np.asarray(scene.convert("RGB"))
    height, width = rgb.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    for polygon in polygons:
        points = np.round(np.asarray(polygon, dtype=float) * [width, height]).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    grow = max(2, round(0.01 * max(width, height)))
    mask = cv2.dilate(mask, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))

    scale = min(1.0, ERASE_LONG_SIDE / max(width, height))
    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = cv2.resize(rgb, small_size, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, small_size, interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(small, small_mask, max(3, round(grow * scale)), cv2.INPAINT_TELEA)
    filled = cv2.resize(filled, (width, height), interpolation=cv2.INTER_CUBIC)

    out = rgb.copy()
    out[mask > 0] = filled[mask > 0]
    return Image.fromarray(out)


# Positioned placements per generation. Each is a warp plus a FLUX instruction;
# beyond a dozen the prompt stops being followed reliably anyway.
MAX_EDITS = 12


def _edits(ids):
    """`edits` validated: positioned placements, each naming one of `ids`."""
    edits = json_field("edits", required=False)
    if edits is None:
        return []
    if not isinstance(edits, list) or len(edits) > MAX_EDITS:
        raise _bad(f"`edits` must be an array of at most {MAX_EDITS} placements")

    cleaned, total_points = [], 0
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise _bad(f"edits[{index}] must be an object")
        image_id = edit.get("image_id")
        if isinstance(image_id, bool) or not isinstance(image_id, int) or image_id not in ids:
            raise _bad(f"edits[{index}].image_id must be one of image_ids {ids}")
        position = edit.get("position")
        if not (isinstance(position, list) and len(position) == 2
                and all(_unit_number(v) for v in position)):
            raise _bad(f"edits[{index}].position must be [x, y] with values between 0 and 1")

        remove = edit.get("remove") or []
        if not isinstance(remove, list):
            raise _bad(f"edits[{index}].remove must be a list of outlines")
        outlines = []
        for polygon in remove:
            if not (isinstance(polygon, list) and len(polygon) >= 3 and all(
                isinstance(p, list) and len(p) == 2 and all(_unit_number(v) for v in p)
                for p in polygon
            )):
                raise _bad(f"edits[{index}].remove must hold outlines of at least three "
                           "[x, y] points between 0 and 1")
            total_points += len(polygon)
            outlines.append([[float(x), float(y)] for x, y in polygon])
        if total_points > 5000:
            raise _bad("`edits` outlines are too detailed — at most 5000 points in total")
        cleaned.append({"image_id": image_id, "position": [float(v) for v in position],
                        "remove": outlines})
    return cleaned


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


def _composite(ids, specs, scenes, names, camera_height, token, edits=None):
    """Fetch `ids` from the catalogue and paste them into the room.

    Returns `(result, geometry, names, fetched)`: the prompt builder needs the
    same geometry and the same catalogue rows the composite was built from, and
    re-fetching them would risk the prompt and the picture disagreeing.

    All room views condition VGGT. Catalogue cutouts are warped into the
    first view, and that composite becomes FLUX's first reference image.
    With `edits`, the furniture they replace is erased from that view first,
    and their pieces stand at the given floor positions (see `_place_at`).
    VGGT still sees the original photos: an inpainted patch is no evidence
    about where the floor and walls are.

    Takes everything as arguments rather than from `request`: it runs inside
    the plan request, but for /generate in a job worker with no request at all.
    """
    with stage(log, "catalogue fetch", ids=ids):
        fetched = _images().fetch(token, ids)
    try:
        geometry = _reconstruct(scenes, camera_height)
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
            placements = _catalogue_placements(fetched, specs, geometry, edits or [])
    except PlacementConflict as exc:
        raise AppError(str(exc), status=422, code="placement_conflict",
                       details={"issues": exc.issues}) from None
    canvas = scenes[0]
    if edits:
        with stage(log, "erase replaced furniture",
                   placements=len(edits), outlines=sum(len(e["remove"]) for e in edits)):
            canvas = _erase(scenes[0], edits)
    try:
        with stage(log, "composite"):
            result = compose(canvas, placements, geometry)
    except ValueError as exc:
        raise _bad(str(exc)) from None
    return result, geometry, names, fetched


def _reconstruct(scenes, camera_height):
    """VGGT over every view; if that finds no floor, over the main photo alone.

    Extra views usually sharpen the reconstruction, but views of the far side of
    the room share little with the main photo, and VGGT then reports too little
    confident floor depth in frame 0 to fit a plane. Only frame 0's geometry is
    used, so the main photo by itself is a valid answer rather than a failure.
    """
    try:
        with stage(log, "room reconstruction (VGGT)", views=len(scenes),
                   camera_height_m=round(camera_height, 3)):
            return solve_scene(scenes, camera_height_m=camera_height)
    except ValueError as exc:
        if len(scenes) < 2:
            raise
        log.warning("reconstruction from %d views failed (%s); retrying with the "
                    "main photo only", len(scenes), exc)
    with stage(log, "room reconstruction (VGGT), main photo only",
               camera_height_m=round(camera_height, 3)):
        return solve_scene(scenes[:1], camera_height_m=camera_height)


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
    edits = _edits(ids)
    scenes, names = _scenes()
    result, geometry, _names, _fetched = _composite(
        ids, specs, scenes, names, camera_height, g.token, edits
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
    edits = _edits(ids)

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
        "edits": edits,
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
    edits = params.get("edits") or []  # jobs queued before edits existed have none

    # First, so the prompt can call it "image 1" — the room is what the
    # generation is conditioned on.
    composited, geometry, scene_names, fetched = _composite(
        ids, specs, scenes, params["scene_names"], params["camera_height_m"], token, edits
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
    # Order follows how FLUX.2 reads: the start weighs most. Task and product
    # lines first, then anything special about this request, then the layout,
    # the standing directives, the user's words and the room size.
    sections = [intro_block(catalogue_refs, uploads=len(uploads))]

    if any(edit["remove"] for edit in edits):
        sections.append(
            "# REPLACED FURNITURE\n"
            "The furniture the user is replacing has been erased from image 1, and the "
            "erased areas look smudged. Show bare floor, wall and skirting there, "
            "continuing naturally from their surroundings; the catalogue products stand "
            "at their planned positions."
        )

    fixed = _fixed_positions_prompt(composited.placement_plan, geometry, fetched, ids)
    if fixed:
        sections.append(fixed.strip())

    sections.append(build_prompt(
        params["prompt"],
        catalogue_refs,
        geometry=geometry,
        scene_name=scene_names[0],
        plan=composited.placement_plan,
    ))
    full_prompt = "\n\n".join(section for section in sections if section)

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
        # Kept with the job (var/jobs/<id>/composite.png locally) to see what
        # FLUX was shown when its result disagrees with the placement.
        "composite": composited.image,
        "references_used": len(references),
        "catalogue_images": len(ids),
    }


def _unit_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)


def _selection_box():
    """`box` as (x0, y0, x1, y1), normalised 0..1, with x0 < x1 and y0 < y1."""
    box = json_field("box")
    if not (isinstance(box, list) and len(box) == 4 and all(_unit_number(v) for v in box)):
        raise _bad("`box` must be [x0, y0, x1, y1] with values between 0 and 1")
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        raise _bad("`box` must have x0 < x1 and y0 < y1")
    return tuple(float(v) for v in box)


def _selection_points():
    """Optional refinement clicks: `points` [[x, y], ...] and `point_labels` [1|0, ...]."""
    points = json_field("points", required=False)
    if points is None:
        return None, None
    if not (isinstance(points, list) and len(points) <= 32 and all(
        isinstance(p, list) and len(p) == 2 and all(_unit_number(v) for v in p)
        for p in points
    )):
        raise _bad("`points` must be at most 32 [x, y] pairs with values between 0 and 1")
    labels = json_field("point_labels", required=False)
    if labels is None:
        labels = [1] * len(points)
    if not (isinstance(labels, list) and len(labels) == len(points)
            and all(label in (0, 1) and not isinstance(label, bool) for label in labels)):
        raise _bad("`point_labels` must hold one 1 (object) or 0 (background) per point")
    return [tuple(p) for p in points], labels


@bp.post("/segment_selection")
@require_auth
def segment_selection():
    """Outline of the object inside a dragged box, by SAM 2.1. Synchronous, free.

    Multipart:

        image         the room photograph the box was drawn on
        photo_id      the client's stable id for that photo; with a hash of the
                      pixels it keys the embedding cache, so the expensive image
                      encoder runs once per photo, not once per box
        box           JSON [x0, y0, x1, y1], normalised 0..1
        points        optional JSON [[x, y], ...], normalised — refinement clicks
        point_labels  optional JSON [1|0, ...]; 1 = object, 0 = background.
                      Defaults to all 1.

    200: {"polygons": [[[x, y], ...], ...], "score": float,
          "bounds": [x0, y0, x1, y1] | null, "width": px, "height": px}
    Outlines are normalised, largest first. An empty list means SAM found no
    object in the box; the client then keeps the box itself as the area.

    No token cost: it is a selection aid on the way to a paid generation, and
    charging per drag would punish the user for adjusting a box.
    """
    started = time.perf_counter()
    upload = request.files.get("image")
    if upload is None:
        raise _bad("`image` is required — the photo the box was drawn on")
    try:
        image = Image.open(upload.stream)
        image.load()
    except UnidentifiedImageError:
        raise _bad("`image` is not a readable image") from None

    box = _selection_box()
    points, labels = _selection_points()
    photo_id = (request.form.get("photo_id") or "").strip()[:128] or None

    config = _config()
    try:
        selection = sam_selection.segment(
            image, box, points=points, point_labels=labels, photo_id=photo_id,
            model=getattr(config, "SAM_MODEL", sam_selection.DEFAULT_MODEL),
            device=getattr(config, "SAM_DEVICE", None),
            cache_size=getattr(config, "SAM_EMBEDDING_CACHE", 8),
        )
    except sam_selection.SegmentationUnavailable as exc:
        raise AppError(str(exc), status=503, code="segmentation_unavailable") from None

    log.info("segment_selection: user=%s photo=%s %dx%d polygons=%d score=%.3f in %.2fs",
             (g.user or {}).get("id"), photo_id, selection.width, selection.height,
             len(selection.polygons), selection.score, time.perf_counter() - started)
    return jsonify(
        polygons=selection.polygons,
        score=round(selection.score, 4),
        bounds=list(selection.bounds) if selection.bounds else None,
        width=selection.width,
        height=selection.height,
    )


def _where(x, y):
    """Plain-language spot in the frame for a normalised point."""
    across = "left" if x < 1 / 3 else "centre" if x < 2 / 3 else "right"
    depth = ("close to the camera" if y > 0.75 else "in the middle of the room" if y > 0.55
             else "towards the back of the room")
    return f"{across} of the frame, {depth}"


def _fixed_positions_prompt(plan, geometry, fetched, ids):
    """Where each hand-placed piece is, in terms of the picture.

    The planned composition is metres in camera space, which FLUX cannot relate
    to the image. This states the same thing as percentages of the frame,
    taken from the projected outline the composite pasted the piece into.
    Hand-placed pieces are the plan entries without a `relation`: the planner
    always records one.
    """
    # The wall planner fills in `relation` for everything it places, so a
    # plan entry with neither a relation nor a wall was positioned by hand.
    hand_placed = [entry for entry in plan or ()
                   if entry.get("relation") is None and entry.get("wall_id") is None]
    if not hand_placed or geometry is None:
        return ""
    names = {str(m.get("id")): (m.get("name") or "piece", i + 2)
             for i, (m, _content) in enumerate(fetched)}
    width, height = geometry.size
    lines = []
    for entry in hand_placed:
        try:
            quad, _ppm = geometry.object_quad(
                None, entry["width_m"], entry["height_m"],
                base_m=entry["base_m"], right_axis=entry["right_axis"],
            )
        except (ValueError, KeyError, TypeError):
            continue
        x0, y0 = quad[:, 0].min() / width, quad[:, 1].min() / height
        x1, y1 = quad[:, 0].max() / width, quad[:, 1].max() / height
        foot_x, foot_y = (x0 + x1) / 2, y1
        label = entry.get("label") or "piece"
        image_index = next((index for key, (_name, index) in names.items()
                            if f"(id {key})" in label), None)
        source = f" (image {image_index})" if image_index else ""
        lines.append(
            f"* {label}{source}: stands {_where(foot_x, foot_y)} — its feet at "
            f"{foot_x:.0%} from the left and {foot_y:.0%} from the top; it fills the "
            f"area from {max(x0, 0):.0%} to {min(x1, 1):.0%} across and "
            f"{max(y0, 0):.0%} to {min(y1, 1):.0%} down."
        )
    if not lines:
        return ""
    return (
        "# FIXED POSITIONS — HIGHEST PRIORITY\n"
        "The user placed these pieces by hand. Image 1 shows each one at its final "
        "spot and size. Each one stays exactly there, at the same place and size in "
        "the frame, even where another spot would look more natural. These positions "
        "take priority over the layout below.\n"
        + "\n".join(lines) + "\n\n"
    )


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
