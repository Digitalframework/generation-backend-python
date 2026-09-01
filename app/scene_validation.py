"""Validation of the JSON that describes a room and what goes into it.

Shared by /api/compose, where the caller uploads the cutouts, and /api/generate,
where they come out of the catalogue — both end up building the same
`Placement` objects, and a room described one way in one endpoint and
another way in the other would be a trap.
"""

import json

from flask import request

from .errors import AppError


def bad_request(message):
    return AppError(message, status=400, code="bad_request")


def json_field(name, required=True):
    raw = request.form.get(name)
    if raw is None:
        if required:
            raise bad_request(f"missing form field `{name}`")
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise bad_request(f"`{name}` is not valid JSON: {exc}") from None


def point_list(value, name):
    """Validate [[x, y], [x, y]] — cv2 would otherwise fail deep in the warp."""
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(
            isinstance(p, list) and len(p) == 2 and all(isinstance(c, (int, float)) for c in p)
            for p in value
        )
    ):
        raise bad_request(f"`{name}` must be [[x, y], [x, y]]")
    return [tuple(p) for p in value]


def build_geometry(spec):
    """The one number left in `geometry`: how high the camera stood.

    Floor and ceiling lines are gone - VGGT reconstructs the room from the
    photographs themselves. `camera_height_m` survives because it is the single
    real-world measurement that gives that reconstruction a metric scale.
    """
    if not isinstance(spec, dict):
        raise bad_request("`geometry` must be an object")

    for obsolete in ("floor_line", "ceiling_line", "horizon_y"):
        if spec.get(obsolete) is not None:
            raise bad_request(
                f"`geometry.{obsolete}` is no longer used — the room is "
                "reconstructed from the `rooms` images. Send only "
                "`geometry.camera_height_m`."
            )

    camera_height = spec.get("camera_height_m")
    if camera_height is None:
        return None
    if not isinstance(camera_height, (int, float)) or isinstance(camera_height, bool):
        raise bad_request("`geometry.camera_height_m` must be a number")
    if not 0 < camera_height <= 10:
        raise bad_request("`geometry.camera_height_m` must be between 0 and 10 metres")
    return float(camera_height)


def validate_placement(spec, index, require_size=True):
    """Check one placement without touching its image.

    Kept separate from decoding so a malformed request is a 400 on the strength
    of the request alone — no balance lookup, no image decode, and no dependence
    on AuthAndSecurity being up to reject something plainly wrong.

    `require_size` is False for the catalogue path, where width_m and height_m
    come from the stored width_cm/height_cm unless the caller overrides them.
    """
    if not isinstance(spec, dict):
        raise bad_request(f"placements[{index}] must be an object")

    required = ("center_x", "width_m", "height_m") if require_size else ()
    for key in required:
        value = spec.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise bad_request(f"placements[{index}].{key} is required and must be a number")

    for key in ("center_x", "width_m", "height_m", "forward_offset_px"):
        value = spec.get(key)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise bad_request(f"placements[{index}].{key} must be a number")
    for key in ("width_m", "height_m"):
        if spec.get(key) is not None and spec[key] <= 0:
            raise bad_request(f"placements[{index}].{key} must be positive")

    if spec.get("light_from", "left") not in ("left", "right"):
        raise bad_request(f"placements[{index}].light_from must be 'left' or 'right'")

    # Shadow and reflection default off, so these keys are the only way to ask
    # for them - which makes them the ones a caller is most likely to get wrong.
    for key in ("shadow_core", "shadow_ambient"):
        value = spec.get(key)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise bad_request(f"placements[{index}].{key} must be a number")
        if not 0 <= value <= 1:
            raise bad_request(f"placements[{index}].{key} must be between 0 and 1")

    if spec.get("glossy_floor") is not None and not isinstance(spec["glossy_floor"], bool):
        raise bad_request(f"placements[{index}].glossy_floor must be true or false")

    return spec
