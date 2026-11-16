"""Conservative, geometric room estimates; no semantic wall segmentation.

Assumes one room with a flat floor, flat ceiling and perpendicular wall pairs.
Large furniture can imitate walls. Evidence is heuristic, not a probability
or an accuracy guarantee. Missing surfaces never become bounding-box guesses.
"""

import numpy as np


def _unavailable(reason):
    return {"value_m": None, "status": "unavailable", "reason": reason}


def _estimate(value, planes):
    return {
        "value_m": round(float(value), 2),
        "status": "estimated",
        "evidence": {
            "support_points": [p["count"] for p in planes],
            "plane_residual_m": [round(p["residual"], 4) for p in planes],
        },
    }


def _planes(points, tolerance=0.04):
    """Extract large planes with bounded, deterministic RANSAC and thin SVD."""
    rng = np.random.default_rng(17)
    remaining = points.copy()
    minimum = max(100, int(len(points) * 0.025))
    planes = []
    for _ in range(12):
        if len(remaining) < minimum:
            break
        best = None
        best_count = 0
        for _ in range(250):
            a, b, c = remaining[rng.choice(len(remaining), 3, replace=False)]
            normal = np.cross(b - a, c - a)
            norm = np.linalg.norm(normal)
            if norm < 1e-8:
                continue
            normal /= norm
            offset = -float(normal @ a)
            mask = np.abs(remaining @ normal + offset) < tolerance
            count = int(mask.sum())
            if count > best_count:
                best, best_count = mask, count
        if best is None or best_count < minimum:
            break
        support = remaining[best]
        centre = support.mean(axis=0)
        _, _, vt = np.linalg.svd(support - centre, full_matrices=False)
        normal = vt[-1]
        offset = -float(normal @ centre)
        mask = np.abs(remaining @ normal + offset) < tolerance
        support = remaining[mask]
        planes.append({
            "normal": normal,
            "offset": offset,
            "points": support,
            "count": int(len(support)),
            "residual": float(np.median(np.abs(support @ normal + offset))),
        })
        remaining = remaining[~mask]
    return planes


def estimate_room_dimensions(point_maps, confidence, floor_normal, floor_distance,
                             camera_height_m, min_confidence=1.5):
    """Use aligned point maps in the first camera's coordinates.

    width is the wall-pair normal most aligned with camera-right; length is
    its perpendicular horizontal axis. Labels do not mean shorter/longer.
    Scale comes entirely from the supplied first-camera height.
    """
    if not np.isfinite(camera_height_m) or camera_height_m <= 0:
        raise ValueError("camera_height_m must be finite and positive")
    if not np.isfinite(floor_distance) or floor_distance <= 0:
        raise ValueError("floor_distance must be finite and positive")
    up = np.asarray(floor_normal, dtype=float)
    if not np.isfinite(up).all() or not np.isclose(np.linalg.norm(up), 1):
        raise ValueError("floor_normal must be a finite unit vector")
    point_maps = np.asarray(point_maps)
    confidence = np.asarray(confidence)
    if point_maps.shape[:-1] != confidence.shape or point_maps.shape[-1] != 3:
        raise ValueError("point maps and confidence shapes do not match")
    if point_maps.ndim == 3:
        point_maps, confidence = point_maps[None], confidence[None]
    result = {
        "width": _unavailable("Two opposing side walls were not detected."),
        "length": _unavailable("Two opposing front/back walls were not detected."),
        "height": _unavailable("A sufficiently broad ceiling plane was not detected."),
        "method": "multi_view_plane_fitting",
        "views": int(len(point_maps)),
        "camera_height_m": float(camera_height_m),
        "assumptions": [
            "All photos show one room with a flat floor and ceiling and perpendicular walls.",
            "Scale depends on the supplied camera height for the first photo.",
            "Plane evidence is geometric only: furniture can be mistaken for walls.",
            "Width is the room axis most aligned with the first camera's right direction.",
        ],
    }
    rng = np.random.default_rng(9)
    samples = []
    metres_per_unit = camera_height_m / floor_distance
    for view, conf in zip(point_maps, confidence):
        valid = np.isfinite(view).all(axis=-1) & np.isfinite(conf) & (conf > min_confidence)
        points = view[valid] * metres_per_unit
        if len(points) > 5000:
            points = points[rng.choice(len(points), 5000, replace=False)]
        samples.append(points)
    points = np.concatenate(samples) if samples else np.empty((0, 3))
    # Floor-relative height: camera origin is camera_height_m above the floor.
    heights = points @ up + camera_height_m
    points = points[heights > -0.15]
    if len(points) < 300:
        for key in ("width", "length", "height"):
            result[key] = _unavailable("Too few confident reconstructed points.")
        return result
    if len(points) > 15000:
        points = points[rng.choice(len(points), 15000, replace=False)]
    right = np.array([1., 0., 0.])
    right -= (right @ up) * up
    right /= np.linalg.norm(right)
    forward = np.cross(right, up)
    heights = points @ up + camera_height_m
    planes = _planes(points)
    ceiling_candidates, walls = [], []
    for plane in planes:
        normal, support = plane["normal"], plane["points"]
        vertical_alignment = abs(float(normal @ up))
        support_heights = support @ up + camera_height_m
        if vertical_alignment > np.cos(np.deg2rad(8)):
            h = float(np.median(support_heights))
            spans = [float(np.ptp(np.quantile(support @ axis, [.05, .95])))
                     for axis in (right, forward)]
            if (h > max(1.8, camera_height_m + .3) and min(spans) >= 1.0
                    and np.mean(heights > h + .15) < .02):
                ceiling_candidates.append((h, plane))
        elif vertical_alignment < np.sin(np.deg2rad(8)):
            low, high = np.quantile(support_heights, [.05, .95])
            horizontal = normal - (normal @ up) * up
            horizontal /= np.linalg.norm(horizontal)
            tangent = np.cross(up, horizontal)
            span = float(np.ptp(np.quantile(support @ tangent, [.05, .95])))
            # Exclude floating panels and small furniture faces.
            if high - low >= 1.2 and low < .45 and span >= 1.0:
                walls.append(plane)
    if ceiling_candidates:
        h, ceiling = max(ceiling_candidates, key=lambda item: item[0])
        result["height"] = _estimate(h, [ceiling])

    pairs = []
    for i, first in enumerate(walls):
        for second in walls[i + 1:]:
            if abs(first["normal"] @ second["normal"]) < np.cos(np.deg2rad(8)):
                continue
            axis = first["normal"] - (first["normal"] @ up) * up
            axis /= np.linalg.norm(axis)
            offsets = sorted(float(np.median(p["points"] @ axis))
                             for p in (first, second))
            lo, hi = offsets
            separation = hi - lo
            projections = points @ axis
            # Both observed walls must bracket the first camera and enclose
            # almost all reconstructed points. A partial cloud is not a wall.
            if (lo >= -.1 or hi <= .1 or separation < 1.2
                    or np.mean((projections < lo - .15) | (projections > hi + .15)) > .03):
                continue
            pairs.append((axis, separation, first, second))
    # Require distinct perpendicular pairs for the two horizontal dimensions.
    chosen = []
    for pair in sorted(pairs, key=lambda p: p[2]["count"] + p[3]["count"], reverse=True):
        if all(abs(pair[0] @ other[0]) < np.sin(np.deg2rad(10)) for other in chosen):
            chosen.append(pair)
        if len(chosen) == 2:
            break
    for axis, separation, first, second in chosen:
        key = "width" if abs(axis @ right) >= abs(axis @ forward) else "length"
        result[key] = _estimate(separation, [first, second])
    return result
