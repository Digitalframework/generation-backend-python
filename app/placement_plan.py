"""Deterministic placement on observed wall segments, in first-camera metres.

Planes are geometric candidates, not semantic wall/opening detection. Observed
support bounds are conservative; missing walls are never invented from extents.
"""

from dataclasses import dataclass

import numpy as np

from .room_dimensions import _planes


@dataclass
class Wall:
    id: str
    origin: np.ndarray  # floor/wall intersection, metres in camera frame
    right: np.ndarray  # along wall
    inward: np.ndarray  # into room, perpendicular to wall and floor
    start_m: float
    end_m: float
    residual_m: float = 0.0


def fit_walls(point_maps, confidence, up, camera_height_m, units_per_metre,
              min_confidence=1.5):
    cloud = np.asarray(point_maps).reshape(-1, 3)
    conf = np.asarray(confidence).reshape(-1)
    valid = np.isfinite(cloud).all(1) & np.isfinite(conf) & (conf > min_confidence)
    cloud = cloud[valid] / units_per_metre
    if len(cloud) < 300:
        return []
    if len(cloud) > 15000:
        cloud = cloud[np.random.default_rng(23).choice(len(cloud), 15000, replace=False)]
    floor_origin = -camera_height_m * up
    walls = []
    for plane in _planes(cloud):
        normal = plane["normal"]
        if abs(normal @ up) > np.sin(np.deg2rad(8)):
            continue
        support = plane["points"]
        low, high = np.quantile(support @ up + camera_height_m, [.05, .95])
        if low > .45 or high - low < 1.2:
            continue
        inward = normal - (normal @ up) * up
        inward /= np.linalg.norm(inward)
        level = float(np.median(support @ inward))
        # Both camera and most room evidence must be on the room side.
        if level > 0:
            inward, level = -inward, -level
        if level >= -.1 or np.mean(cloud @ inward < level - .15) > .05:
            continue
        right = np.cross(up, inward)
        right /= np.linalg.norm(right)
        # Preserve the product's left/right handedness, rather than flipping
        # the texture according to the projected endpoints.
        origin = floor_origin + inward * level
        start, end = np.quantile((support - origin) @ right, [.05, .95])
        if end - start < 1.0:
            continue
        if any(abs(w.inward @ inward) > .99 and
               abs((w.origin - origin) @ inward) < .12 for w in walls):
            continue
        walls.append(Wall("", origin, right, inward, float(start), float(end),
                          plane["residual"]))
    walls.sort(key=lambda w: tuple(np.round(w.origin, 4)))
    for index, wall in enumerate(walls):
        wall.id = f"wall_{index + 1}"
    return walls


class PlacementConflict(ValueError):
    def __init__(self, issues):
        super().__init__("Furniture needs a supported wall placement before generation.")
        self.issues = issues


def _footprint(wall, offset, width, depth, gap):
    rear = wall.origin + wall.right * offset + wall.inward * gap
    return np.array([rear - wall.right * width / 2,
                     rear + wall.right * width / 2,
                     rear + wall.right * width / 2 + wall.inward * depth,
                     rear - wall.right * width / 2 + wall.inward * depth])


def _overlap(a, b, clearance=.05):
    # Separating axis theorem for two rectangles on the same floor.
    for poly in (a, b):
        for axis in (poly[1] - poly[0], poly[3] - poly[0]):
            axis = axis / np.linalg.norm(axis)
            x, y = a @ axis, b @ axis
            if x.max() + clearance <= y.min() or y.max() + clearance <= x.min():
                return False
    return True


def plan_placements(placements, specs, geometry):
    """Largest first, wall-aligned rectangles; no silent resizing or omission.

Uses bounding footprints, including full chaise depth for an L sofa. Failure
means no placement was found by this bounded planner, not proof of no solution.
"""
    walls = geometry.walls
    issues, occupied = [], []
    if not walls:
        raise PlacementConflict([{"reason": "walls_unavailable",
                                  "message": "No supported wall planes; supply clearer room views."}])
    for index in sorted(range(len(placements)), key=lambda i: -placements[i].width_m):
        item = placements[index]
        spec = specs[index] if index < len(specs) else {}
        if item.depth_size_m is None:
            issues.append({"index": index, "label": item.label,
                           "reason": "depth_required", "message": "Supply full footprint depth_m."})
            continue
        choices = [w for w in walls if spec.get("wall_id") in (None, w.id)]
        placed = False
        for wall in choices:
            half = item.width_m / 2
            lo, hi = wall.start_m + half + .05, wall.end_m - half - .05
            if lo > hi:
                continue
            offsets = ([spec["wall_offset_m"]] if spec.get("wall_offset_m") is not None
                       else sorted(np.linspace(lo, hi, max(2, int((hi-lo)/.1)+1)), key=abs))
            for offset in offsets:
                if not lo <= offset <= hi:
                    continue
                gap = spec.get("wall_gap_m")
                gap = .05 if gap is None else gap
                footprint = _footprint(wall, offset, item.width_m, item.depth_size_m, gap)
                # Every footprint must remain on the room side of all observed walls.
                if any(np.min((footprint - other.origin) @ other.inward) < -.03 for other in walls):
                    continue
                bottom = spec.get("mount_height_m") or 0.
                top = bottom + item.height_m
                if any(bottom < other_top and top > other_bottom and _overlap(footprint, other)
                       for other, other_bottom, other_top in occupied):
                    continue
                base = footprint[:2].mean(0) + wall.inward * item.depth_size_m
                base += geometry.normal * bottom
                try:
                    quad, _ = geometry.object_quad(None, item.width_m, item.height_m,
                                                   base_m=base, right_axis=wall.right)
                except ValueError:
                    continue
                # Require the whole guide to be visible; off-frame is a separate
                # issue, never grounds for silently dropping a catalogue item.
                if (quad[:, 0].min() < 0 or quad[:, 0].max() >= geometry.size[0]
                        or quad[:, 1].min() < 0 or quad[:, 1].max() >= geometry.size[1]):
                    continue
                item.base_m, item.right_axis = base, wall.right.copy()
                item.wall_id, item.mount_height_m = wall.id, bottom
                occupied.append((footprint, bottom, top))
                placed = True
                break
            if placed:
                break
        if not placed:
            issues.append({"index": index, "label": item.label, "reason": "no_supported_placement",
                           "width_m": item.width_m, "depth_m": item.depth_size_m,
                           "message": "No collision-free, fully visible placement found on observed walls.",
                           "options": ["choose_another_wall", "supply_measurements_or_views",
                                       "exclude_from_this_generation"]})
    if issues:
        raise PlacementConflict(issues)
    return placements
