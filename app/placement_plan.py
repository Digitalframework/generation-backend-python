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




# --------------------------------------------------------------------------- #
# layout rules
# --------------------------------------------------------------------------- #
# The living-room rule set, as numbers the planner can check. Rule ids in the
# comments (S1, C2, ...) refer to the written rule set.
#
# Placement order (rule set, step 0): lower rank is placed first. Types not
# listed are placed last, against a wall, the way every piece used to be.
RANK = {
    "sofa": 0,
    "lowboard": 1,
    "coffee_table": 2,
    "side_table": 3,
    "ottoman": 3,
    "armchair": 4,
    "sideboard": 5,
    "dresser": 5,
    "shelf": 5,
}
OTHER_RANK = 9

# Positioned relative to the sofa instead of against a wall.
SOFA_RELATIVE = {"coffee_table", "side_table", "ottoman", "armchair"}
SEATING_GROUP = SOFA_RELATIVE | {"sofa"}
# Stand against a wall, after the media wall: sideboard, shelves (B1-B5).
STORAGE = {"sideboard", "dresser", "shelf"}
# May be left out before a walkway is narrowed (conflict rule 4).
OPTIONAL = {"armchair", "side_table", "sideboard", "dresser", "shelf"}

DEFAULT_WALL_GAP_M = .05
SOFA_MAX_WALL_GAP_M = .50  # S2
COFFEE_GAPS_M = (.45, .40, .50)  # C1 / C2, tried in this order
SIDE_TABLE_GAP_M = .05  # T1: right at the armrest
OTTOMAN_GAP_M = .10  # O2: beside the coffee table
ARMCHAIR_GAPS_M = (.45, .30, .60)  # A1
TV_DISTANCE_M = (2.0, 3.5)  # M3
SIDE_PASSAGE_M = .60  # W2, between a free-standing piece and anything outside its group
FRONT_CLEARANCE_M = {  # W1 / B4: kept free in front of these pieces
    "lowboard": .80,
    "sideboard": .80,
    "dresser": .80,
    "shelf": .60,
}
SIGHTLINE_HALF_WIDTH_M = .40  # A5: corridor between sofa and lowboard
SIGHTLINE_MIN_HEIGHT_M = .60  # lower pieces (ottoman, coffee table) do not block it
DEFAULT_SEAT_DEPTH_M = .95
DEFAULT_CHAISE_WIDTH_M = .90


class PlacementConflict(ValueError):
    def __init__(self, issues):
        super().__init__("Furniture needs a supported wall placement before generation.")
        self.issues = issues


def _unit(vector):
    return vector / np.linalg.norm(vector)


def _rect(rear, right, facing, width, depth):
    """Floor rectangle: rear-left, rear-right, front-right, front-left."""
    return np.array([rear - right * width / 2,
                     rear + right * width / 2,
                     rear + right * width / 2 + facing * depth,
                     rear - right * width / 2 + facing * depth])


def _footprint(wall, offset, width, depth, gap):
    rear = wall.origin + wall.right * offset + wall.inward * gap
    return _rect(rear, wall.right, wall.inward, width, depth)


def _overlap(a, b, clearance=.05):
    """Separating axis theorem for two convex polygons on the same floor."""
    normal = np.cross(a[1] - a[0], a[-1] - a[0])
    if np.linalg.norm(normal) < 1e-9:
        normal = np.cross(b[1] - b[0], b[-1] - b[0])
    normal = _unit(normal)
    for poly in (a, b):
        for i in range(len(poly)):
            axis = np.cross(normal, poly[(i + 1) % len(poly)] - poly[i])
            length = np.linalg.norm(axis)
            if length < 1e-9:
                continue
            axis = axis / length
            x, y = a @ axis, b @ axis
            if x.max() + clearance <= y.min() or y.max() + clearance <= x.min():
                return False
    return True


def _opposite(a, b):
    """True when wall `b` faces wall `a` across the room."""
    return a is not b and float(a.inward @ b.inward) < -.9


def _wall_side(wall):
    """Where a wall is, seen from the camera (camera frame: x right, z forward)."""
    x, z = float(wall.inward[0]), float(wall.inward[2])
    if abs(z) >= abs(x):
        return "back wall" if z < 0 else "wall behind the camera"
    return "left wall" if x > 0 else "right wall"


@dataclass
class _Placed:
    item: object
    kind: str
    parts: list  # floor polygons; an L-sofa has two
    bottom: float
    top: float
    zone: np.ndarray  # floor kept clear in front, or None
    rear: np.ndarray  # rear centre on the floor
    right: np.ndarray
    facing: np.ndarray
    free: bool  # stands in the room rather than against a wall
    wall: Wall = None
    width: float = 0.
    depth: float = 0.
    seat_depth: float = 0.  # sofa: depth of the main body
    chaise: str = None  # sofa: "left" / "right", seen facing the sofa
    chaise_width: float = 0.
    u: float = 0.  # position in the sofa's frame, for pieces placed from it
    v: float = 0.

    @property
    def centre(self):
        return self.rear + self.facing * self.depth / 2


class _Planner:
    def __init__(self, geometry):
        self.geometry = geometry
        self.walls = geometry.walls
        self.up = _unit(np.asarray(geometry.normal, dtype=float))
        self.placed = []
        self.sofa = self.media = self.coffee = None
        self.sightline = None

    # -- checks ----------------------------------------------------------

    def _clearance(self, kind, free, other):
        if kind in SEATING_GROUP and other.kind in SEATING_GROUP:
            return .05
        if free or other.free:
            return SIDE_PASSAGE_M
        return .05

    def _fits(self, kind, parts, zone, bottom, top, free, avoid_sightline):
        # Every footprint must remain on the room side of all observed walls.
        for poly in parts:
            if any(np.min((poly - wall.origin) @ wall.inward) < -.03 for wall in self.walls):
                return False
        on_floor = bottom < .05
        for other in self.placed:
            if bottom < other.top and top > other.bottom:
                gap = self._clearance(kind, free, other)
                if any(_overlap(a, b, gap) for a in parts for b in other.parts):
                    return False
            if on_floor and other.zone is not None:
                if any(_overlap(a, other.zone, 0.) for a in parts):
                    return False
            if zone is not None and other.bottom < .05:
                if any(_overlap(zone, b, 0.) for b in other.parts):
                    return False
        if avoid_sightline and self.sightline is not None:
            if any(_overlap(a, self.sightline, 0.) for a in parts):
                return False
        return True

    def _visible_base(self, item, rect, right, facing, bottom):
        """Front-centre baseline if the whole piece is in frame, else None."""
        base = (rect[2] + rect[3]) / 2 + self.up * bottom
        try:
            quad, _ = self.geometry.object_quad(None, item.width_m, item.height_m,
                                                base_m=base, right_axis=right)
        except ValueError:
            return None
        width, height = self.geometry.size
        # Require the whole guide to be visible; off-frame is a separate
        # issue, never grounds for silently dropping a catalogue item.
        if (quad[:, 0].min() < 0 or quad[:, 0].max() >= width
                or quad[:, 1].min() < 0 or quad[:, 1].max() >= height):
            return None
        return base

    def _try(self, item, kind, rect, right, facing, *, parts=None, bottom=0.,
             wall=None, free=False, relation=None, avoid_sightline=False, **extra):
        parts = parts or [rect]
        top = bottom + item.height_m
        zone = None
        depth = FRONT_CLEARANCE_M.get(kind)
        if depth and bottom < .05:
            zone = _rect((rect[2] + rect[3]) / 2, right, facing, item.width_m, depth)
        if not self._fits(kind, parts, zone, bottom, top, free, avoid_sightline):
            return None
        base = self._visible_base(item, rect, right, facing, bottom)
        if base is None:
            return None
        item.base_m, item.right_axis = base, right.copy()
        item.wall_id = wall.id if wall is not None else None
        item.mount_height_m = bottom
        item.kind = kind
        placed = _Placed(item, kind, parts, bottom, top, zone, (rect[0] + rect[1]) / 2,
                         right, facing, free, wall, item.width_m, item.depth_size_m, **extra)
        self.placed.append(placed)
        item.relation = dict(relation or {}, type=kind)
        if wall is not None:
            item.relation.setdefault("wall_side", _wall_side(wall))
        return placed

    # -- picture terms ---------------------------------------------------

    def _picture_side(self, point, anchor):
        """'left' or 'right' of `anchor`, as the camera sees it."""
        try:
            upm = self.geometry.units_per_metre
            x = self.geometry._project(np.asarray(point) * upm)[0]
            x0 = self.geometry._project(np.asarray(anchor) * upm)[0]
        except Exception:  # noqa: BLE001 - only used for wording
            return None
        return "left" if x < x0 else "right"

    # -- wall placement --------------------------------------------------

    def _on_wall(self, item, kind, spec, walls, aim=None, relation=None, sofa=False):
        """Slide the piece along each wall in turn, nearest `aim(wall)` first.

        With no `aim`, offsets are tried nearest the wall's point closest to
        the camera, which is how every piece was placed before the rule set.
        """
        gap = spec.get("wall_gap_m")
        gap = DEFAULT_WALL_GAP_M if gap is None else gap
        if sofa:
            gap = min(gap, SOFA_MAX_WALL_GAP_M)
        bottom = spec.get("mount_height_m") or 0.
        for wall in walls:
            half = item.width_m / 2
            lo, hi = wall.start_m + half + .05, wall.end_m - half - .05
            if lo > hi:
                continue
            if spec.get("wall_offset_m") is not None:
                offsets = [spec["wall_offset_m"]]
            else:
                target = aim(wall) if aim else 0.
                offsets = sorted(np.linspace(lo, hi, max(2, int((hi - lo) / .1) + 1)),
                                 key=lambda o: abs(o - target))
            for offset in offsets:
                if not lo <= offset <= hi:
                    continue
                rect = _footprint(wall, offset, item.width_m, item.depth_size_m, gap)
                extra, parts = {}, None
                if sofa:
                    extra, parts = self._sofa_shape(item, spec, wall, offset, gap)
                rel = dict(relation(wall) if callable(relation) else (relation or {}))
                rel.setdefault("rule", "against_wall")
                rel["wall_gap_m"] = round(float(gap), 2)
                placed = self._try(item, kind, rect, wall.right, wall.inward, parts=parts,
                                   bottom=bottom, wall=wall, relation=rel, **extra)
                if placed:
                    return placed
        return None

    def _choices(self, spec, ordered):
        wanted = spec.get("wall_id")
        return [w for w in ordered if wanted in (None, w.id)]

    @staticmethod
    def _centre_of(wall):
        return (wall.start_m + wall.end_m) / 2

    def _aim_at(self, point):
        return lambda wall: float((point - wall.origin) @ wall.right)

    # -- sofa ------------------------------------------------------------

    def _sofa_shape(self, item, spec, wall, offset, gap):
        """An L-sofa is two rectangles, so the inner corner stays usable (C2)."""
        width, depth = item.width_m, item.depth_size_m
        chaise = spec.get("chaise_side")
        seat = min(spec.get("seat_depth_m") or DEFAULT_SEAT_DEPTH_M, depth)
        chaise_width = min(spec.get("chaise_width_m") or DEFAULT_CHAISE_WIDTH_M, width)
        extra = {"seat_depth": seat if chaise else depth, "chaise": chaise,
                 "chaise_width": chaise_width if chaise else 0.}
        if not chaise or seat >= depth - .05:
            return extra, None
        rear = wall.origin + wall.right * offset + wall.inward * gap
        sign = 1. if chaise == "right" else -1.
        body = _rect(rear, wall.right, wall.inward, width, seat)
        chaise_rear = rear + wall.right * sign * (width - chaise_width) / 2
        lounge = _rect(chaise_rear, wall.right, wall.inward, chaise_width, depth)
        return extra, [body, lounge]

    def place_sofa(self, item, spec):
        def score(wall):
            opposite = [o for o in self.walls if _opposite(wall, o)]
            distances = [float((o.origin - wall.origin) @ wall.inward) - item.depth_size_m
                         for o in opposite]
            good = any(TV_DISTANCE_M[0] <= d <= TV_DISTANCE_M[1] + .5 for d in distances)
            return (0 if good else 1 if opposite else 2, -(wall.end_m - wall.start_m))

        walls = self._choices(spec, sorted(self.walls, key=score))
        placed = self._on_wall(item, "sofa", spec, walls, aim=self._centre_of,
                               relation={"chaise_side": spec.get("chaise_side")}, sofa=True)
        if placed:
            self.sofa = placed
        return placed

    def _sofa_point(self, u, v):
        s = self.sofa
        return s.rear + s.right * u + s.facing * v

    def _sofa_front(self):
        """Centre of the main body's front edge, in sofa coordinates."""
        s = self.sofa
        if s.chaise:
            sign = 1. if s.chaise == "right" else -1.
            return -sign * s.chaise_width / 2, s.seat_depth
        return 0., s.depth

    # -- media wall ------------------------------------------------------

    def place_media(self, item, spec):
        if self.sofa is None:
            walls = self._choices(spec, self.walls)
            placed = self._on_wall(item, "lowboard", spec, walls, aim=self._centre_of)
        else:
            s = self.sofa
            ideal = s.depth + sum(TV_DISTANCE_M) / 2
            opposite = sorted((w for w in self.walls if _opposite(s.wall, w)),
                              key=lambda w: abs(float((w.origin - s.wall.origin)
                                                      @ s.wall.inward) - ideal))
            others = [w for w in self.walls if w is not s.wall and w not in opposite]
            walls = self._choices(spec, opposite + others + [s.wall])

            def relation(wall):
                if _opposite(s.wall, wall):
                    return {"rule": "opposite_sofa", "anchor": "sofa"}
                return {}

            placed = self._on_wall(item, "lowboard", spec, walls,
                                   aim=self._aim_at(s.centre), relation=relation)
        if placed and self.media is None:
            self.media = placed
            self._set_sightline()
        return placed

    def _set_sightline(self):
        if self.sofa is None or self.media is None:
            return
        u, v = self._sofa_front()
        start = self._sofa_point(u, v)
        end = (self.media.parts[0][2] + self.media.parts[0][3]) / 2
        if float((end - start) @ self.sofa.facing) < .5:
            return  # media is not in front of the sofa: no sightline to keep
        r = self.sofa.right * SIGHTLINE_HALF_WIDTH_M
        self.sightline = np.array([start - r, start + r, end + r, end - r])

    # -- pieces placed from the sofa ------------------------------------

    def _from_sofa(self, item, kind, u, v, facing=None, relation=None, avoid_sightline=False,
                   **extra):
        s = self.sofa
        if facing is None:
            rear, right, facing = self._sofa_point(u, v), s.right, s.facing
        else:
            centre = self._sofa_point(u, v)
            facing = facing - (facing @ self.up) * self.up
            facing = _unit(facing)
            right = _unit(np.cross(self.up, facing))
            rear = centre - facing * item.depth_size_m / 2
        rect = _rect(rear, right, facing, item.width_m, item.depth_size_m)
        rel = dict(relation or {})
        side = self._picture_side(rect.mean(0), s.centre)
        if side and rel.get("rule") in ("beside_sofa", "near_sofa", "beside_coffee_table"):
            rel["picture_side"] = side
        return self._try(item, kind, rect, right, facing, free=True, relation=rel,
                         avoid_sightline=avoid_sightline, u=u, v=v, **extra)

    def place_coffee_table(self, item, spec):
        s, width = self.sofa, item.width_m
        sign = 1. if s.chaise == "right" else -1.
        for gap in COFFEE_GAPS_M:
            if s.chaise:
                # C2: about the same distance from both arms of the L.
                inner = sign * (s.width / 2 - s.chaise_width)
                u0, v = inner - sign * (gap + width / 2), s.seat_depth + gap
                shifts = (0., -sign * .1, -sign * .2)
            else:
                u0, v = 0., s.depth + gap
                shifts = (0., .1, -.1, .2, -.2, .3, -.3)
            for shift in shifts:
                placed = self._from_sofa(item, "coffee_table", u0 + shift, v,
                                         relation={"rule": "in_front_of_sofa", "anchor": "sofa",
                                                   "distance_m": gap})
                if placed:
                    self.coffee = self.coffee or placed
                    return placed
        return None

    def _open_sides(self):
        """+1 / -1 along the sofa, the side with more free floor first (T3)."""
        s = self.sofa
        if s.chaise:  # T1: never beside the armless chaise end
            return [-1. if s.chaise == "right" else 1.]

        def room(side):
            end = self._sofa_point(side * s.width / 2, s.depth / 2)
            ahead = side * s.right
            limits = [float((end - w.origin) @ w.inward) for w in self.walls
                      if float(ahead @ -w.inward) > .5]
            return min(limits, default=np.inf)

        return sorted((1., -1.), key=room, reverse=True)

    def place_side_table(self, item, spec):
        s = self.sofa
        for side in self._open_sides():
            u = side * (s.width / 2 + SIDE_TABLE_GAP_M + item.width_m / 2)
            placed = self._from_sofa(item, "side_table", u, 0.,
                                     relation={"rule": "beside_sofa", "anchor": "sofa"})
            if placed:
                return placed
        return None

    def place_ottoman(self, item, spec):
        s = self.sofa
        if self.coffee is not None:
            c = self.coffee
            for side in self._open_sides() + [-x for x in self._open_sides()]:
                u = c.u + side * (c.width / 2 + OTTOMAN_GAP_M + item.width_m / 2)
                placed = self._from_sofa(item, "ottoman", u, c.v,
                                         relation={"rule": "beside_coffee_table",
                                                   "anchor": "coffee_table"})
                if placed:
                    return placed
            return None
        u0, front = self._sofa_front()
        for gap in COFFEE_GAPS_M:  # O3: takes the coffee table's place
            for shift in (0., .1, -.1, .2, -.2):
                placed = self._from_sofa(item, "ottoman", u0 + shift, front + gap,
                                         relation={"rule": "in_front_of_sofa", "anchor": "sofa",
                                                   "distance_m": gap})
                if placed:
                    return placed
        return None

    def place_armchair(self, item, spec):
        s = self.sofa
        u_front, v_front = self._sofa_front()
        if self.coffee is not None:
            target_u, target_v = self.coffee.u, self.coffee.v + self.coffee.depth / 2
            target_half = self.coffee.width / 2
        else:
            target_u, target_v, target_half = u_front, v_front + .6, .3
        target = self._sofa_point(target_u, target_v)
        avoid = item.height_m >= SIGHTLINE_MIN_HEIGHT_M
        half_w, half_d = item.width_m / 2, item.depth_size_m / 2
        for gap in ARMCHAIR_GAPS_M:
            candidates = []
            for side in self._open_sides() + [-x for x in self._open_sides()]:
                # A1/A2: at the open end of the coffee table, facing it.
                candidates.append((target_u + side * (target_half + gap + half_d), target_v,
                                   "near_sofa"))
                # Beside the sofa's end, turned in toward the table.
                candidates.append((side * (s.width / 2 + gap + half_w),
                                   s.depth + half_d + .1, "near_sofa"))
            for u, v, rule in dict.fromkeys(candidates):
                centre = self._sofa_point(u, v)
                facing = target - centre
                if np.linalg.norm(facing - (facing @ self.up) * self.up) < 1e-6:
                    continue
                placed = self._from_sofa(item, "armchair", u, v, facing=facing,
                                         relation={"rule": rule, "anchor": "sofa",
                                                   "distance_m": gap},
                                         avoid_sightline=avoid)
                if placed:
                    return placed
        return None

    # -- storage ---------------------------------------------------------

    def place_storage(self, item, kind, spec):
        """B2: next to the media wall's lowboard first, then a free wall."""
        sofa_wall = self.sofa.wall if self.sofa else None
        first = [self.media.wall] if self.media is not None else []
        rest = [w for w in self.walls if w not in first and w is not sofa_wall]
        last = [sofa_wall] if sofa_wall is not None and sofa_wall not in first else []
        walls = self._choices(spec, first + rest + last)
        media = self.media

        def aim(wall):
            if media is not None and wall is media.wall:
                return float((media.centre - wall.origin) @ wall.right)
            return self._centre_of(wall)

        def relation(wall):
            if media is not None and wall is media.wall:
                return {"rule": "next_to_media", "anchor": "lowboard"}
            return {}

        return self._on_wall(item, kind, spec, walls, aim=aim, relation=relation)

    def place_on_wall(self, item, kind, spec):
        return self._on_wall(item, kind, spec, self._choices(spec, self.walls))


def plan_placements(placements, specs, geometry):
    """Place furniture following the living-room rule set.

    Order: sofa, lowboard, coffee table, side tables and ottoman, armchair,
    sideboard and shelves, then everything else. The sofa, lowboard and
    storage stand against walls; the coffee table, side tables, ottoman and
    armchair are placed from the sofa. A placement's `kind` is its furniture
    type slug (prompt_builder.TYPES); without one, a piece is placed against
    a wall exactly as before.

    A spec that names `wall_id` or `wall_offset_m` puts the piece on a wall
    whatever its type. No silent resizing or omission: a piece that cannot be
    placed without breaking a walkway is reported, marked `optional` when the
    rule set says it is the one to leave out. Failure means no placement was
    found by this bounded planner, not proof of no solution.

    Uses bounding footprints, including full chaise depth for an L sofa unless
    the spec gives `chaise_side` (then the inner corner of the L is free).
    """
    walls = geometry.walls
    issues = []
    if not walls:
        raise PlacementConflict([{"reason": "walls_unavailable",
                                  "message": "No supported wall planes; supply clearer room views."}])
    planner = _Planner(geometry)

    def rank(index):
        kind = getattr(placements[index], "kind", None)
        return RANK.get(kind, OTHER_RANK), -placements[index].width_m

    for index in sorted(range(len(placements)), key=rank):
        item = placements[index]
        spec = specs[index] if index < len(specs) else {}
        kind = getattr(item, "kind", None)
        if item.depth_size_m is None:
            issues.append({"index": index, "label": item.label,
                           "reason": "depth_required", "message": "Supply full footprint depth_m."})
            continue
        on_wall = spec.get("wall_id") is not None or spec.get("wall_offset_m") is not None
        if kind == "sofa":
            placed = planner.place_sofa(item, spec)
        elif kind == "lowboard":
            placed = planner.place_media(item, spec)
        elif kind in SOFA_RELATIVE and planner.sofa is not None and not on_wall:
            placed = getattr(planner, f"place_{kind}")(item, spec)
            if placed is None and kind == "armchair":
                # A3: a reading corner on a free wall instead.
                placed = planner.place_storage(item, kind, spec)
        elif kind in STORAGE:
            placed = planner.place_storage(item, kind, spec)
        else:
            placed = planner.place_on_wall(item, kind, spec)
        if placed is None:
            optional = kind in OPTIONAL
            options = ["choose_another_wall", "supply_measurements_or_views",
                       "exclude_from_this_generation"]
            if optional:
                options = ["exclude_from_this_generation"] + options[:-1]
            issues.append({"index": index, "label": item.label, "reason": "no_supported_placement",
                           "type": kind, "optional": optional,
                           "width_m": item.width_m, "depth_m": item.depth_size_m,
                           "message": "No collision-free, fully visible placement found that "
                                      "keeps the walkways clear.",
                           "options": options})
    if issues:
        raise PlacementConflict(issues)
    return placements
