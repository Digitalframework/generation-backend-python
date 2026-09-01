"""Room geometry from VGGT, in place of a hand-measured horizon.

What this replaces
------------------
The old sections 2 and 3 of `composite.py` reconstructed the room from two
lines the caller measured off the photograph — baseboard and crown molding —
and turned them into a horizon, a px/m ratio and a keystone quad:

    px_per_metre(x) = (floor_y(x) - horizon_y) / camera_height_m

That only holds for a floor whose vanishing line the two drawn lines actually
meet on, which means one flat wall, no tilt, and a caller willing to click four
points per room. Here VGGT (`facebook/VGGT-1B`, a feed-forward transformer that
predicts camera intrinsics, extrinsics and a dense depth map from one or more
images) does the reconstruction instead, and the placement becomes plain 3D:

    1. VGGT gives intrinsics K and a depth map for the scene photograph;
    2. the depth map is back-projected into camera space and the dominant plane
       under the camera is RANSAC-fitted — that is the floor, as a normal `n`
       and a distance `d` from the camera;
    3. an object is a rectangle standing *on that plane*, sized in metres,
       turned to face the camera; its four corners are projected through K and
       the cutout is warped onto them.

Everything the old code derived from the horizon falls out of the plane: the
horizon is the plane's vanishing line (l = K^-T n), the floor line under the
wall is the topmost pixel per column whose 3D point lies on the plane, and
depth ordering is the true camera-space distance rather than a proxy y.

Scale
-----
VGGT is metric-*less*: its depth is consistent but has an arbitrary unit. One
real measurement fixes it, and `camera_height_m` is the same number the old
code already took — the camera's height above the floor — so the caller-facing
contract does not change. `metres_per_unit = camera_height_m / d`.

Install
-------
    pip install torch
    pip install "vggt @ git+https://github.com/facebookresearch/vggt"

Inference is heavy (1B parameters). The model is loaded once per process and
solved scenes are cached by image content, so repeated requests against the
same room photo pay for it once.
"""

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

DEFAULT_MODEL = "facebook/VGGT-1B"

# VGGT was trained at 518px on the long side, in patches of 14. Feeding it a
# multiple of 14 on both axes keeps the mapping back to full resolution a plain
# scale factor: no crop offset, no padding to undo.
LONG_SIDE = 518
PATCH = 14

# `depth_conf` is a log-space confidence; ~1.0 is VGGT's own "trust this" line.
MIN_CONFIDENCE = 1.5

# Plane inlier band, as a fraction of the camera's height above the floor.
# Relative, because at this point the scene has no metric scale yet.
FLOOR_TOLERANCE = 0.012

_MODELS = {}
_MODEL_LOCK = threading.Lock()
_SOLVED = OrderedDict()
_SOLVED_LOCK = threading.Lock()
_SOLVED_MAX = 8


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def _resolve_device(device=None):
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _model(model_id, device):
    """One VGGT per (model, device), loaded on first use and kept.

    A billion parameters is 20-30s of load time and several GB; reloading it
    per request would dominate everything else the endpoint does.
    """
    key = (model_id, device)
    with _MODEL_LOCK:
        if key not in _MODELS:
            try:
                from vggt.models.vggt import VGGT
            except ImportError as exc:
                raise ImportError(
                    "VGGT is not installed — `pip install torch` and "
                    '`pip install "vggt @ git+https://github.com/facebookresearch/vggt"`'
                ) from exc
            model = VGGT.from_pretrained(model_id).to(device)
            model.eval()
            _MODELS[key] = model
        return _MODELS[key]


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def _preprocess(scenes):
    """RGB frames -> one float array at VGGT's resolution, plus the scale back.

    Returns (array SxHxWx3 in 0..1, sx, sy) where sx/sy map a coordinate at
    VGGT's resolution back to frame 0's photograph.

    The grid is sized from frame 0 and every other view is resized onto it,
    because VGGT wants one resolution for the whole batch and only frame 0's
    depth is ever used. The extra views condition the prediction; they are not
    themselves measured, so their own aspect ratios do not need preserving.
    """
    first = scenes[0].convert("RGB")
    width, height = first.size
    if width >= height:
        small_w = LONG_SIDE
        small_h = max(PATCH, round(height * LONG_SIDE / width))
    else:
        small_h = LONG_SIDE
        small_w = max(PATCH, round(width * LONG_SIDE / height))
    small_w -= small_w % PATCH
    small_h -= small_h % PATCH

    frames = [
        np.asarray(scene.convert("RGB").resize((small_w, small_h), Image.LANCZOS), np.float32)
        / 255.0
        for scene in scenes
    ]
    return np.stack(frames), width / small_w, height / small_h


def _predict(scenes, model_id, device):
    """Run VGGT once over every view: intrinsics, depth and confidence for frame 0.

    Extra views cost inference time and buy accuracy. VGGT attends across the
    whole batch, so a floor that is barely visible in frame 0 is still
    reconstructed from what frames 1 and 2 saw of it, and the intrinsics are
    constrained by more than one viewpoint.
    """
    import torch

    array, sx, sy = _preprocess(scenes)
    model = _model(model_id, device)

    images = torch.from_numpy(array).permute(0, 3, 1, 2).to(device)  # (S,3,H,W)
    autocast_dtype = torch.float32
    if device.startswith("cuda"):
        autocast_dtype = (
            torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        )

    with torch.no_grad():
        with torch.autocast(
            device_type="cuda" if device.startswith("cuda") else "cpu",
            dtype=autocast_dtype,
            enabled=device.startswith("cuda"),
        ):
            predictions = model(images[None])  # (B,S,3,H,W)

    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    # Frame 0: its extrinsic is the identity by construction, so VGGT's world
    # frame *is* that camera's frame and its depth back-projects straight into
    # camera coordinates. The other views move the prediction but are never
    # measured directly - doing that would mean transforming their depth through
    # their extrinsics first.
    _extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    K = intrinsic[0, 0].float().cpu().numpy().astype(np.float64)
    depth = predictions["depth"][0, 0, ..., 0].float().cpu().numpy()
    conf = predictions["depth_conf"][0, 0].float().cpu().numpy()

    # K is at VGGT's resolution; stretch it to the photograph's.
    K = K.copy()
    K[0, :] *= sx
    K[1, :] *= sy
    return K, depth, conf, (sx, sy)


# --------------------------------------------------------------------------- #
# floor plane
# --------------------------------------------------------------------------- #
def _camera_points(depth, K, sx, sy):
    """Back-project the depth map into camera coordinates (x right, y down, z forward)."""
    height, width = depth.shape
    v, u = np.mgrid[0:height, 0:width].astype(np.float64)
    # The depth map is at VGGT's resolution and K at the photograph's, so put
    # the pixel grid into the photograph's coordinates before unprojecting.
    u = (u + 0.5) * sx
    v = (v + 0.5) * sy
    x = (u - K[0, 2]) * depth / K[0, 0]
    y = (v - K[1, 2]) * depth / K[1, 1]
    return np.stack([x, y, depth], -1)


def _fit_floor(points, conf, seed=0):
    """RANSAC the dominant near-horizontal plane below the camera.

    Candidates are restricted to the lower half of the frame and to normals
    within ~45 degrees of the image's vertical axis, which is what separates
    the floor from a wall or a table top that happens to be larger in view.
    """
    height, width, _ = points.shape
    band = points[int(height * 0.45) :].reshape(-1, 3)
    weights = conf[int(height * 0.45) :].reshape(-1)

    keep = np.isfinite(band).all(1) & (band[:, 2] > 1e-3) & (weights > MIN_CONFIDENCE)
    band = band[keep]
    if len(band) < 200:
        raise ValueError(
            "VGGT returned too little confident depth below the camera to fit a "
            "floor — is this a photograph of a room interior?"
        )

    rng = np.random.default_rng(seed)
    scale = float(np.median(np.abs(band[:, 1])) + 1e-6)  # rough camera height
    tolerance = FLOOR_TOLERANCE * scale * 3

    best_n, best_d, best_count = None, None, 0
    for _ in range(600):
        trio = band[rng.choice(len(band), 3, replace=False)]
        n = np.cross(trio[1] - trio[0], trio[2] - trio[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if abs(n[1]) < 0.7:  # not a floor: too far from horizontal
            continue
        d = -float(n @ trio[0])
        count = int((np.abs(band @ n + d) < tolerance).sum())
        if count > best_count:
            best_n, best_d, best_count = n, d, count

    if best_n is None or best_count < len(band) * 0.05:
        raise ValueError("no floor plane found in the scene's depth")

    # Least-squares refit on the inliers: RANSAC picks the support, three
    # points pick a poor normal.
    inliers = band[np.abs(band @ best_n + best_d) < tolerance]
    centroid = inliers.mean(0)
    _u, _s, vt = np.linalg.svd(inliers - centroid)
    n = vt[-1]
    d = -float(n @ centroid)

    if n[1] > 0:  # point the normal up (camera y is down)
        n, d = -n, -d
    if d <= 1e-6:
        raise ValueError("the fitted floor is level with or above the camera")
    return n / np.linalg.norm(n), d


def _floor_profile(points, n, d, conf, width, sx):
    """Topmost floor pixel per column, in the photograph's coordinates.

    The successor to `floor_line`: where the floor stops being visible going up
    a column is the wall junction, and unlike a straight measured line it
    follows a bay, a step or a piece of furniture standing against the wall.
    """
    on_plane = np.abs(points @ n + d) < FLOOR_TOLERANCE * d
    on_plane &= conf > MIN_CONFIDENCE
    rows, cols = points.shape[:2]

    # argmax on a boolean picks the first True; columns with none get NaN.
    first = np.argmax(on_plane, axis=0).astype(np.float64)
    first[~on_plane.any(axis=0)] = np.nan

    xs_small = (np.arange(cols) + 0.5) * sx
    known = np.isfinite(first)
    if known.sum() < max(4, cols * 0.05):
        raise ValueError("the floor plane is barely visible in this scene")

    # Fill the gaps a doorway or a dark corner leaves, then resample to full
    # width and smooth: a per-column argmax is noisy by a pixel or two.
    ys = np.interp(
        np.arange(width),
        xs_small[known],
        first[known] * (points.shape[0] / rows),
    )
    ys = ys * (1.0)  # rows are already at VGGT resolution; scaled below
    return ys, xs_small


# --------------------------------------------------------------------------- #
# the solved scene
# --------------------------------------------------------------------------- #
@dataclass
class SceneGeometry:
    """A room photograph solved into a camera and a floor plane.

    Replaces the old `Geometry`: no floor_line, no ceiling_line, no horizon_y
    to measure. `camera_height_m` survives, because it is the one number that
    gives VGGT's unit-less reconstruction a metric scale.
    """

    K: np.ndarray  # 3x3 intrinsics, photograph resolution
    normal: np.ndarray  # floor normal, unit, pointing up in camera coords
    distance: float  # camera to floor, in VGGT units
    camera_height_m: float
    size: tuple  # (width, height) of the photograph
    floor_v: np.ndarray  # per-column floor/wall junction, length = width
    model: str = DEFAULT_MODEL
    _K_inv: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self._K_inv = np.linalg.inv(self.K)

    # -- scale ------------------------------------------------------------- #
    @property
    def units_per_metre(self):
        return self.distance / self.camera_height_m

    # -- rays and the plane ------------------------------------------------ #
    def ray(self, x, y):
        return self._K_inv @ np.array([float(x), float(y), 1.0])

    def floor_point(self, x, y):
        """Where the ray through pixel (x, y) meets the floor, in VGGT units."""
        r = self.ray(x, y)
        denominator = float(self.normal @ r)
        if denominator > -1e-9:
            raise ValueError(
                f"pixel ({x:.0f}, {y:.0f}) is at or above the horizon — nothing "
                "on the floor projects there"
            )
        return r * (-self.distance / denominator)

    def horizon_y(self, x=None):
        """The floor plane's vanishing line, l = K^-T n. Kept for logging."""
        a, b, c = self._K_inv.T @ self.normal
        if abs(b) < 1e-12:
            return float("nan")
        u = self.size[0] / 2 if x is None else float(x)
        return float(-(a * u + c) / b)

    def floor_y_at(self, x):
        """Image y of the floor/wall junction under column `x`."""
        index = int(np.clip(round(float(x)), 0, self.size[0] - 1))
        return float(self.floor_v[index])

    @property
    def floor_span(self):
        """(x_min, x_max) of the columns where floor was actually reconstructed."""
        visible = np.nonzero(np.isfinite(self.floor_v))[0]
        return float(visible.min()), float(visible.max())

    def depth_m(self, center_x, forward_offset_px=0.0):
        """Distance from the camera to where an object stands, in metres.

        The depth-sort key. The old code sorted on base y in pixels, which is
        monotonic in depth only for a level camera; this is the real thing.
        """
        base_y = self.floor_y_at(center_x) + float(forward_offset_px)
        point = self.floor_point(center_x, base_y)
        return float(np.linalg.norm(point)) / self.units_per_metre

    # -- placement --------------------------------------------------------- #
    def _project(self, point):
        p = self.K @ point
        if p[2] <= 1e-9:
            raise ValueError("an object corner falls behind the camera")
        return p[0] / p[2], p[1] / p[2]

    def object_quad(self, center_x, width_m, height_m, forward_offset_px=0.0):
        """Corners of an upright, camera-facing rectangle standing on the floor.

        Returns (quad, px_per_metre) with the quad in TL, TR, BR, BL order, to
        match the corner order of the cutout being warped onto it.
        """
        if width_m <= 0 or height_m <= 0:
            raise ValueError("object size must be positive")

        base_y = self.floor_y_at(center_x) + float(forward_offset_px)
        base = self.floor_point(center_x, base_y)

        up = self.normal
        # The object faces the camera: its width axis is horizontal, lies in the
        # floor plane, and is perpendicular to the line of sight to its base.
        sight = base / (np.linalg.norm(base) + 1e-12)
        sight = sight - (sight @ up) * up
        norm = np.linalg.norm(sight)
        if norm < 1e-6:
            raise ValueError("the object stands directly below the camera")
        sight /= norm
        right = np.cross(sight, up)
        right /= np.linalg.norm(right)

        upm = self.units_per_metre
        half = right * (width_m / 2.0) * upm
        rise = up * height_m * upm

        bl, br = base - half, base + half
        if self._project(br)[0] < self._project(bl)[0]:  # keep image-left left
            bl, br = br, bl
        corners = [bl + rise, br + rise, br, bl]  # TL, TR, BR, BL
        quad = np.float32([self._project(c) for c in corners])

        left_px = float(np.linalg.norm(quad[0] - quad[3]))
        right_px = float(np.linalg.norm(quad[1] - quad[2]))
        return quad, (left_px + right_px) / 2.0 / height_m

    def warp(self, cutout, canvas_size, center_x, width_m, height_m, forward_offset_px=0.0):
        """Cutout warped onto its place in the room. Successor to `keystone_fit`.

        A homography is exact here: the cutout is a flat billboard, and the
        quad is the projection of the plane it stands in.
        """
        quad, ppm = self.object_quad(center_x, width_m, height_m, forward_offset_px)
        src = np.float32(
            [[0, 0], [cutout.width, 0], [cutout.width, cutout.height], [0, cutout.height]]
        )
        M = cv2.getPerspectiveTransform(src, quad)
        warped = cv2.warpPerspective(
            np.array(cutout),
            M,
            canvas_size,
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        return Image.fromarray(warped, "RGBA"), ppm


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def _open(source):
    image = Image.open(source)
    image.load()
    return image


def _cache_key(scenes, model_id):
    """Hash every view: a second view changes the answer for frame 0."""
    digest = hashlib.sha1()
    parts = []
    for scene in scenes:
        rgb = scene.convert("RGB")
        digest.update(rgb.tobytes())
        parts.append(f"{rgb.width}x{rgb.height}")
    return f"{model_id}:{'+'.join(parts)}:{digest.hexdigest()}"


def solve_scene(scene, camera_height_m=1.10, model=DEFAULT_MODEL, device=None, verbose=False):
    """Solve a room photograph into a `SceneGeometry`.

    `scene` is a PIL image, a path, or a list of either: several views of the
    *same* room, of which the first is the one being composited into and the
    only one measured. The rest are extra evidence for the reconstruction.

    `camera_height_m` is the only measurement left: it scales VGGT's unit-less
    reconstruction to metres, and lowering it makes every object larger.
    """
    scenes = scene if isinstance(scene, (list, tuple)) else [scene]
    if not scenes:
        raise ValueError("give at least one room photograph")
    scenes = [s if isinstance(s, Image.Image) else _open(s) for s in scenes]
    scene = scenes[0]
    if camera_height_m <= 0:
        raise ValueError("camera_height_m must be positive")

    key = _cache_key(scenes, model)
    with _SOLVED_LOCK:
        solved = _SOLVED.get(key)
        if solved is not None:
            _SOLVED.move_to_end(key)

    if solved is None:
        device = _resolve_device(device)
        K, depth, conf, (sx, sy) = _predict(scenes, model, device)
        points = _camera_points(depth, K, sx, sy)
        normal, distance = _fit_floor(points, conf)

        rows_small = points.shape[0]
        first, xs_small = _floor_profile(points, normal, distance, conf, scene.width, sx)
        # `first` came back in VGGT rows; scale it to the photograph's rows.
        floor_v = first * sy
        floor_v = np.clip(floor_v, 0, scene.height - 1)
        solved = (K, normal, distance, floor_v)
        with _SOLVED_LOCK:
            _SOLVED[key] = solved
            while len(_SOLVED) > _SOLVED_MAX:
                _SOLVED.popitem(last=False)
        if verbose:
            print(
                f"vggt: f={K[0, 0]:.0f}px horizon_y="
                f"{-(np.linalg.inv(K).T @ normal)[0] * 0 or 0:.0f} "
                f"floor rows {floor_v.min():.0f}-{floor_v.max():.0f} "
                f"({rows_small} depth rows)"
            )

    K, normal, distance, floor_v = solved
    geometry = SceneGeometry(
        K=K,
        normal=normal,
        distance=distance,
        camera_height_m=float(camera_height_m),
        size=scene.size,
        floor_v=floor_v,
        model=model,
    )
    if verbose:
        print(
            f"vggt: horizon_y={geometry.horizon_y():.0f} "
            f"scale={geometry.units_per_metre:.3f} units/m"
        )
    return geometry
