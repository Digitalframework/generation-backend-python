"""SAM 2.1: turn a box the user dragged around a piece of furniture into its outline.

Used by POST /api/segment_selection (app/generate_routes.py). The designer calls
it every time a box is released, so it has to answer in well under a second.
SAM splits the work accordingly:

    1. the image encoder embeds the whole photograph — the expensive part;
    2. the mask decoder turns that embedding plus a box (and optional clicks)
       into a mask — milliseconds.

Step 1 depends only on the photo, so embeddings are cached per photo (the
client's `photo_id` plus a hash of the pixels, so a reused id with a different
picture never hits a stale entry). Every further box on the same photo pays for
step 2 only.

GPU: the same device logic as VGGT in app/vggt_scene.py — AMD ROCm and NVIDIA
CUDA are both `cuda` to PyTorch, with fp16 autocast on ROCm, so the ROCm wheels
from requirements-rocm-windows.txt serve this model too. Without a GPU it runs
on CPU (seconds per photo for the embedding, still fast per box).

Install: `sam2==1.1.0` in requirements.txt. Its optional CUDA extension only
speeds up hole filling and is not needed; on Windows/ROCm install with
`SAM2_BUILD_CUDA=0` (see README.md). Weights come from the Hugging Face hub on
first use (SAM_MODEL, default facebook/sam2.1-hiera-large) and are cached there.
"""

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .vggt_scene import _autocast_dtype, _resolve_device

log = logging.getLogger(__name__)

DEFAULT_MODEL = "facebook/sam2.1-hiera-large"

# Outline simplification, as a fraction of each contour's perimeter. 0.002
# keeps chair legs and armrests while cutting a 4000-point contour to ~100.
SIMPLIFY_EPSILON = 0.002
# Islands smaller than this share of the largest region are noise, not furniture.
MIN_REGION_SHARE = 0.02

_PREDICTORS = {}
_PREDICTOR_LOCK = threading.Lock()
# The predictor holds the "current image" as state, so one request at a time
# per process. Separate from VGGT's lock: a segmentation must not wait for a
# several-second room reconstruction while the user is dragging boxes.
_INFERENCE_LOCK = threading.Lock()
_EMBEDDINGS = OrderedDict()


class SegmentationUnavailable(RuntimeError):
    """sam2 or one of its dependencies is not installed."""


@dataclass
class Selection:
    """Result in normalised coordinates (0..1, x right, y down)."""

    polygons: list = field(default_factory=list)  # [[[x, y], ...], ...], largest first
    score: float = 0.0
    bounds: tuple | None = None  # (x0, y0, x1, y1) of all polygons
    width: int = 0
    height: int = 0


def _predictor(model_id, device):
    """One SAM2ImagePredictor per (model, device), loaded on first use and kept."""
    key = (model_id, device)
    with _PREDICTOR_LOCK:
        if key not in _PREDICTORS:
            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError as exc:
                raise SegmentationUnavailable(
                    "SAM 2.1 is not installed. `pip install sam2==1.1.0` "
                    "(with SAM2_BUILD_CUDA=0 on Windows/ROCm). See README.md."
                ) from exc
            log.info("loading %s on %s", model_id, device)
            _PREDICTORS[key] = SAM2ImagePredictor.from_pretrained(model_id, device=device)
        return _PREDICTORS[key]


def _embedding_key(photo_id, image):
    digest = hashlib.sha1(image.tobytes()).hexdigest()[:16]
    return f"{photo_id or '-'}:{image.width}x{image.height}:{digest}"


def _set_image(predictor, image, key, cache_size):
    """Embed `image`, or restore its cached embedding into the predictor.

    SAM2ImagePredictor keeps the current image in four attributes that
    set_image() fills; restoring them is exactly what set_image() would have
    produced, without the encoder pass. Must run under _INFERENCE_LOCK.
    """
    cached = _EMBEDDINGS.get(key)
    if cached is not None:
        _EMBEDDINGS.move_to_end(key)
        predictor.reset_predictor()
        predictor._features, predictor._orig_hw = cached
        predictor._is_image_set = True
        predictor._is_batch = False
        return True

    predictor.set_image(np.asarray(image))
    if cache_size > 0:
        _EMBEDDINGS[key] = (predictor._features, predictor._orig_hw)
        while len(_EMBEDDINGS) > cache_size:
            _EMBEDDINGS.popitem(last=False)
    return False


def segment(image, box, points=None, point_labels=None, photo_id=None,
            model=DEFAULT_MODEL, device=None, cache_size=8):
    """Mask of the object inside `box`, as outline polygons.

    image         PIL image of the room
    box           (x0, y0, x1, y1), normalised 0..1
    points        optional [(x, y), ...] normalised — refinement clicks
    point_labels  1 = part of the object, 0 = not, one per point
    """
    import torch

    image = image.convert("RGB")
    width, height = image.size
    device = _resolve_device(device)
    predictor = _predictor(model, device)

    box_px = np.array([box[0] * width, box[1] * height, box[2] * width, box[3] * height],
                      dtype=np.float32)
    coords = labels = None
    if points:
        coords = np.array([[x * width, y * height] for x, y in points], dtype=np.float32)
        labels = np.array(point_labels, dtype=np.int32)

    key = _embedding_key(photo_id, image)
    on_gpu = device.startswith("cuda")
    with _INFERENCE_LOCK, torch.inference_mode(), torch.autocast(
        device_type="cuda" if on_gpu else "cpu",
        dtype=_autocast_dtype(device),
        enabled=on_gpu,
    ):
        cached = _set_image(predictor, image, key, cache_size)
        # One mask for a box prompt: SAM's own guidance, and its multimask
        # fallback still kicks in internally when the single mask is unstable.
        masks, scores, _ = predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box=box_px[None, :],
            multimask_output=False,
        )
    log.info("segmented %s box=%s points=%d embedding=%s score=%.3f",
             key, [round(v, 3) for v in box], len(points or []),
             "cached" if cached else "computed", float(scores[0]))

    mask = np.asarray(masks[0]) > 0.5
    polygons = mask_to_polygons(mask)
    selection = Selection(polygons=polygons, score=float(scores[0]), width=width, height=height)
    if polygons:
        xs = [p[0] for poly in polygons for p in poly]
        ys = [p[1] for poly in polygons for p in poly]
        selection.bounds = (min(xs), min(ys), max(xs), max(ys))
    return selection


def mask_to_polygons(mask, epsilon=SIMPLIFY_EPSILON, min_share=MIN_REGION_SHARE):
    """Boolean HxW mask -> outer outlines, normalised to 0..1, largest first.

    Only outer contours: a hole in a chair back is still the chair as far as
    replacing it goes, and the removal step wants the filled shape anyway.
    """
    import cv2

    height, width = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if not contours:
        return []

    largest = cv2.contourArea(contours[0])
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < largest * min_share:
            break
        simplified = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
        if len(simplified) < 3:
            continue
        polygons.append([
            [round(float(x) / width, 4), round(float(y) / height, 4)]
            for x, y in simplified.reshape(-1, 2)
        ])
    return polygons
