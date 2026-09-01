"""
Place product cutouts into a room photo with correct perspective and scale.

v4 changes:
  - the hand-measured geometry (baseboard + crown molding -> horizon) and the
    keystone warp derived from it are gone. `app/vggt_scene.py` reconstructs the
    room instead: VGGT predicts intrinsics and depth, the floor is RANSAC-fitted
    as a plane, and an object is a rectangle standing on that plane. Placement
    is plain 3D projection rather than a horizon proxy.
  - depth ordering uses true camera-space distance (`SceneGeometry.depth_m`)
    instead of base y in pixels, which is monotonic in depth only for a level
    camera.
  - `camera_height_m` survives as the one measurement, since it is what gives
    VGGT's unit-less reconstruction a metric scale.

v3 changes (from `working comp`):
  - takes N reference images instead of one. Each gets its own position, real
    size and forward offset; they share the room's geometry, since the camera
    and the floor are properties of the photograph, not of any object.
  - objects are drawn far-to-near, and each object's shadow is laid down
    immediately before it. A nearer object therefore occludes a farther one and
    can cast onto it, which compositing all shadows up front would not allow.
  - module-level config became arguments. `SCENE`/`CUTOUT` constants can't
    serve concurrent requests, and the CLI at the bottom keeps the old workflow.
  - cutouts load from file objects as well as paths, so an upload never needs
    to touch disk.

WHAT LIVES WHERE
    vggt_scene.SceneGeometry   camera, floor plane, metric scale, the warp
    this module                cutting out, shadowing, depth-ordered compositing

  pip install pillow numpy rembg onnxruntime
  # geometry, see vggt_scene.py:
  pip install torch "vggt @ git+https://github.com/facebookresearch/vggt"
"""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Colour the shadow multiplies toward - a warm near-black, not pure black.
SHADOW_COLOUR = (26, 20, 16)


@dataclass
class Placement:
    """One reference image and where it goes."""

    image: Image.Image  # RGBA cutout, already trimmed
    center_x: float  # where the object's centre sits horizontally
    width_m: float
    height_m: float
    forward_offset_px: float = 0.0  # push off the wall toward the camera
    light_from: str = "left"
    # Shadow and reflection are opt-in. FLUX relights the composite itself, and
    # a baked-in ellipse under a fixed light direction fights the lighting it
    # infers from the room - two shadows, or one pointing the wrong way, made
    # photographic. Pass a positive core or ambient to draw one anyway.
    shadow_core: float = 0.0  # darkness right at the feet
    shadow_ambient: float = 0.0  # wide soft pool
    glossy_floor: bool = False  # faint vertical reflection
    label: str = ""  # for error messages only

    @property
    def casts_shadow(self):
        return self.shadow_core > 0 or self.shadow_ambient > 0

    def depth_m(self, geometry):
        """Distance from the camera to where this object stands, for sorting."""
        return geometry.depth_m(self.center_x, self.forward_offset_px)


@dataclass
class CompositeResult:
    image: Image.Image  # RGB composite
    mask: Image.Image  # L, union of every object and its shadow
    scale: list = field(default_factory=list)  # px/m per placement, for logging


# --------------------------------------------------------------------------- #
# 1. cutout
# --------------------------------------------------------------------------- #
def load_cutout(source, auto_rembg=True):
    """RGBA trimmed to the object.

    `source` is a path or any file-like object, so an upload can be composited
    without being written to disk first.

    Guard is `min alpha >= 250`, not a percentage: a trimmed cutout of a boxy
    object is legitimately ~99% opaque.
    """
    img = Image.open(source)
    img.load()  # a SpooledTemporaryFile may close under us otherwise
    rgba = img.convert("RGBA")
    a = np.array(rgba.split()[-1])

    if a.min() >= 250:
        if not auto_rembg:
            raise ValueError("image is fully opaque - no cutout present")
        from rembg import new_session, remove

        rgba = remove(
            img.convert("RGB"),
            session=new_session("isnet-general-use"),
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=15,
            alpha_matting_erode_size=8,
        ).convert("RGBA")
        a = np.array(rgba.split()[-1])
        if a.min() >= 250:
            raise ValueError("background removal produced no transparency")

    box = rgba.split()[-1].point(lambda p: 255 if p > 8 else 0).getbbox()
    if box is None:
        raise ValueError("cutout is empty")
    return rgba.crop(box)


# --------------------------------------------------------------------------- #
# 2. shadow: tight core + wide ambient
# --------------------------------------------------------------------------- #
def floor_shadow(canvas_size, warped, light_from="left", core=0.55, ambient=0.28):
    a = np.array(warped.split()[-1])
    ys, xs = np.nonzero(a > 8)
    if len(xs) == 0:
        return Image.new("L", canvas_size, 0)

    x0, x1, base_y = int(xs.min()), int(xs.max()), int(ys.max())
    w = x1 - x0
    off = w * (0.07 if light_from == "left" else -0.07)
    cx = (x0 + x1) / 2

    core_img = Image.new("L", canvas_size, 0)
    ImageDraw.Draw(core_img).ellipse(
        [cx - w * 0.50, base_y - w * 0.020, cx + w * 0.50, base_y + w * 0.030],
        fill=int(255 * core),
    )
    core_img = core_img.filter(ImageFilter.GaussianBlur(max(3, w * 0.010)))

    amb = Image.new("L", canvas_size, 0)
    ImageDraw.Draw(amb).ellipse(
        [cx - w * 0.62 + off, base_y - w * 0.05, cx + w * 0.62 + off, base_y + w * 0.11],
        fill=int(255 * ambient),
    )
    amb = amb.filter(ImageFilter.GaussianBlur(max(8, w * 0.045)))

    return Image.fromarray(np.maximum(np.array(core_img), np.array(amb)))


def floor_reflection(scene_size, warped, strength=0.18, height_frac=0.32):
    """Faint mirrored smear below the object. Cheap, but glossy floors need it."""
    a = np.array(warped.split()[-1])
    ys, xs = np.nonzero(a > 8)
    if len(xs) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    patch = warped.crop((x0, y0, x1, y1))
    refl_h = max(1, int((y1 - y0) * height_frac))
    refl = patch.transpose(Image.FLIP_TOP_BOTTOM).crop((0, 0, patch.width, refl_h))
    refl = refl.resize((patch.width, refl_h), Image.LANCZOS)
    refl = refl.filter(ImageFilter.GaussianBlur(4))

    a2 = np.array(refl.split()[-1], np.float32)
    fade = np.linspace(strength, 0.0, refl_h)[:, None]
    refl.putalpha(Image.fromarray((a2 * fade).astype(np.uint8)))

    canvas = Image.new("RGBA", scene_size, (0, 0, 0, 0))
    canvas.paste(refl, (x0, y1), refl)
    return canvas


def _apply_shadow(scene_rgb, shadow):
    s = np.array(shadow, np.float32)[..., None] / 255.0
    out = np.array(scene_rgb, np.float32) * (1 - s) + np.array(SHADOW_COLOUR, np.float32) * s
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# 3. composite
# --------------------------------------------------------------------------- #
def compose(scene, placements, geometry, verbose=False):
    """Composite every placement into `scene`, far to near.

    `geometry` is a `vggt_scene.SceneGeometry`: it owns the camera, the floor
    plane and the warp, so this function only decides order and lighting.

    Shadow and reflection are drawn only for placements that ask for them; see
    `Placement`. By default nothing is baked in and FLUX does the lighting.

    Depth order is what makes more than one object work: an object standing
    closer to the camera occludes a farther one, so they are drawn in order of
    decreasing distance. Each shadow goes down immediately before its object,
    so it can fall across whatever is behind.
    """
    if not placements:
        raise ValueError("no placements given")

    scene = scene.convert("RGB")
    comp = scene
    mask = Image.new("L", scene.size, 0)
    scales = []

    ordered = sorted(placements, key=lambda p: p.depth_m(geometry), reverse=True)

    for item in ordered:
        warped, ppm = geometry.warp(
            item.image,
            scene.size,
            item.center_x,
            item.width_m,
            item.height_m,
            item.forward_offset_px,
        )
        scales.append(ppm)
        if verbose:
            print(
                f"{item.label or 'object'}: {ppm:.0f} px/m, "
                f"{item.depth_m(geometry):.2f} m from the camera"
            )

        shadow = None
        if item.casts_shadow:
            shadow = floor_shadow(
                scene.size, warped, item.light_from, item.shadow_core, item.shadow_ambient
            )
            comp = _apply_shadow(comp, shadow)

        if item.glossy_floor:
            refl = floor_reflection(scene.size, warped)
            if refl is not None:
                comp = Image.alpha_composite(comp.convert("RGBA"), refl).convert("RGB")

        comp.paste(warped, (0, 0), warped)

        # The mask is the union of what was actually drawn, so with shadows off
        # it is the objects alone.
        layer = np.array(warped.split()[-1])
        if shadow is not None:
            layer = np.maximum(layer, np.array(shadow))
        mask = Image.fromarray(np.maximum(np.array(mask), layer))

    return CompositeResult(image=comp, mask=mask, scale=scales)


# --------------------------------------------------------------------------- #
# CLI - no points to measure now, just the camera height
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Relative when run as `python -m app.composite`, plain from inside app/.
    try:
        from .vggt_scene import solve_scene
    except ImportError:
        from vggt_scene import solve_scene

    SCENE = "scene.png"
    CAMERA_HEIGHT_M = 1.10  # LOWER THIS TO MAKE OBJECTS BIGGER
    OBJECTS = [
        # path, center_x, width_m, height_m, forward_offset_px
        ("console.png", 615, 1.65, 0.76, 14),
        ("lamp.png", 300, 0.40, 1.55, 40),
    ]

    scene_img = Image.open(SCENE).convert("RGB")
    geometry = solve_scene(scene_img, camera_height_m=CAMERA_HEIGHT_M, verbose=True)
    result = compose(
        scene_img,
        [
            Placement(
                image=load_cutout(path),
                center_x=cx,
                width_m=w,
                height_m=h,
                forward_offset_px=off,
                label=path,
            )
            for path, cx, w, h, off in OBJECTS
        ],
        geometry,
        verbose=True,
    )

    result.image.save("composite.png")
    result.mask.save("composite_mask.png")
    print("wrote composite.png + composite_mask.png")
