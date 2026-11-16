r"""Solve one room from three photographs using app/vggt_scene.py.

Edit IMAGE_PATHS and CAMERA_HEIGHT_M below, then run from the project root:
    .venv\Scripts\python.exe scripts/run_vggt_scene.py

All three photographs must show the same room. Geometry is returned for the
first photograph; the other two provide additional views. The trained VGGT
weights download on the first run, which requires an internet connection.
"""

import json
from pathlib import Path
import sys


# EDIT THESE THREE PATHS. Keep the r prefix for Windows backslashes.
IMAGE_PATHS = [
    r"C:\Users\stefa\Downloads\room1_empty.png",  # Main view: geometry is measured here.
    r"C:\Users\stefa\Downloads\room2_empty.png",
    r"C:\Users\stefa\Downloads\room3_empty.png",
]

# Height of the camera above the floor for the FIRST photograph, in metres.
PERSON_HEIGHT_M = 1.75
CHEST_HEIGHT_RATIO = 0.73
CAMERA_HEIGHT_M = PERSON_HEIGHT_M * CHEST_HEIGHT_RATIO

# AMD ROCm also uses the name "cuda". Use "cpu" for CPU execution or None
# for automatic GPU selection with a CPU fallback.
DEVICE = "cuda"

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(IMAGE_PATHS) != 3:
        raise ValueError("Set IMAGE_PATHS to exactly three room photographs.")
    if CAMERA_HEIGHT_M <= 0:
        raise ValueError("CAMERA_HEIGHT_M must be positive.")

    paths = []
    for value in IMAGE_PATHS:
        path = Path(value).expanduser()
        # Relative paths resolve from the project root, regardless of cwd.
        paths.append(path if path.is_absolute() else PROJECT_ROOT / path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Edit IMAGE_PATHS at the top of this script. Files not found:\n"
            + "\n".join(missing)
        )

    # Allow direct execution from scripts/ or from an IDE.
    sys.path.insert(0, str(PROJECT_ROOT))
    import torch
    from app.vggt_scene import solve_scene
    from app.prompt_builder import room_block

    if DEVICE and DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU unavailable. Run with the project's ROCm .venv, "
            "or change DEVICE to 'cpu'."
        )

    print("Solving the room from these three views:", flush=True)
    for index, path in enumerate(paths, start=1):
        print(f"  {index}: {path}", flush=True)

    geometry = solve_scene(
        paths,
        camera_height_m=CAMERA_HEIGHT_M,
        device=DEVICE,
        verbose=True,
    )

    print(json.dumps({
        "main_image": str(paths[0]),
        "image_size_px": list(geometry.size),
        "camera_height_m": geometry.camera_height_m,
        "camera_intrinsics": geometry.K.tolist(),
        "floor_normal": geometry.normal.tolist(),
        "camera_to_floor_model_units": float(geometry.distance),
        "model_units_per_metre": float(geometry.units_per_metre),
        "horizon_y_at_image_center_px": float(geometry.horizon_y()),
        "room_dimensions": geometry.room_dimensions,
        "flux_room_prompt": room_block(geometry, paths[0].name),
    }, indent=2))
    return geometry


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        print(f"VGGT scene failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
