"""`edits` on /api/generate: positioned placements of catalogue pieces.

Each edit names an image_id and puts one copy of that piece at `position` (where
it stands on the floor, normalised image point); its `remove` outlines are
erased from the room photo first. Several edits may use the same piece.
"""

import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from app import generate_routes as routes
from app.composite import CompositeResult, Placement
from app.placement_plan import Wall

SQUARE = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]


def photo(colour, size=(32, 24)):
    stream = io.BytesIO()
    Image.new("RGB", size, colour).save(stream, format="PNG")
    stream.seek(0)
    return stream


def fake_geometry(walls=()):
    """Camera 1.5 units above a level floor, identity intrinsics, 1 unit = 1 m."""
    K = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
    normal = np.array([0.0, -1.0, 0.0])  # up is -y in camera coordinates

    def floor_point(x, y):
        ray = np.linalg.inv(K) @ np.array([x, y, 1.0])
        denominator = float(normal @ ray)
        if denominator > -1e-9:
            raise ValueError("above the horizon")
        return ray * (-1.5 / denominator)

    return SimpleNamespace(size=(100, 80), units_per_metre=1.0, normal=normal,
                           walls=list(walls), floor_point=floor_point)


def placement(label="chair"):
    return Placement(image=Image.new("RGBA", (4, 4)), center_x=0, width_m=0.8,
                     height_m=1.0, label=label)


# --------------------------------------------------------------------------- #
# _place_at
# --------------------------------------------------------------------------- #
def test_place_at_stands_piece_on_floor_under_the_point():
    item = placement()
    routes._place_at(item, (0.5, 0.75), fake_geometry())
    # Pixel (50, 60) is 20 px below the principal point: ray (0, .2, 1) meets
    # the floor 1.5 below the camera at z = 7.5.
    assert item.base_m == pytest.approx([0.0, 1.5, 7.5])
    assert item.right_axis == pytest.approx([1.0, 0.0, 0.0])
    assert item.wall_id is None and item.mount_height_m == 0


def test_place_at_aligns_with_nearest_wall():
    diagonal = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    wall = Wall("wall_1", origin=np.array([0.0, 1.5, 9.0]), right=diagonal,
                inward=np.array([0.0, 0.0, -1.0]), start_m=-3, end_m=3)
    item = placement()
    routes._place_at(item, (0.5, 0.75), fake_geometry([wall]))
    assert item.right_axis == pytest.approx(diagonal)


def test_place_at_above_horizon_is_422():
    with pytest.raises(routes.AppError) as raised:
        routes._place_at(placement(), (0.5, 0.2), fake_geometry())
    assert raised.value.status == 422 and raised.value.code == "position_off_floor"


FETCHED = [({"id": 1, "name": "chair", "widthCm": 80, "heightCm": 100},
            photo("red").getvalue()),
           ({"id": 2, "name": "lamp", "widthCm": 30, "heightCm": 150},
            photo("blue").getvalue())]


@pytest.fixture
def planner(monkeypatch):
    planner = Mock()
    monkeypatch.setattr(routes, "plan_placements", planner)
    monkeypatch.setattr(routes, "load_cutout", lambda stream: Image.new("RGBA", (4, 4)))
    return planner


def test_one_piece_placed_at_several_positions(planner):
    """Two armchairs replaced by the same catalogue chair, each where it was put."""
    edits = [{"image_id": 1, "position": [0.5, 0.75], "remove": []},
             {"image_id": 1, "position": [0.8, 0.9], "remove": []}]
    placements = routes._catalogue_placements(FETCHED, [], fake_geometry(), edits)
    chairs = [p for p in placements if p.label == "chair (id 1)"]
    assert len(chairs) == 2
    assert chairs[0].base_m == pytest.approx([0.0, 1.5, 7.5])
    assert chairs[1].base_m[0] > 0.5            # the second one further right
    assert chairs[0].image is chairs[1].image   # one cutout, not two
    # The lamp has no edit and is wall-planned; the chair is not.
    assert [p.label for p in planner.call_args.args[0]] == ["lamp (id 2)"]
    assert len(placements) == 3


def test_all_pieces_positioned_skips_planner(planner):
    edits = [{"image_id": 2, "position": [0.3, 0.8], "remove": []},
             {"image_id": 1, "position": [0.7, 0.8], "remove": []}]
    placements = routes._catalogue_placements(FETCHED, [], fake_geometry(), edits)
    planner.assert_not_called()
    assert [p.label for p in placements] == ["lamp (id 2)", "chair (id 1)"]


# --------------------------------------------------------------------------- #
# _erase
# --------------------------------------------------------------------------- #
def test_erase_paints_out_the_outline_only():
    room = np.zeros((40, 40, 3), np.uint8)
    room[:, :] = (200, 200, 200)
    room[14:26, 14:26] = (0, 0, 0)  # the old chair
    scene = Image.fromarray(room)
    erased = np.asarray(routes._erase(scene, [
        {"image_id": 1, "position": [0.5, 0.9],
         "remove": [[[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]]},
    ]))
    assert erased[20, 20].mean() > 150          # chair gone, filled from the wall
    assert (erased[0:5, 0:5] == 200).all()      # far away: untouched


def test_erase_without_outlines_returns_same_image():
    scene = Image.new("RGB", (10, 10))
    assert routes._erase(scene, [{"image_id": 1, "position": [0.5, 0.5], "remove": []}]) is scene
    assert routes._erase(scene, []) is scene


# --------------------------------------------------------------------------- #
# the endpoint
# --------------------------------------------------------------------------- #
@pytest.fixture
def wired(app, monkeypatch):
    account, flux = Mock(), Mock()
    account.get_balance.return_value = 100
    account.spend.return_value = 90
    image = Image.new("RGB", (32, 24), "purple")
    flux.generate.return_value = image
    composite = Mock(return_value=(CompositeResult(image, Image.new("L", image.size)),
                                   None, ["room1.png"],
                                   [({"id": 7, "name": "chair"}, photo("red").getvalue())]))
    monkeypatch.setattr(app.extensions["verifier"], "verify", lambda token: {})
    monkeypatch.setattr(routes, "_client", lambda: account)
    monkeypatch.setattr(routes, "_flux", lambda: flux)
    monkeypatch.setattr(routes, "_composite", composite)
    monkeypatch.setattr(routes, "build_prompt", lambda *args, **kwargs: "room prompt")
    return SimpleNamespace(account=account, flux=flux, composite=composite)


def generate(client, **fields):
    data = {"prompt": "swap the chair", "image_ids": "7",
            "scene": (photo("white"), "room.png"), **fields}
    return client.post("/api/generate", headers={"Authorization": "Bearer test"}, data=data)


def test_edits_reach_the_composite_and_the_prompt(client, wired):
    edits = [{"image_id": 7, "position": [0.62, 0.81], "remove": [SQUARE]},
             {"image_id": 7, "position": [0.2, 0.9], "remove": [SQUARE]}]
    response = generate(client, edits=json.dumps(edits))
    assert response.status_code == 202
    assert response.json["status"] == "done"
    assert wired.composite.call_args.args[6] == edits
    assert "# REPLACED FURNITURE" in wired.flux.generate.call_args.args[0]


def test_without_edits_nothing_is_erased(client, wired):
    assert generate(client).status_code == 202
    assert wired.composite.call_args.args[6] == []
    assert "# REPLACED FURNITURE" not in wired.flux.generate.call_args.args[0]


@pytest.mark.parametrize("edits", [
    [{"position": [0.5, 0.5]}],                                   # no image_id
    [{"image_id": 8, "position": [0.5, 0.5]}],                    # not in image_ids
    [{"image_id": "7", "position": [0.5, 0.5]}],
    [{"image_id": 7, "position": [0.5]}],
    [{"image_id": 7, "position": [0.5, 1.5]}],
    [{"image_id": 7, "position": [0.5, 0.5], "remove": [[[0.1, 0.1], [0.2, 0.2]]]}],
    [{"image_id": 7, "position": [0.5, 0.5], "remove": "square"}],
    [{"image_id": 7, "position": [0.5, 0.5]}] * 13,               # over MAX_EDITS
    [None],
    ["chair"],
    {"image_id": 7, "position": [0.5, 0.5]},
])
def test_bad_edits_rejected_before_balance_check(client, wired, edits):
    assert generate(client, edits=json.dumps(edits)).status_code == 400
    wired.account.get_balance.assert_not_called()
    wired.composite.assert_not_called()


# --------------------------------------------------------------------------- #
# what FLUX is told
# --------------------------------------------------------------------------- #
def real_geometry():
    from app.vggt_scene import SceneGeometry

    K = np.array([[700.0, 0, 400], [0, 700.0, 300], [0, 0, 1]])
    up = np.array([0, -np.cos(0.15), -np.sin(0.15)])
    return SceneGeometry(K=K, normal=up / np.linalg.norm(up), distance=1.3,
                         camera_height_m=1.3, size=(800, 600), floor_v=np.full(800, 350.0))


def test_fixed_positions_are_stated_in_frame_terms():
    geometry = real_geometry()
    item = placement("Sessel (id 67)")
    routes._place_at(item, (0.786, 0.686), geometry)
    plan = [{"label": item.label, "wall_id": None, "base_m": item.base_m.tolist(),
             "right_axis": item.right_axis.tolist(), "width_m": 0.8, "height_m": 1.0}]
    fetched = [({"id": 67, "name": "Sessel"}, b"")]
    text = routes._fixed_positions_prompt(plan, geometry, fetched, [67])
    assert text.startswith("# FIXED POSITIONS")
    assert "Sessel (id 67) (image 2)" in text
    assert "right of the frame" in text
    assert "79% from the left" in text and "69% from the top" in text


def test_wall_planned_pieces_are_not_fixed():
    plan = [{"label": "lamp (id 2)", "wall_id": "wall_1", "base_m": [0, 1, 3],
             "right_axis": [1, 0, 0], "width_m": 0.3, "height_m": 1.5}]
    assert routes._fixed_positions_prompt(plan, real_geometry(), [], [2]) == ""
    assert routes._fixed_positions_prompt(plan, None, [], [2]) == ""
