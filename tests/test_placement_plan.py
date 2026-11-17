"""Exercise measured wall placement, not a camera-facing billboard fallback."""

import io
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from app import generate_routes as routes
from app.composite import Placement, compose
from app.placement_plan import Wall, fit_walls, plan_placements, PlacementConflict
from app.scene_validation import validate_placement
from app.errors import AppError
from app.vggt_scene import SceneGeometry


def room():
    up = np.array([0., -1., 0.])
    inward = np.array([.5, 0., -np.sqrt(.75)])
    wall = Wall("wall_1", np.array([0., 1., 4.]), np.cross(up, inward), inward, -2., 2.)
    return SceneGeometry(np.array([[350., 0., 400.], [0., 350., 300.], [0., 0., 1.]]),
                         up, 1., 1., (800, 600), np.full(800, 400.), walls=[wall])


def furniture(width=1., depth=.6):
    return Placement(Image.new("RGBA", (100, 100), "red"), 0., width, .8,
                     depth_size_m=depth)


def test_oblique_wall_controls_rotation_and_perspective():
    geometry = room()
    item = furniture()
    plan_placements([item], [], geometry)
    assert np.allclose(item.right_axis, geometry.walls[0].right)
    view_right = np.cross(geometry.normal, item.base_m)
    view_right /= np.linalg.norm(view_right)
    assert abs(view_right @ item.right_axis) < .99
    quad, _ = geometry.object_quad(None, 1., .8, base_m=item.base_m,
                                   right_axis=item.right_axis)
    # Wall tangent has depth: its two projected vertical edges have different scales.
    assert not np.isclose(np.linalg.norm(quad[0]-quad[3]), np.linalg.norm(quad[1]-quad[2]))
    result = compose(Image.new("RGB", geometry.size, "white"), [item], geometry)
    assert result.mask.getbbox() is not None
    assert result.placement_plan[0]["wall_id"] == "wall_1"


def test_detects_supported_wall_from_cloud():
    x, y = np.meshgrid(np.linspace(-3., 3., 40), np.linspace(-1.8, 1., 30))
    points = np.stack([x, y, np.full_like(x, 4.)], -1)
    walls = fit_walls(points, np.full(x.shape, 3.), np.array([0., -1., 0.]), 1., 1.)
    assert len(walls) == 1
    assert np.allclose(walls[0].inward, [0, 0, -1], atol=.001)
    assert np.allclose(walls[0].right, [1, 0, 0], atol=.001)
    assert walls[0].end_m - walls[0].start_m > 5


def test_missing_wall_never_becomes_camera_facing():
    geometry = room()
    geometry.walls = []
    with pytest.raises(PlacementConflict):
        plan_placements([furniture()], [], geometry)
    with pytest.raises(ValueError, match="wall orientation"):
        geometry.object_quad(400, 1., .8)


@pytest.mark.parametrize("width,depth,reason", [(5., .6, "no_supported_placement"),
                                               (1., None, "depth_required")])
def test_unplaceable_furniture_is_reported(width, depth, reason):
    with pytest.raises(PlacementConflict) as error:
        plan_placements([furniture(width, depth)], [], room())
    assert error.value.issues[0]["reason"] == reason


def test_same_requested_wall_position_cannot_overlap():
    with pytest.raises(PlacementConflict):
        plan_placements([furniture(), furniture()], [{"wall_offset_m": 0.}] * 2, room())


def test_wall_mount_elevation_and_deterministic_plan():
    geometry = room()
    first, second = furniture(), furniture()
    spec = [{"wall_id": "wall_1", "wall_offset_m": 0., "mount_height_m": .5}]
    plan_placements([first], spec, geometry)
    plan_placements([second], spec, geometry)
    assert np.allclose(first.base_m, second.base_m)
    assert np.isclose(first.base_m @ geometry.normal + geometry.camera_height_m, .5)


@pytest.mark.parametrize("spec", [{"depth_m": float("nan")}, {"wall_gap_m": -1.},
                                 {"wall_id": 12}, {"mount_height_m": float("inf")}])
def test_invalid_transform_fields_are_rejected(spec):
    with pytest.raises(AppError):
        validate_placement(spec, 0, require_size=False)


def test_plan_endpoint_returns_transforms_without_flux_or_billing(app, client, monkeypatch):
    geometry = room()
    item = furniture()
    plan_placements([item], [], geometry)
    result = compose(Image.new("RGB", geometry.size, "white"), [item], geometry)
    monkeypatch.setattr(app.extensions["verifier"], "verify", lambda token: {})
    monkeypatch.setattr(routes, "_composite", lambda *args: (result, geometry, ["room1.png"], []))
    account, flux = Mock(), Mock()
    monkeypatch.setattr(routes, "_client", lambda: account)
    monkeypatch.setattr(routes, "_flux", lambda: flux)
    scene = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(scene, format="PNG")
    scene.seek(0)
    response = client.post("/api/generate/plan", headers={"Authorization": "Bearer test"},
                           data={"image_ids": "1", "scene": (scene, "room.png")})
    assert response.status_code == 200
    assert response.json["placements"][0]["wall_id"] == "wall_1"
    assert response.json["walls"][0]["id"] == "wall_1"
    account.get_balance.assert_not_called()
    account.spend.assert_not_called()
    flux.generate.assert_not_called()


def test_conflict_stops_generation_before_flux_and_debit(app, client, monkeypatch):
    account, flux = Mock(), Mock()
    account.get_balance.return_value = 100
    monkeypatch.setattr(app.extensions["verifier"], "verify", lambda token: {})
    monkeypatch.setattr(routes, "_client", lambda: account)
    monkeypatch.setattr(routes, "_flux", lambda: flux)
    monkeypatch.setattr(routes, "_images", lambda: SimpleNamespace(fetch=lambda *args: []))
    monkeypatch.setattr(routes, "solve_scene", lambda *args, **kwargs: room())
    def conflict(*args):
        raise PlacementConflict([{"reason": "no_supported_placement"}])
    monkeypatch.setattr(routes, "_catalogue_placements", conflict)
    stream = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(stream, format="PNG")
    stream.seek(0)
    response = client.post("/api/generate", headers={"Authorization": "Bearer test"},
                           data={"prompt": "furnish", "image_ids": "1", "scene": (stream, "room.png")})
    # The conflict surfaces in the job, which inline mode has already run.
    assert response.status_code == 202
    assert response.json["status"] == "failed"
    assert response.json["error"]["code"] == "placement_conflict"
    flux.generate.assert_not_called()
    account.spend.assert_not_called()
