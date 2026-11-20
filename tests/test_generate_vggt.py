"""Verify multi-view uploads reach reconstruction before FLUX is called."""

import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from app import generate_routes as routes
from app.composite import CompositeResult


def photo(colour):
    stream = io.BytesIO()
    Image.new("RGB", (32, 24), colour).save(stream, format="PNG")
    stream.seek(0)
    return stream


@pytest.mark.parametrize("room_count, expected_status", [(3, 202), (4, 400)])
def test_room_uploads_reconstruct_and_composite_before_flux(
    app, client, monkeypatch, room_count, expected_status
):
    geometry = object()
    reconstruction = Mock(return_value=geometry)
    placements = [object()]
    composite_image = Image.new("RGB", (32, 24), "purple")
    compose = Mock(return_value=CompositeResult(
        image=composite_image, mask=Image.new("L", composite_image.size, 0)
    ))
    flux = Mock()
    flux.generate.return_value = composite_image
    account = Mock()
    account.get_balance.return_value = 100
    account.spend.return_value = 93
    monkeypatch.setattr(app.extensions["verifier"], "verify", lambda token: {})
    monkeypatch.setattr(routes, "_client", lambda: account)
    monkeypatch.setattr(routes, "_flux", lambda: flux)
    monkeypatch.setattr(routes, "_images", lambda: SimpleNamespace(
        fetch=lambda token, ids: [({"id": 1, "name": "chair"}, photo("orange").getvalue())]
    ))
    monkeypatch.setattr(routes, "solve_scene", reconstruction)
    monkeypatch.setattr(routes, "_catalogue_placements", Mock(return_value=placements))
    monkeypatch.setattr(routes, "compose", compose)
    monkeypatch.setattr(routes, "build_prompt", Mock(return_value="room prompt"))
    response = client.post("/api/generate", headers={"Authorization": "Bearer test"}, data={
        "prompt": "Place the chair", "image_ids": "1",
        "user_height": "1.70",
        "rooms": [(photo(colour), f"room{i}.png")
                  for i, colour in enumerate(["red", "green", "blue", "white"][:room_count])],
    })
    assert response.status_code == expected_status
    if expected_status == 400:
        reconstruction.assert_not_called()
        flux.generate.assert_not_called()
        account.spend.assert_not_called()
        return
    views = reconstruction.call_args.args[0]
    assert len(views) == 3
    assert views[0].getpixel((0, 0)) == (255, 0, 0)
    assert reconstruction.call_args.kwargs["camera_height_m"] == pytest.approx(1.70 * .73)
    compose.assert_called_once_with(views[0], placements, geometry)
    references = flux.generate.call_args.kwargs["references"]
    assert references[0] is composite_image
    assert references[1].getpixel((0, 0)) == (255, 165, 0)
    catalogue_refs = routes.build_prompt.call_args.args[1]
    assert catalogue_refs[0].image_index == 2
    assert "Image 2:" in flux.generate.call_args.args[0]
    assert response.json["status"] == "done"
    assert response.json["referencesUsed"] == 2
    account.spend.assert_called_once()


def test_original_products_precede_uploaded_refs_and_quote_matches(app, client, monkeypatch):
    account, flux = Mock(), Mock()
    flux.configured = True
    account.get_balance.return_value = 100
    account.spend.return_value = 90
    composite = Image.new("RGB", (32, 24), "purple")
    flux.generate.return_value = composite
    monkeypatch.setattr(app.extensions["verifier"], "verify", lambda token: {})
    monkeypatch.setattr(routes, "_client", lambda: account)
    monkeypatch.setattr(routes, "_flux", lambda: flux)
    products = [({"id": 67, "name": "armchair"}, photo("red").getvalue()),
                ({"id": 74, "name": "sofa"}, photo("green").getvalue())]
    monkeypatch.setattr(routes, "_composite", lambda *args: (
        CompositeResult(composite, Image.new("L", composite.size)), None, ["room1.png"], products))
    prompt_builder = Mock(return_value="room prompt")
    monkeypatch.setattr(routes, "build_prompt", prompt_builder)
    response = client.post("/api/generate", headers={"Authorization": "Bearer test"},
                           data={"prompt": "furnish", "image_ids": "67,74",
                                 "scene": (photo("white"), "room.png"),
                                 "references": (photo("blue"), "extra.png")})
    assert response.status_code == 202
    assert response.json["status"] == "done"
    assert [im.getpixel((0, 0)) for im in flux.generate.call_args.kwargs["references"]] == [
        (128, 0, 128), (255, 0, 0), (0, 128, 0), (0, 0, 255)]
    assert [ref.image_index for ref in prompt_builder.call_args.args[1]] == [2, 3]
    quote = client.get("/api/generate/quote?images=2&references=1",
                       headers={"Authorization": "Bearer test"})
    assert quote.status_code == 200
    assert quote.json["totalReferences"] == 4
    assert quote.json["cost"] == response.json["tokensSpent"]


def test_reference_overflow_rejected_without_dropping_products(app, client, monkeypatch):
    monkeypatch.setattr(app.extensions["verifier"], "verify", lambda token: {})
    account = Mock()
    monkeypatch.setattr(routes, "_client", lambda: account)
    response = client.post("/api/generate", headers={"Authorization": "Bearer test"},
                           data={"prompt": "furnish", "image_ids": "1,2,3,4,5,6,7,8"})
    assert response.status_code == 400
    account.get_balance.assert_not_called()


@pytest.mark.parametrize("endpoint", ["/api/generate", "/api/generate/plan"])
@pytest.mark.parametrize("height,expected", [("1.70", 1.241), (None, 1.2775)])
def test_both_endpoints_use_person_height_for_reconstruction(
    app, client, monkeypatch, endpoint, height, expected
):
    reconstruction = Mock(return_value=SimpleNamespace(walls=[]))
    account, flux = Mock(), Mock()
    account.get_balance.return_value = 100
    account.spend.return_value = 90
    image = Image.new("RGB", (32, 24), "purple")
    flux.generate.return_value = image
    monkeypatch.setattr(app.extensions["verifier"], "verify", lambda token: {})
    monkeypatch.setattr(routes, "_client", lambda: account)
    monkeypatch.setattr(routes, "_flux", lambda: flux)
    monkeypatch.setattr(routes, "_images", lambda: SimpleNamespace(fetch=lambda *args: []))
    monkeypatch.setattr(routes, "solve_scene", reconstruction)
    monkeypatch.setattr(routes, "_catalogue_placements", lambda *args: [])
    monkeypatch.setattr(routes, "compose", lambda *args: CompositeResult(
        image, Image.new("L", image.size, 0)))
    monkeypatch.setattr(routes, "build_prompt", lambda *args, **kwargs: "room prompt")
    data = {"prompt": "furnish", "image_ids": "1", "scene": (photo("red"), "room.png")}
    if height is not None:
        data["user_height"] = height
    response = client.post(endpoint, headers={"Authorization": "Bearer test"}, data=data)
    assert response.status_code == (200 if endpoint.endswith("/plan") else 202)
    assert reconstruction.call_args.kwargs["camera_height_m"] == pytest.approx(expected)
    if endpoint.endswith("/plan"):
        assert response.json["camera_height_m"] == pytest.approx(expected)
        flux.generate.assert_not_called()
        account.spend.assert_not_called()


@pytest.mark.parametrize("endpoint", ["/api/generate", "/api/generate/plan"])
@pytest.mark.parametrize("fields", [{"user_height": "nan"}, {"user_height": "inf"},
                                    {"user_height": "0"}, {"user_height": "170"},
                                    {"user_height": "hello"},
                                    {"geometry": '{"camera_height_m":1.1}'}])
def test_bad_person_height_rejected_before_external_work(app, client, monkeypatch, endpoint, fields):
    account, images, reconstruction, flux = Mock(), Mock(), Mock(), Mock()
    monkeypatch.setattr(app.extensions["verifier"], "verify", lambda token: {})
    monkeypatch.setattr(routes, "_client", lambda: account)
    monkeypatch.setattr(routes, "_images", lambda: images)
    monkeypatch.setattr(routes, "solve_scene", reconstruction)
    monkeypatch.setattr(routes, "_flux", lambda: flux)
    response = client.post(endpoint, headers={"Authorization": "Bearer test"},
                           data={"prompt": "furnish", "image_ids": "1", **fields})
    assert response.status_code == 400
    account.get_balance.assert_not_called()
    images.fetch.assert_not_called()
    reconstruction.assert_not_called()
    flux.generate.assert_not_called()


def test_failed_multi_view_reconstruction_retries_with_main_photo(app, client, monkeypatch):
    """Views of the far side of a room can leave VGGT without a floor in frame 0."""
    geometry = SimpleNamespace(walls=[])
    reconstruction = Mock(side_effect=[ValueError("too little confident depth"), geometry])
    account, flux = Mock(), Mock()
    account.get_balance.return_value = 100
    account.spend.return_value = 90
    image = Image.new("RGB", (32, 24), "purple")
    flux.generate.return_value = image
    monkeypatch.setattr(app.extensions["verifier"], "verify", lambda token: {})
    monkeypatch.setattr(routes, "_client", lambda: account)
    monkeypatch.setattr(routes, "_flux", lambda: flux)
    monkeypatch.setattr(routes, "_images", lambda: SimpleNamespace(fetch=lambda *args: []))
    monkeypatch.setattr(routes, "solve_scene", reconstruction)
    monkeypatch.setattr(routes, "_catalogue_placements", lambda *args: [])
    monkeypatch.setattr(routes, "compose", lambda *args: CompositeResult(
        image, Image.new("L", image.size, 0)))
    monkeypatch.setattr(routes, "build_prompt", lambda *args, **kwargs: "room prompt")
    response = client.post("/api/generate", headers={"Authorization": "Bearer test"}, data={
        "prompt": "furnish", "image_ids": "1",
        "rooms": [(photo(c), f"room{i}.png") for i, c in enumerate(["red", "green", "blue"])],
    })
    assert response.status_code == 202
    assert response.json["status"] == "done"
    first, second = reconstruction.call_args_list
    assert len(first.args[0]) == 3
    assert len(second.args[0]) == 1
    assert second.args[0][0].getpixel((0, 0)) == (255, 0, 0)


def test_single_photo_reconstruction_failure_is_still_422(app, client, monkeypatch):
    reconstruction = Mock(side_effect=ValueError("not a room"))
    account = Mock()
    account.get_balance.return_value = 100
    monkeypatch.setattr(app.extensions["verifier"], "verify", lambda token: {})
    monkeypatch.setattr(routes, "_client", lambda: account)
    monkeypatch.setattr(routes, "_images", lambda: SimpleNamespace(fetch=lambda *args: []))
    monkeypatch.setattr(routes, "solve_scene", reconstruction)
    response = client.post("/api/generate/plan", headers={"Authorization": "Bearer test"},
                           data={"image_ids": "1", "scene": (photo("red"), "room.png")})
    assert response.status_code == 422
    assert response.json["error"] == "scene_unsolvable"
    assert reconstruction.call_count == 1
