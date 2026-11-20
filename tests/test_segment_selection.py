"""POST /api/segment_selection: validation, the SAM call, and mask -> outline."""

import io
import json
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from app import sam_selection
from app.sam_selection import Selection, mask_to_polygons


def photo(size=(40, 30)):
    stream = io.BytesIO()
    Image.new("RGB", size, "white").save(stream, format="PNG")
    stream.seek(0)
    return stream


@pytest.fixture
def authed(app, monkeypatch):
    monkeypatch.setattr(app.extensions["verifier"], "verify", lambda token: {"id": "u1"})
    return {"Authorization": "Bearer test"}


def post(client, headers, **fields):
    data = {"image": (photo(), "room.png"), "photo_id": "photo-1",
            "box": json.dumps([0.1, 0.2, 0.6, 0.9])}
    data.update(fields)
    data = {k: v for k, v in data.items() if v is not None}
    return client.post("/api/segment_selection", headers=headers, data=data,
                       content_type="multipart/form-data")


def test_returns_outline_from_sam(client, authed, monkeypatch):
    segment = Mock(return_value=Selection(
        polygons=[[[0.1, 0.2], [0.6, 0.2], [0.6, 0.9]]], score=0.97,
        bounds=(0.1, 0.2, 0.6, 0.9), width=40, height=30,
    ))
    monkeypatch.setattr(sam_selection, "segment", segment)

    response = post(client, authed, points=json.dumps([[0.3, 0.5]]),
                    point_labels=json.dumps([0]))

    assert response.status_code == 200
    body = response.get_json()
    assert body["polygons"] == [[[0.1, 0.2], [0.6, 0.2], [0.6, 0.9]]]
    assert body["bounds"] == [0.1, 0.2, 0.6, 0.9]
    args, kwargs = segment.call_args
    assert args[1] == (0.1, 0.2, 0.6, 0.9)
    assert kwargs["points"] == [(0.3, 0.5)] and kwargs["point_labels"] == [0]
    assert kwargs["photo_id"] == "photo-1"


def test_requires_auth(client):
    assert post(client, {}).status_code == 401


@pytest.mark.parametrize("fields", [
    {"image": None},
    {"box": None},
    {"box": json.dumps([0.5, 0.5, 0.4, 0.9])},       # x1 < x0
    {"box": json.dumps([0, 0, 1.5, 1])},             # outside the image
    {"box": json.dumps([0, 0, 1])},                  # too short
    {"points": json.dumps([[0.5]])},
    {"points": json.dumps([[0.5, 0.5]]), "point_labels": json.dumps([2])},
])
def test_rejects_bad_requests(client, authed, monkeypatch, fields):
    segment = Mock()
    monkeypatch.setattr(sam_selection, "segment", segment)
    assert post(client, authed, **fields).status_code == 400
    segment.assert_not_called()


def test_missing_sam_is_503(client, authed, monkeypatch):
    monkeypatch.setattr(sam_selection, "segment", Mock(
        side_effect=sam_selection.SegmentationUnavailable("not installed")))
    response = post(client, authed)
    assert response.status_code == 503
    assert response.get_json()["error"] == "segmentation_unavailable"


def test_mask_to_polygons_keeps_object_and_drops_specks():
    mask = np.zeros((100, 200), dtype=bool)
    mask[20:80, 40:160] = True     # the sofa
    mask[2:4, 2:4] = True          # noise
    polygons = mask_to_polygons(mask)
    assert len(polygons) == 1
    xs = [p[0] for p in polygons[0]]
    ys = [p[1] for p in polygons[0]]
    assert min(xs) == pytest.approx(0.2, abs=0.01) and max(xs) == pytest.approx(0.795, abs=0.01)
    assert min(ys) == pytest.approx(0.2, abs=0.01) and max(ys) == pytest.approx(0.79, abs=0.01)


def test_mask_to_polygons_empty():
    assert mask_to_polygons(np.zeros((10, 10), dtype=bool)) == []


def test_embedding_cache_skips_encoder_for_same_photo():
    """Second box on the same photo restores the embedding instead of re-encoding."""
    sam_selection._EMBEDDINGS.clear()
    predictor = Mock()

    def set_image(array):
        predictor._features, predictor._orig_hw = {"image_embed": 1}, [array.shape[:2]]
    predictor.set_image.side_effect = set_image

    image = Image.new("RGB", (8, 6), "white")
    key = sam_selection._embedding_key("p1", image)
    assert sam_selection._set_image(predictor, image, key, cache_size=2) is False
    assert sam_selection._set_image(predictor, image, key, cache_size=2) is True
    assert predictor.set_image.call_count == 1
    assert predictor._is_image_set is True and predictor._features == {"image_embed": 1}
    assert sam_selection._embedding_key("p1", Image.new("RGB", (8, 6), "black")) != key
