"""The job pipeline around a generation: status, ownership, failures, retries."""

import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from app import generate_routes as routes
from app import job_runner
from app.composite import CompositeResult
from app.errors import UpstreamError
from app.flux import ContentModerated
from app.job_store import CHARGING, DONE, FAILED, QUEUED


def photo(colour="red"):
    stream = io.BytesIO()
    Image.new("RGB", (32, 24), colour).save(stream, format="PNG")
    stream.seek(0)
    return stream


@pytest.fixture
def pipeline(app, monkeypatch):
    """Stub every external call; returns the mocks to steer and inspect."""
    result = Image.new("RGB", (32, 24), "purple")
    account, flux = Mock(), Mock()
    account.get_balance.return_value = 100
    account.spend.return_value = 93
    flux.generate.return_value = result
    user = {"id": "user-1"}
    monkeypatch.setattr(app.extensions["verifier"], "verify", lambda token: user)
    monkeypatch.setattr(routes, "_client", lambda: account)
    monkeypatch.setattr(routes, "_flux", lambda: flux)
    monkeypatch.setattr(routes, "_images", lambda: SimpleNamespace(
        fetch=lambda token, ids: [({"id": 1, "name": "chair"}, photo("orange").getvalue())]))
    monkeypatch.setattr(routes, "solve_scene", Mock(return_value=object()))
    monkeypatch.setattr(routes, "_catalogue_placements", Mock(return_value=[]))
    monkeypatch.setattr(routes, "compose", Mock(return_value=CompositeResult(
        image=result, mask=Image.new("L", result.size, 0))))
    monkeypatch.setattr(routes, "build_prompt", Mock(return_value="room prompt"))
    return SimpleNamespace(account=account, flux=flux, user=user, result=result)


def submit(client):
    return client.post("/api/generate", headers={"Authorization": "Bearer t"}, data={
        "prompt": "Place the chair", "image_ids": "1", "scene": (photo(), "room.png")})


def test_generate_answers_202_and_the_job_reports_the_image(client, pipeline):
    response = submit(client)
    assert response.status_code == 202
    job_id = response.json["jobId"]
    assert response.headers["Location"] == f"/api/jobs/{job_id}"

    status = client.get(f"/api/jobs/{job_id}", headers={"Authorization": "Bearer t"})
    assert status.status_code == 200
    assert status.json["status"] == DONE
    assert status.json["tokensSpent"] == response.json["cost"]
    assert status.json["balance"] == 93
    # Internals never leave the server.
    assert "params" not in status.json and "inputs" not in status.json

    image = client.get(status.json["imageUrl"], headers={"Authorization": "Bearer t"})
    assert image.status_code == 200
    assert image.mimetype == "image/png"
    assert Image.open(io.BytesIO(image.data)).getpixel((0, 0)) == (128, 0, 128)


def test_a_job_is_invisible_to_other_users(client, pipeline):
    job_id = submit(client).json["jobId"]
    pipeline.user["id"] = "someone-else"
    for path in (f"/api/jobs/{job_id}", f"/api/jobs/{job_id}/image", "/api/jobs/nope"):
        assert client.get(path, headers={"Authorization": "Bearer t"}).status_code == 404


def test_a_moderated_generation_fails_the_job_and_costs_nothing(client, pipeline):
    pipeline.flux.generate.side_effect = ContentModerated("refused")
    response = submit(client)
    assert response.status_code == 202
    assert response.json["status"] == FAILED
    assert response.json["error"]["code"] == "content_moderated"
    pipeline.account.spend.assert_not_called()


def test_upstream_trouble_asks_for_a_retry_until_the_last_attempt(app, client, pipeline,
                                                                  monkeypatch):
    # Queue without running, so the runner can be driven attempt by attempt.
    monkeypatch.setattr(app.extensions["dispatcher"], "dispatch", lambda *args: None)
    job_id = submit(client).json["jobId"]
    jobs = app.extensions["jobs"]
    pipeline.flux.generate.side_effect = UpstreamError("FLUX is down")

    with app.app_context():
        assert job_runner.run_job(job_id, "t", attempt=1, max_attempts=3) is False
        assert jobs.get(job_id)["status"] == QUEUED
        assert job_runner.run_job(job_id, "t", attempt=3, max_attempts=3) is True
    assert jobs.get(job_id)["status"] == FAILED
    pipeline.account.spend.assert_not_called()


def test_a_retry_after_a_failed_debit_does_not_pay_flux_twice(app, client, pipeline,
                                                             monkeypatch):
    monkeypatch.setattr(app.extensions["dispatcher"], "dispatch", lambda *args: None)
    job_id = submit(client).json["jobId"]
    jobs = app.extensions["jobs"]
    pipeline.account.spend.side_effect = [UpstreamError("AuthAndSecurity down"), 80]

    with app.app_context():
        assert job_runner.run_job(job_id, "t", attempt=1, max_attempts=3) is False
        assert jobs.get(job_id)["status"] == CHARGING
        assert job_runner.run_job(job_id, "t", attempt=2, max_attempts=3) is True

    job = jobs.get(job_id)
    assert job["status"] == DONE and job["result"]["balance"] == 80
    pipeline.flux.generate.assert_called_once()


def test_a_finished_job_delivered_again_is_not_rerun(app, client, pipeline):
    job_id = submit(client).json["jobId"]
    with app.app_context():
        assert job_runner.run_job(job_id, "t") is True
    pipeline.flux.generate.assert_called_once()
    pipeline.account.spend.assert_called_once()


def test_job_params_round_trip_through_the_store(app, client, pipeline, monkeypatch):
    monkeypatch.setattr(app.extensions["dispatcher"], "dispatch", lambda *args: None)
    response = client.post("/api/generate", headers={"Authorization": "Bearer t"}, data={
        "prompt": "Place the chair", "image_ids": "1", "user_height": "1.70", "seed": "42",
        "scene": (photo(), "room.png"), "references": (photo("blue"), "extra.png")})
    job = app.extensions["jobs"].get(response.json["jobId"])
    params = json.loads(job["params"])
    assert params["seed"] == 42 and params["image_ids"] == [1]
    assert params["camera_height_m"] == pytest.approx(1.70 * .73)
    assert len(job["inputs"]["rooms"]) == 1 and len(job["inputs"]["references"]) == 1


def test_worker_endpoint_does_not_exist_outside_cloud_mode(client):
    response = client.post("/internal/jobs/run", json={"jobId": "x", "token": "t"})
    assert response.status_code == 404
