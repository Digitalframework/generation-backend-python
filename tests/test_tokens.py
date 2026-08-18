"""Balance reads and spends against AuthAndSecurity, with the HTTP calls stubbed."""

import pytest
import responses

from config import TestConfig

BALANCE_URL = f"{TestConfig.AUTH_BASE_URL}{TestConfig.BALANCE_PATH}"
SPEND_URL = f"{TestConfig.AUTH_BASE_URL}{TestConfig.SPEND_PATH}"


@responses.activate
def test_balance_is_returned(client, auth):
    responses.add(responses.GET, BALANCE_URL, json={"balance": 250}, status=200)

    resp = client.get("/api/tokens/balance", headers=auth)

    assert resp.status_code == 200
    assert resp.get_json()["balance"] == 250


@responses.activate
def test_current_user_shape_is_also_accepted(client, auth):
    """CurrentUser names the field `tokenBalance`; a dedicated endpoint may not."""
    responses.add(responses.GET, BALANCE_URL, json={"tokenBalance": 12}, status=200)

    resp = client.get("/api/tokens/balance", headers=auth)

    assert resp.get_json()["balance"] == 12


@responses.activate
def test_caller_token_is_forwarded(client, mint):
    token = mint()
    responses.add(responses.GET, BALANCE_URL, json={"balance": 1}, status=200)

    client.get("/api/tokens/balance", headers={"Authorization": f"Bearer {token}"})

    assert responses.calls[0].request.headers["Authorization"] == f"Bearer {token}"


@responses.activate
@pytest.mark.parametrize(
    ("balance", "sufficient"),
    [(10, True), (5, True), (3, False), (0, False)],
)
def test_check_reports_sufficiency(client, auth, balance, sufficient):
    responses.add(responses.GET, BALANCE_URL, json={"balance": balance}, status=200)

    resp = client.get("/api/tokens/check?amount=5", headers=auth)

    assert resp.get_json()["sufficient"] is sufficient


@responses.activate
def test_check_rejects_a_bad_amount(client, auth):
    resp = client.get("/api/tokens/check?amount=-1", headers=auth)
    assert resp.status_code == 400


@responses.activate
def test_spend_debits_and_returns_remaining(client, auth):
    responses.add(responses.POST, SPEND_URL, json={"balance": 95}, status=200)

    resp = client.post("/api/tokens/spend", json={"amount": 5, "reason": "image"}, headers=auth)

    assert resp.status_code == 200
    assert resp.get_json()["spent"] == 5
    assert resp.get_json()["balance"] == 95
    assert responses.calls[0].request.body == b'{"amount": 5, "reason": "image"}'


@responses.activate
def test_spend_rejects_non_positive_amounts(client, auth):
    resp = client.post("/api/tokens/spend", json={"amount": 0}, headers=auth)
    assert resp.status_code == 400


@responses.activate
def test_spend_rejects_a_boolean_amount(client, auth):
    """`True` is an int in Python; it must not read as "spend one"."""
    resp = client.post("/api/tokens/spend", json={"amount": True}, headers=auth)
    assert resp.status_code == 400


@responses.activate
def test_insufficient_balance_upstream_is_402(client, auth):
    responses.add(responses.POST, SPEND_URL, json={"detail": "Not enough tokens"}, status=409)

    resp = client.post("/api/tokens/spend", json={"amount": 500}, headers=auth)

    assert resp.status_code == 402
    assert resp.get_json()["error"] == "insufficient_tokens"


@responses.activate
def test_upstream_failure_is_502(client, auth):
    responses.add(responses.GET, BALANCE_URL, status=500)

    resp = client.get("/api/tokens/balance", headers=auth)

    assert resp.status_code == 502
    assert resp.get_json()["error"] == "upstream_unavailable"


@responses.activate
def test_redirect_to_login_is_reported_clearly(client, auth):
    """What today's SecurityConfig actually does: bounce a bearer call to /login."""
    responses.add(responses.GET, BALANCE_URL, status=302, headers={"Location": "/login"})

    resp = client.get("/api/tokens/balance", headers=auth)

    assert resp.status_code == 502
    assert "does not accept bearer tokens" in resp.get_json()["message"]


@responses.activate
def test_html_response_is_not_mistaken_for_a_balance(client, auth):
    responses.add(responses.GET, BALANCE_URL, body="<html>login</html>", status=200)

    resp = client.get("/api/tokens/balance", headers=auth)

    assert resp.status_code == 502
