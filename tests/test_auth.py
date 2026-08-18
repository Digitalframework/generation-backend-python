"""Verification of tokens minted by AuthAndSecurity."""

import uuid


def test_no_header_is_401(client):
    resp = client.get("/api/me")
    assert resp.status_code == 401
    assert "Bearer" in resp.headers["WWW-Authenticate"]


def test_non_bearer_scheme_is_401(client):
    assert client.get("/api/me", headers={"Authorization": "Basic abc"}).status_code == 401


def test_claims_become_the_principal(client, mint):
    user_id = str(uuid.uuid4())
    token = mint(subject=user_id, email="sam@example.com", roles=("ADMIN", "USER"))

    resp = client.get("/api/me", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200
    assert resp.get_json() == {
        "id": user_id,
        "email": "sam@example.com",
        "roles": ["ADMIN", "USER"],
    }


def test_expired_token_is_rejected(client, mint):
    token = mint(expires_in=-60)
    resp = client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert "expired" in resp.get_json()["message"].lower()


def test_token_for_another_service_is_rejected(client, mint):
    """The audience check is what stops a token being replayed across services."""
    token = mint(audience="some-other-service")
    resp = client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert "audience" in resp.get_json()["message"].lower()


def test_wrong_issuer_is_rejected(client, mint):
    token = mint(issuer="http://evil.test")
    resp = client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_tampered_token_is_rejected(client, mint):
    token = mint()
    header, payload, signature = token.split(".")
    tampered = f"{header}.{payload}x.{signature}"

    resp = client.get("/api/me", headers={"Authorization": f"Bearer {tampered}"})

    assert resp.status_code == 401


def test_unknown_key_id_is_rejected(client, mint):
    token = mint(kid="not-a-key-we-know")
    resp = client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_role_gate_denies_plain_user(client, mint):
    token = mint(roles=("USER",))
    resp = client.get("/api/admin/health", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


def test_role_gate_admits_admin(client, mint):
    token = mint(roles=("ADMIN",))
    resp = client.get("/api/admin/health", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
