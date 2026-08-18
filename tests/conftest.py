"""Test fixtures that stand in for AuthAndSecurity.

A real RSA key pair is generated per session and published as a JWK set, so the
tokens under test are signed and verified exactly as the running system does it
— the same claims AccessTokenService mints, the same RS256 verification. Only
the JWKS fetch is stubbed, since PyJWKClient reaches for the network with
urllib rather than requests.
"""

import json
import time
import uuid

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from app import create_app
from config import TestConfig

KID = "test-key-1"


@pytest.fixture(scope="session")
def keypair():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private, private.public_key()


@pytest.fixture(scope="session")
def jwks(keypair):
    _, public = keypair
    key = json.loads(RSAAlgorithm.to_jwk(public))
    key.update(kid=KID, use="sig", alg="RS256")
    return {"keys": [key]}


@pytest.fixture(autouse=True)
def stub_jwks(monkeypatch, jwks):
    monkeypatch.setattr(jwt.PyJWKClient, "fetch_data", lambda self: jwks)


@pytest.fixture
def mint(keypair):
    """Mint a token shaped like AccessTokenService.issueFor."""
    private, _ = keypair

    def _mint(
        subject=None,
        email="sam@example.com",
        roles=("USER",),
        audience=TestConfig.JWT_AUDIENCE,
        issuer=TestConfig.JWT_ISSUER,
        expires_in=900,
        kid=KID,
    ):
        now = int(time.time())
        claims = {
            "iss": issuer,
            "aud": audience,
            "sub": subject or str(uuid.uuid4()),
            "iat": now,
            "exp": now + expires_in,
            "email": email,
            "roles": list(roles),
        }
        return jwt.encode(claims, private, algorithm="RS256", headers={"kid": kid})

    return _mint


@pytest.fixture
def app():
    application = create_app(TestConfig())
    application.config.update(TESTING=True)
    return application


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def auth(mint):
    return {"Authorization": f"Bearer {mint()}"}
