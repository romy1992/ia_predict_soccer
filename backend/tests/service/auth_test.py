from __future__ import annotations

import time
from unittest import mock

import bcrypt
import jwt
from fastapi.testclient import TestClient

from src.api import auth


def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def test_verify_password_correct_and_wrong():
    password_hash = _hash("correct-password")
    assert auth.verify_password("correct-password", password_hash) is True
    assert auth.verify_password("wrong-password", password_hash) is False


def test_verify_password_empty_hash_rejects():
    assert auth.verify_password("anything", "") is False


def test_create_and_decode_access_token_roundtrip():
    with mock.patch.object(auth, "JWT_SECRET", "unit-test-secret"):
        token = auth.create_access_token("admin")
        assert auth.decode_token(token) == "admin"


def test_decode_token_rejects_tampered_token():
    with mock.patch.object(auth, "JWT_SECRET", "unit-test-secret"):
        token = auth.create_access_token("admin")
    assert auth.decode_token(token + "tampered") is None


def test_decode_token_rejects_expired_token():
    with mock.patch.object(auth, "JWT_SECRET", "unit-test-secret"):
        expired_payload = {"sub": "admin", "exp": int(time.time()) - 10}
        expired_token = jwt.encode(expired_payload, auth.JWT_SECRET, algorithm=auth.JWT_ALGORITHM)
        assert auth.decode_token(expired_token) is None


def test_decode_token_none_or_garbage_returns_none():
    assert auth.decode_token("not-a-jwt") is None


def test_cookie_kwargs_secure_uses_samesite_none():
    with mock.patch.object(auth, "COOKIE_SECURE", True):
        kwargs = auth.cookie_kwargs()
    assert kwargs["secure"] is True
    assert kwargs["samesite"] == "none"
    assert kwargs["httponly"] is True


def test_cookie_kwargs_insecure_falls_back_to_samesite_lax():
    with mock.patch.object(auth, "COOKIE_SECURE", False):
        kwargs = auth.cookie_kwargs()
    assert kwargs["secure"] is False
    assert kwargs["samesite"] == "lax"


def test_login_logout_and_protected_route_end_to_end():
    with (
        mock.patch.object(auth, "JWT_SECRET", "unit-test-secret"),
        mock.patch.object(auth, "ADMIN_USERNAME", "admin"),
        mock.patch.object(auth, "ADMIN_PASSWORD_HASH", _hash("s3cret")),
        mock.patch.object(auth, "COOKIE_SECURE", False),
    ):
        from src.api.main import app

        client = TestClient(app)

        # Pubblico senza cookie.
        assert client.get("/health").status_code == 200

        # Una route protetta qualsiasi, senza cookie, deve essere negata.
        assert client.get("/jobs/history").status_code == 401

        # Login con credenziali sbagliate: 401, nessun cookie settato.
        wrong = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
        assert wrong.status_code == 401
        assert auth.AUTH_COOKIE_NAME not in wrong.cookies

        # Login corretto: 200 + cookie settato.
        ok = client.post("/auth/login", json={"username": "admin", "password": "s3cret"})
        assert ok.status_code == 200
        assert ok.json() == {"username": "admin"}
        assert auth.AUTH_COOKIE_NAME in client.cookies

        # Con il cookie, la stessa route protetta ora risponde.
        assert client.get("/auth/me").status_code == 200
        assert client.get("/auth/me").json() == {"username": "admin"}

        # Logout cancella il cookie: la route torna negata.
        logout_response = client.post("/auth/logout")
        assert logout_response.status_code == 200
        assert client.get("/auth/me").status_code == 401
