from __future__ import annotations

from unittest import mock

import bcrypt
from fastapi.testclient import TestClient

from src.api import auth
from src.service_ia.config import secret_store


def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def test_encrypt_roundtrip_and_wrong_key_is_unreadable():
    with mock.patch.dict("os.environ", {"SECRETS_ENCRYPTION_KEY": "chiave-uno"}):
        token = secret_store.encrypt("valore-segreto")
        assert token != "valore-segreto"
        assert secret_store.decrypt(token) == "valore-segreto"
    with mock.patch.dict("os.environ", {"SECRETS_ENCRYPTION_KEY": "chiave-due"}):
        assert secret_store.decrypt(token) is None


def test_mask_hides_all_but_last_four():
    assert secret_store.mask("abcdefghijkl").endswith("ijkl")
    assert "abcdefgh" not in secret_store.mask("abcdefghijkl")


def test_get_secret_seeds_db_from_env_once_then_db_wins():
    db = {}
    secret_store.reset_cache()
    with (
        mock.patch.object(secret_store, "_read_db", side_effect=lambda n: db.get(n)),
        mock.patch.object(secret_store, "_write_db", side_effect=lambda n, v: db.__setitem__(n, v)),
        mock.patch.dict("os.environ", {"API_ODDS_KEY": "seme-da-env-1"}),
    ):
        assert secret_store.get_secret("api_odds_key") == "seme-da-env-1"
        assert db["api_odds_key"] == "seme-da-env-1"
        # L'ambiente cambia, ma conta solo il DB.
        db["api_odds_key"] = "valore-nel-db"
        secret_store.reset_cache()
        with mock.patch.dict("os.environ", {"API_ODDS_KEY": "altro-valore-env"}):
            assert secret_store.get_secret("api_odds_key") == "valore-nel-db"
    secret_store.reset_cache()


def test_provider_keys_and_credentials_reveal_endpoints():
    db = {}
    store = {}
    with (
        mock.patch.object(auth, "JWT_SECRET", "unit-test-secret"),
        mock.patch.object(auth, "COOKIE_SECURE", False),
        mock.patch.object(auth, "load_override", return_value=auth.Credentials("admin", _hash("s3cret"), 1)),
        mock.patch.object(auth, "reveal_password", return_value="s3cret"),
        mock.patch.object(secret_store, "_read_db", side_effect=lambda n: db.get(n)),
        mock.patch.object(secret_store, "_write_db", side_effect=lambda n, v: db.__setitem__(n, v)),
        mock.patch.dict("os.environ", {"API_SPORTS_KEY": "", "API_ODDS_KEY": ""}),
    ):
        auth.reset_credentials_cache()
        secret_store.reset_cache()
        from src.api.main import app

        client = TestClient(app)
        assert client.get("/settings/provider-keys").status_code == 401
        assert client.get("/auth/credentials/reveal").status_code == 401

        assert client.post("/auth/login", json={"username": "admin", "password": "s3cret"}).status_code == 200
        assert client.get("/auth/credentials/reveal").json() == {"username": "admin", "password": "s3cret"}

        listing = client.get("/settings/provider-keys").json()["keys"]
        assert {k["name"]: k["configured"] for k in listing} == {"api_sports_key": False, "api_odds_key": False}

        bad = client.put("/settings/provider-keys/api_sports_key", json={"current_password": "no", "value": "chiave-lunga-123"})
        assert bad.status_code == 403
        assert client.put("/settings/provider-keys/sconosciuta", json={"current_password": "s3cret", "value": "chiave-lunga-123"}).status_code == 404
        ok = client.put("/settings/provider-keys/api_sports_key", json={"current_password": "s3cret", "value": "chiave-lunga-123"})
        assert ok.status_code == 200
        assert client.get("/settings/provider-keys/api_sports_key/reveal").json()["value"] == "chiave-lunga-123"
        masked = [k for k in client.get("/settings/provider-keys").json()["keys"] if k["name"] == "api_sports_key"][0]
        assert masked["configured"] is True and "chiave-lunga" not in masked["masked"]
    auth.reset_credentials_cache()
    secret_store.reset_cache()
