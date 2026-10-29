"""Chiavi dei provider (API-Sports, The Odds API) e copia cifrata della password
del pannello, conservate nel DB (tabella `app_secret`).

Dopo il primo avvio il DB e' l'UNICA fonte: le variabili d'ambiente servono solo
come seme, copiate nel DB la prima volta che un valore manca. I valori sono cifrati
(Fernet) con `SECRETS_ENCRYPTION_KEY` (in sua assenza, derivata da `JWT_SECRET`);
cambiando quella chiave i valori gia' salvati non si leggono piu' e vanno reinseriti.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import time
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

# nome -> (etichetta mostrata nelle Impostazioni, variabile d'ambiente usata come seme)
PROVIDER_SECRETS: dict[str, tuple[str, str]] = {
    "api_sports_key": ("Chiave API-Sports", "API_SPORTS_KEY"),
    "api_odds_key": ("Chiave The Odds API", "API_ODDS_KEY"),
}

_TTL_SECONDS = 30.0
_cache: dict[str, tuple[float, str]] = {}


def _fernet() -> Fernet:
    secret = os.environ.get("SECRETS_ENCRYPTION_KEY") or os.environ.get("JWT_SECRET") or ""
    if not secret:
        raise RuntimeError("SECRETS_ENCRYPTION_KEY (o JWT_SECRET) mancante: impossibile cifrare")
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest()))


def encrypt(value: str) -> str:
    return _fernet().encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    try:
        return _fernet().decrypt(token.encode("utf-8")).decode("utf-8")
    except (InvalidToken, RuntimeError):
        logging.warning("Segreto non decifrabile: chiave di cifratura cambiata? Va reinserito.")
        return None


def mask(value: str) -> str:
    return "•" * 8 + value[-4:] if len(value) > 8 else "•" * len(value)


def _read_db(name: str) -> Optional[str]:
    from src.repository.base.repository_db import SessionLocal
    from src.service_ia.model.admin_credential import AppSecret

    with SessionLocal() as session:
        row = session.get(AppSecret, name)
        return decrypt(row.value_encrypted) if row is not None else None


def _write_db(name: str, value: str) -> None:
    from src.repository.base.repository_db import SessionLocal
    from src.service_ia.model.admin_credential import AppSecret

    with SessionLocal() as session:
        row = session.get(AppSecret, name)
        if row is None:
            session.add(AppSecret(name=name, value_encrypted=encrypt(value)))
        else:
            row.value_encrypted = encrypt(value)
        session.commit()


def get_secret(name: str) -> str:
    """Valore corrente ("" se non configurato). Cache breve; se il DB non risponde
    si tiene l'ultimo valore noto."""
    if name not in PROVIDER_SECRETS:
        raise KeyError(name)
    now = time.monotonic()
    hit = _cache.get(name)
    if hit is not None and now - hit[0] < _TTL_SECONDS:
        return hit[1]
    try:
        value = _read_db(name)
        if value is None:
            # Prima volta: semina dal valore dell'ambiente, poi vale solo il DB.
            seed = os.environ.get(PROVIDER_SECRETS[name][1]) or ""
            if seed:
                _write_db(name, seed)
            value = seed
    except Exception:  # noqa: BLE001
        logging.exception("Segreto %s: lettura dal DB fallita", name)
        return hit[1] if hit is not None else ""
    _cache[name] = (now, value)
    return value


def set_secret(name: str, value: str) -> None:
    if name not in PROVIDER_SECRETS:
        raise KeyError(name)
    _write_db(name, value)
    _cache.pop(name, None)


def reset_cache() -> None:
    _cache.clear()
