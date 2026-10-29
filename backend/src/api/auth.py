"""Login JWT (cookie httpOnly) per le API - un solo account fisso via env var.

Nessuna tabella utenti/registrazione: ``ADMIN_USERNAME``/``ADMIN_PASSWORD_HASH``
sono impostate su Railway (hash bcrypt, mai la password in chiaro). Il
token non e' mai letto da JS lato frontend - viaggia solo nel cookie
httpOnly settato da ``/auth/login`` e allegato automaticamente dal browser.

``SameSite=None`` richiede ``Secure`` (HTTPS): in locale, dove backend e
frontend girano su ``http://localhost`` a porte diverse, si scende a
``SameSite=Lax`` via ``COOKIE_SECURE=false`` - funziona comunque perche'
``localhost:3000``/``localhost:8000`` sono same-site per il browser (cambia
solo la porta), a differenza dei due domini Railway distinti in produzione.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt

JWT_SECRET = os.environ.get("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.environ.get("JWT_EXPIRE_HOURS", "24"))

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "")
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH", "")

COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").lower() == "true"
AUTH_COOKIE_NAME = "access_token"

PUBLIC_PATHS = {"/", "/health", "/health/database", "/auth/login"}


def verify_password(plain_password: str, password_hash: str) -> bool:
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


@dataclass(frozen=True)
class Credentials:
    username: str
    password_hash: str
    version: int = 0  # 0 = variabili d'ambiente, >=1 = modificate dal portale


_CREDENTIALS_TTL_SECONDS = 10.0
_cache: Optional[tuple[float, Credentials]] = None


def _env_credentials() -> Credentials:
    return Credentials(ADMIN_USERNAME, ADMIN_PASSWORD_HASH, 0)


def load_override() -> Optional[Credentials]:
    """Riga `admin_credential` (None se mai modificate dal portale)."""
    from src.repository.base.repository_db import SessionLocal
    from src.service_ia.model.admin_credential import AdminCredential

    with SessionLocal() as session:
        row = session.get(AdminCredential, 1)
        if row is None:
            return None
        return Credentials(row.username, row.password_hash, int(row.version))


def current_credentials() -> Credentials:
    """Credenziali valide ora. Cache breve; se il DB non risponde si tiene
    l'ultimo valore noto (mai tornare alla vecchia password delle env var dopo
    che e' stata cambiata)."""
    global _cache
    now = time.monotonic()
    if _cache is not None and now - _cache[0] < _CREDENTIALS_TTL_SECONDS:
        return _cache[1]
    try:
        creds = load_override() or _env_credentials()
    except Exception:  # noqa: BLE001 - DB irraggiungibile: non bloccare il pannello
        logging.exception("Credenziali admin: lettura dal DB fallita")
        if _cache is not None:
            return _cache[1]
        creds = _env_credentials()
    _cache = (now, creds)
    return creds


def reset_credentials_cache() -> None:
    global _cache
    _cache = None


def save_override(username: str, password_hash: str) -> Credentials:
    """Salva le nuove credenziali (riga unica) e incrementa la versione."""
    from src.repository.base.repository_db import SessionLocal
    from src.service_ia.model.admin_credential import AdminCredential

    with SessionLocal() as session:
        row = session.get(AdminCredential, 1)
        if row is None:
            row = AdminCredential(id=1, username=username, password_hash=password_hash, version=1)
            session.add(row)
        else:
            row.username = username
            row.password_hash = password_hash
            row.version = int(row.version) + 1
        session.commit()
        creds = Credentials(row.username, row.password_hash, int(row.version))
    reset_credentials_cache()
    return creds


def create_access_token(username: str, version: int = 0) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {"sub": username, "exp": expires_at, "cv": version}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
    return payload.get("sub")


def decode_valid_token(token: Optional[str]) -> Optional[str]:
    """Come `decode_token`, ma rifiuta i token emessi prima di un cambio
    credenziali (versione o username diversi da quelli attuali)."""
    if not token:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
    creds = current_credentials()
    if payload.get("cv", 0) != creds.version or payload.get("sub") != creds.username:
        return None
    return payload.get("sub")


def cookie_kwargs() -> dict:
    return {
        "httponly": True,
        "secure": COOKIE_SECURE,
        "samesite": "none" if COOKIE_SECURE else "lax",
        "max_age": JWT_EXPIRE_HOURS * 3600,
    }
