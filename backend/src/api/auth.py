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

import os
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


def create_access_token(username: str) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {"sub": username, "exp": expires_at}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
    return payload.get("sub")


def cookie_kwargs() -> dict:
    return {
        "httponly": True,
        "secure": COOKIE_SECURE,
        "samesite": "none" if COOKIE_SECURE else "lax",
        "max_age": JWT_EXPIRE_HOURS * 3600,
    }
