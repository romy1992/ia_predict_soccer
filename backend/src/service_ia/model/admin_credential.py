"""Credenziali del pannello modificate dal portale (riga unica, id=1).

Se la riga non esiste valgono `ADMIN_USERNAME`/`ADMIN_PASSWORD_HASH` delle
variabili d'ambiente. `version` sale a ogni modifica e finisce nel JWT, cosi'
i token emessi prima del cambio smettono di valere."""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, String, Text, func

from src.service_ia.model.match import Base


class AdminCredential(Base):
    __tablename__ = "admin_credential"

    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    version = Column(Integer, nullable=False, default=1)
    # Copia cifrata della password in chiaro (per l'"occhio"); NULL finche' la
    # password non viene reimpostata dal portale (l'ambiente ha solo l'hash).
    password_enc = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class AppSecret(Base):
    """Chiavi dei provider, cifrate (vedi `service_ia/config/secret_store.py`)."""

    __tablename__ = "app_secret"

    name = Column(String, primary_key=True)
    value_encrypted = Column(Text, nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
