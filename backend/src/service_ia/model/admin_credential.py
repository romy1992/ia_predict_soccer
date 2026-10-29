"""Credenziali del pannello modificate dal portale (riga unica, id=1).

Se la riga non esiste valgono `ADMIN_USERNAME`/`ADMIN_PASSWORD_HASH` delle
variabili d'ambiente. `version` sale a ogni modifica e finisce nel JWT, cosi'
i token emessi prima del cambio smettono di valere."""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, String, func

from src.service_ia.model.match import Base


class AdminCredential(Base):
    __tablename__ = "admin_credential"

    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    version = Column(Integer, nullable=False, default=1)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
