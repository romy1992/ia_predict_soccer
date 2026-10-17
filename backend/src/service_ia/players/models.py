"""Anagrafica dei giocatori indipendente dal calcio dei modelli: `player_identity` e `player_provider_link`.

Una persona = una `player_identity`; ogni provider esterno (API-Sports e' il contenitore principale, FotMob il
secondo) ha un collegamento `player_provider_link` con il proprio ID e il nome come lo scrive quel provider.
Gli ID usati da modelli e storico restano quelli API-Sports."""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, Integer, String, UniqueConstraint, func

from src.service_ia.model.match import Base


class PlayerIdentity(Base):
    __tablename__ = "player_identity"

    id = Column(Integer, primary_key=True, autoincrement=True)
    canonical_name = Column(String, nullable=False)
    birth_date = Column(String, nullable=True)  # ISO YYYY-MM-DD, quando un provider (o un progetto esterno) la fornisce
    position = Column(String, nullable=True)  # G / D / M / F
    team_api_id = Column(Integer, nullable=True, index=True)  # ultima squadra nota (ID API-Sports)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class PlayerProviderLink(Base):
    __tablename__ = "player_provider_link"
    __table_args__ = (UniqueConstraint("provider", "provider_player_id", name="uq_player_provider_link"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    identity_id = Column(Integer, nullable=False, index=True)
    provider = Column(String, nullable=False, index=True)  # "api_sports" | "fotmob"
    provider_player_id = Column(String, nullable=False)
    provider_name = Column(String, nullable=True)
    name_norm = Column(String, nullable=True, index=True)  # nome normalizzato, per le ricerche
    provider_team = Column(String, nullable=True)
    method = Column(String, nullable=False)  # provider_id | exact_name | name_tokens | last_name_initial | surname_subset |
    #                                           official_elimination | external_project | manual
    confidence = Column(Float, nullable=False, default=1.0)
    source = Column(String, nullable=False)  # lineup_history | fotmob_sync | official_confirm | external_project | manual
    first_seen = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_seen = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
