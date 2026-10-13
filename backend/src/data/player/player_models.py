"""Modello ORM dello storico eventi-per-giocatore (cantiere "giocatori che
segnano", v1 = probabilita' dichiarata, NO motore EV/PLAY-BORDERLINE-NO BET).

Tabella PERMANENTE (diversa da `live_match_event`, che e' uno storico di
POLL LIVE - vedi `src/data/live/live_models.py`): qui una riga e' un evento
realmente accaduto in una fixture GIA' CONCLUSA, scritta una volta (backfill
storico o aggancio nel job di import regolare) e mai piu' re-interrogata via
poll. Riusa la STESSA `Base`/metadata di `src.service_ia.model.match` (stesso
principio di `OddsSnapshot`/`LiveMatchEvent`) cosi' Alembic la vede senza una
seconda configurazione.

A differenza di `LiveMatchEvent`, qui si salva anche `player_id`/`assist_id`
(l'id stabile di API-Sports): verificato con chiamate reali che e' sempre
valorizzato sui gol, quindi il training puo' agganciare un giocatore per id
senza fare matching sul nome (instabile tra fonti/grafie diverse).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String

from src.service_ia.model.match import Base


def player_event_id(
    fixture_id: int,
    event_type: str,
    detail: str,
    elapsed: int | None,
    elapsed_extra: int | None,
    team_id: int | None,
    player_id: int | None,
    player_name: str | None,
) -> str:
    """ID deterministico (stesso principio di `live_event_id`): la stessa
    fixture ri-processata (es. finestra rolling del job di import regolare)
    produce sempre lo stesso id per lo stesso evento, rendendo `session.merge`
    un upsert idempotente invece di duplicare la riga."""
    raw = "|".join(
        [
            str(fixture_id),
            str(event_type or ""),
            str(detail or ""),
            str(elapsed if elapsed is not None else ""),
            str(elapsed_extra if elapsed_extra is not None else ""),
            str(team_id if team_id is not None else ""),
            str(player_id if player_id is not None else ""),
            str(player_name or ""),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class PlayerMatchEvent(Base):
    """Un evento (gol, cartellino, sostituzione...) di una fixture conclusa -
    id deterministico (`player_event_id`) per idempotenza su ri-processing."""

    __tablename__ = "player_match_event"

    id_event = Column(String(40), primary_key=True)
    fixture_id = Column(Integer, nullable=False, index=True)
    recorded_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    elapsed_minute = Column(Integer, nullable=True)
    elapsed_extra = Column(Integer, nullable=True)
    event_type = Column(String, nullable=True)
    event_detail = Column(String, nullable=True)
    team_id = Column(Integer, nullable=True)
    team_name = Column(String, nullable=True)
    player_id = Column(Integer, nullable=True, index=True)
    player_name = Column(String, nullable=True)
    assist_id = Column(Integer, nullable=True)
    assist_name = Column(String, nullable=True)
    comments = Column(String, nullable=True)
    source = Column(String, nullable=False, default="api_sports")

    def to_dict(self) -> dict:
        payload = {column.name: getattr(self, column.name) for column in self.__table__.columns}
        if payload.get("recorded_at") is not None:
            payload["recorded_at"] = payload["recorded_at"].isoformat()
        return payload
