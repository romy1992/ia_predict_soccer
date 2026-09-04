"""Modelli ORM del dataset LIVE (LIVE-01, Fase LIVE ORACLE).

Acceptance criteria "Dataset live distinto": queste tabelle sono
COMPLETAMENTE separate da `match`/`statistics`/`odds` (il dataset usato dal
training pre-match, popolato da `download_match_service.py`/DATA-03) - un
polling live non scrive MAI in quelle tabelle, ed un retrain non legge MAI
da queste ("Non mischiare training pre-match e live", `tasks/LIVE-01.md`).

Riusano pero' la STESSA `Base`/metadata di `src.service_ia.model.match`
(stesso principio gia' visto per `OddsSnapshot`/`PredictionLedger`, che
vivono anch'esse in un'unica `Base` condivisa) cosi' che Alembic
(`alembic/env.py::target_metadata = Base.metadata`) le veda senza bisogno
di una seconda configurazione/metadata separata.

Ogni riga e' un OSSERVAZIONE puntuale con `captured_at` esplicito (acceptance
criteria "Timestamp eventi"): per le fixture/statistiche si conserva uno
storico completo (uno snapshot per poll, stesso pattern di `OddsSnapshot`),
per gli eventi si usa un id DETERMINISTICO cosi' che poll ripetuti sullo
stesso evento (l'API ritorna sempre l'elenco COMPLETO degli eventi fin li',
non solo i nuovi) restino idempotenti (`session.merge`, mai un duplicato).
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, JSON, String

from src.service_ia.model.match import Base

# Stessa Base di src.service_ia.model.match: le tabelle LIVE condividono il
# metadata (per Alembic autogenerate/target_metadata) ma restano dataset
# a se' stanti - nessuna relationship verso `Match` (nessun accoppiamento
# col dataset pre-match, per costruzione).


def live_event_id(
    fixture_id: int,
    event_type: str,
    detail: str,
    elapsed: int | None,
    elapsed_extra: int | None,
    team_id: int | None,
    player_name: str | None,
) -> str:
    """ID deterministico per un evento live (stesso principio di
    `pick_pool._pool_id`/`betslip_builder.slip_id`: hash stabile dai campi
    che identificano univocamente l'evento, NON un uuid random) - garantisce
    che lo stesso evento osservato in poll successivi produca sempre lo
    stesso id, rendendo `session.merge` un vero upsert idempotente invece
    di inserire un duplicato ad ogni ciclo di polling."""
    raw = "|".join(
        [
            str(fixture_id),
            str(event_type or ""),
            str(detail or ""),
            str(elapsed if elapsed is not None else ""),
            str(elapsed_extra if elapsed_extra is not None else ""),
            str(team_id if team_id is not None else ""),
            str(player_name or ""),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class LiveFixtureSnapshot(Base):
    """Uno snapshot dello stato/punteggio corrente di UNA fixture live,
    catturato ad ogni ciclo di polling (`LiveDataService.sync_live_data`).
    Storico completo (mai un solo "ultimo stato" sovrascritto) per poter
    ricostruire l'evoluzione minuto per minuto (base per LIVE-02/LIVE-03)."""

    __tablename__ = "live_fixture_snapshot"

    id_snapshot = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    fixture_id = Column(Integer, nullable=False, index=True)
    captured_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    league_id = Column(Integer, nullable=True)
    status = Column(String, nullable=True)
    elapsed_minute = Column(Integer, nullable=True)
    elapsed_extra = Column(Integer, nullable=True)
    home_team_id = Column(Integer, nullable=True)
    away_team_id = Column(Integer, nullable=True)
    home_goals = Column(Integer, nullable=True)
    away_goals = Column(Integer, nullable=True)
    raw_payload = Column(JSON(none_as_null=True), nullable=True)
    source = Column(String, nullable=False, default="api_sports")

    def to_dict(self) -> dict:
        payload = {column.name: getattr(self, column.name) for column in self.__table__.columns}
        if payload.get("captured_at") is not None:
            payload["captured_at"] = payload["captured_at"].isoformat()
        return payload


class LiveMatchEvent(Base):
    """Un evento live (gol, cartellino, sostituzione, VAR...) con timestamp
    esplicito - id deterministico (`live_event_id`) per idempotenza."""

    __tablename__ = "live_match_event"

    id_event = Column(String(40), primary_key=True)
    fixture_id = Column(Integer, nullable=False, index=True)
    captured_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    elapsed_minute = Column(Integer, nullable=True)
    elapsed_extra = Column(Integer, nullable=True)
    event_type = Column(String, nullable=True)
    event_detail = Column(String, nullable=True)
    team_id = Column(Integer, nullable=True)
    team_name = Column(String, nullable=True)
    player_name = Column(String, nullable=True)
    assist_name = Column(String, nullable=True)
    comments = Column(String, nullable=True)
    source = Column(String, nullable=False, default="api_sports")

    def to_dict(self) -> dict:
        payload = {column.name: getattr(self, column.name) for column in self.__table__.columns}
        if payload.get("captured_at") is not None:
            payload["captured_at"] = payload["captured_at"].isoformat()
        return payload


class LiveFixtureStatSnapshot(Base):
    """Uno snapshot delle statistiche correnti (possesso, tiri, corner...)
    per UNA squadra di UNA fixture live, catturato ad ogni poll - stesso
    schema grezzo di `fixtures/statistics` (API-Sports), non normalizzato:
    la normalizzazione/feature engineering e' scope di LIVE-02."""

    __tablename__ = "live_fixture_stat_snapshot"

    id_snapshot = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    fixture_id = Column(Integer, nullable=False, index=True)
    team_id = Column(Integer, nullable=True)
    captured_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    stats = Column(JSON(none_as_null=True), nullable=True)
    source = Column(String, nullable=False, default="api_sports")

    def to_dict(self) -> dict:
        payload = {column.name: getattr(self, column.name) for column in self.__table__.columns}
        if payload.get("captured_at") is not None:
            payload["captured_at"] = payload["captured_at"].isoformat()
        return payload


