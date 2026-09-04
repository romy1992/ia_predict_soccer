from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.data.live.live_models import LiveFixtureSnapshot, LiveFixtureStatSnapshot, LiveMatchEvent
from src.repository.base.repository_db import SessionLocal

# Stessa semantica di FINAL_STATUSES in `src.api.dashboard_service` (non
# importata direttamente per non accoppiare il dataset live al modulo
# dashboard - duplicazione deliberata di una costante, non di logica).
_FINAL_STATUSES = {"FT", "AET", "PEN", "ABD", "CANC", "PST", "WO"}


class LiveDataRepository:
    """Accesso al dataset LIVE (LIVE-01) - pattern dedicato (non
    `CrudRepository` generico) perche' le query servono raggruppamenti per
    fixture/ultimo-stato non coperti da `search_filter`, stesso approccio
    gia' scelto per `OddsSnapshotRepository`."""

    def save_fixture_snapshots(self, snapshots: list[LiveFixtureSnapshot]) -> None:
        if not snapshots:
            return
        with SessionLocal() as session:
            for snapshot in snapshots:
                session.merge(snapshot)
            session.commit()

    def save_events(self, events: list[LiveMatchEvent]) -> None:
        """`session.merge` per id (deterministico, `live_event_id`): un
        evento gia' visto in un poll precedente viene aggiornato sul posto
        invece di duplicarsi ("Persistenza idempotente")."""
        if not events:
            return
        with SessionLocal() as session:
            for event in events:
                session.merge(event)
            session.commit()

    def save_stat_snapshots(self, snapshots: list[LiveFixtureStatSnapshot]) -> None:
        if not snapshots:
            return
        with SessionLocal() as session:
            for snapshot in snapshots:
                session.merge(snapshot)
            session.commit()

    def list_fixture_snapshots(self, fixture_id: int) -> list[LiveFixtureSnapshot]:
        with SessionLocal() as session:
            return (
                session.query(LiveFixtureSnapshot)
                .filter(LiveFixtureSnapshot.fixture_id == int(fixture_id))
                .order_by(LiveFixtureSnapshot.captured_at.asc())
                .all()
            )

    def latest_fixture_snapshots(self) -> dict[int, LiveFixtureSnapshot]:
        """Ultimo snapshot per fixture (in memoria: il volume atteso per il
        solo sottoinsieme "live" e' piccolo, stesso ordine di grandezza di
        `_fetch_api_live_fixtures`/`DashboardService`, non l'intero storico)."""
        with SessionLocal() as session:
            rows = session.query(LiveFixtureSnapshot).order_by(LiveFixtureSnapshot.captured_at.asc()).all()
        latest: dict[int, LiveFixtureSnapshot] = {}
        for row in rows:
            latest[int(row.fixture_id)] = row
        return latest

    def list_active_fixture_ids(self) -> list[int]:
        """Fixture il cui ULTIMO snapshot noto non e' in uno stato finale -
        base per decidere quali fixture ripollare (eventi/statistiche) senza
        rifare da zero la lista "live" via provider ad ogni singola chiamata."""
        latest = self.latest_fixture_snapshots()
        return [
            fixture_id
            for fixture_id, snapshot in latest.items()
            if (snapshot.status or "").upper() not in _FINAL_STATUSES
        ]

    def list_events_for_fixture(self, fixture_id: int) -> list[LiveMatchEvent]:
        with SessionLocal() as session:
            return (
                session.query(LiveMatchEvent)
                .filter(LiveMatchEvent.fixture_id == int(fixture_id))
                .order_by(LiveMatchEvent.elapsed_minute.asc().nullslast(), LiveMatchEvent.captured_at.asc())
                .all()
            )

    def list_stat_snapshots_for_fixture(
        self, fixture_id: int, team_id: Optional[int] = None
    ) -> list[LiveFixtureStatSnapshot]:
        with SessionLocal() as session:
            query = session.query(LiveFixtureStatSnapshot).filter(
                LiveFixtureStatSnapshot.fixture_id == int(fixture_id)
            )
            if team_id is not None:
                query = query.filter(LiveFixtureStatSnapshot.team_id == int(team_id))
            return query.order_by(LiveFixtureStatSnapshot.captured_at.asc()).all()

    def latest_stat_snapshots_for_fixture(self, fixture_id: int) -> list[LiveFixtureStatSnapshot]:
        """Ultimo snapshot statistiche PER SQUADRA (2 righe attese: home/away)."""
        rows = self.list_stat_snapshots_for_fixture(fixture_id=fixture_id)
        latest_by_team: dict[Optional[int], LiveFixtureStatSnapshot] = {}
        for row in rows:
            latest_by_team[row.team_id] = row
        return list(latest_by_team.values())

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

