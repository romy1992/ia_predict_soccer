from __future__ import annotations

from src.data.player.player_models import PlayerMatchEvent
from src.repository.base.repository_db import SessionLocal


class PlayerDataRepository:
    """Accesso allo storico eventi-per-giocatore (`player_match_event`) -
    pattern dedicato, stesso approccio di `LiveDataRepository`/
    `OddsSnapshotRepository` (query non coperte dal `CrudRepository`
    generico, qui un semplice check di presenza per fixture)."""

    def save_events(self, events: list[PlayerMatchEvent]) -> None:
        """`session.merge` per id (deterministico, `player_event_id`): una
        fixture ri-processata nella finestra rolling del job regolare
        aggiorna le righe esistenti sul posto invece di duplicarle."""
        if not events:
            return
        with SessionLocal() as session:
            for event in events:
                session.merge(event)
            session.commit()

    def has_events_for_fixture(self, fixture_id: int) -> bool:
        """Usata per decidere se vale la pena spendere una chiamata
        `fixtures/events` su una fixture conclusa: se la fixture e' gia'
        stata catturata (backfill o passaggio precedente del job regolare)
        non la ri-richiediamo ad ogni giro della finestra rolling."""
        with SessionLocal() as session:
            return (
                session.query(PlayerMatchEvent.id_event)
                .filter(PlayerMatchEvent.fixture_id == int(fixture_id))
                .first()
                is not None
            )

    def list_events_for_fixture(self, fixture_id: int) -> list[PlayerMatchEvent]:
        with SessionLocal() as session:
            return (
                session.query(PlayerMatchEvent)
                .filter(PlayerMatchEvent.fixture_id == int(fixture_id))
                .order_by(PlayerMatchEvent.elapsed_minute.asc().nullslast())
                .all()
            )
