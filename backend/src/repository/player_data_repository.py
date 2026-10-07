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
        aggiorna le righe esistenti sul posto invece di duplicarle.

        Dedup PRIMA del merge (bug reale, 2026-10-07): API-Sports puo'
        restituire lo STESSO evento due volte nella stessa risposta
        (osservato su una fixture reale, un cartellino a ridosso del 90'+6
        ripetuto identico) - stesso id deterministico per entrambe le copie.
        `session.merge` controlla solo il DB, non gli altri oggetti gia'
        mersi in QUESTA sessione non ancora flushata: due eventi con lo
        stesso id finiscono entrambi marcati per INSERT e il bulk insert
        va in UniqueViolation su `id_event`."""
        if not events:
            return
        deduped = list({event.id_event: event for event in events}.values())
        with SessionLocal() as session:
            for event in deduped:
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
