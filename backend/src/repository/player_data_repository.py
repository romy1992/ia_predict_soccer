from __future__ import annotations

from src.data.player.player_lineup_models import PlayerLineup
from src.data.player.player_models import PlayerMatchEvent
from src.repository.base.repository_db import SessionLocal


class PlayerDataRepository:
    """Accesso ai dati storici per giocatore - eventi (`player_match_event`)
    e formazioni (`player_lineup`) - pattern dedicato, stesso approccio di
    `LiveDataRepository`/`OddsSnapshotRepository` (query non coperte dal
    `CrudRepository` generico, qui un semplice check di presenza per
    fixture)."""

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

    def captured_fixture_ids(self, fixture_ids: list[int]) -> set[int]:
        """Sottoinsieme di `fixture_ids` gia' catturato - UNA query, non una
        per fixture (a differenza di `has_events_for_fixture`, pensato per
        il singolo controllo nel job regolare). Il backfill storico deve
        scartare migliaia di fixture gia' processate nei giri precedenti:
        farlo con `has_events_for_fixture` in loop costava un round-trip di
        rete per ciascuna, anche solo per scartarla - con l'andare avanti
        del backfill, la sola fase di skip arrivava a richiedere minuti
        prima di toccare anche una sola fixture nuova."""
        if not fixture_ids:
            return set()
        with SessionLocal() as session:
            rows = (
                session.query(PlayerMatchEvent.fixture_id)
                .filter(PlayerMatchEvent.fixture_id.in_(fixture_ids))
                .distinct()
                .all()
            )
        return {row[0] for row in rows}

    def list_events_for_fixture(self, fixture_id: int) -> list[PlayerMatchEvent]:
        with SessionLocal() as session:
            return (
                session.query(PlayerMatchEvent)
                .filter(PlayerMatchEvent.fixture_id == int(fixture_id))
                .order_by(PlayerMatchEvent.elapsed_minute.asc().nullslast())
                .all()
            )

    # -- Formazioni (PlayerLineup) -------------------------------------
    # Stesso repository degli eventi (dominio "dati giocatore" unico),
    # stesso principio di `LiveDataRepository` che gestisce piu' tabelle
    # LIVE in una sola classe.

    def save_lineups(self, lineups: list[PlayerLineup]) -> None:
        """`session.merge` su PK composita (fixture_id, player_id): una
        fixture ri-processata aggiorna la riga sul posto, mai duplicata.

        Dedup PRIMA del merge - stesso bug/fix gia' visto su `save_events`
        (2026-10-07). Causa reale osservata in produzione (2026-10-08,
        fixture 1601534): NON lo stesso giocatore ripetuto, ma un errore
        di qualita' dati lato API-Sports - due giocatori DIVERSI ("M.
        Curado" in startXI, "M. A. Chakir" in substitutes) con lo STESSO
        player_id. La PK composita (fixture_id, player_id) non puo'
        rappresentare entrambi: qui si tiene l'ultimo (comportamento gia'
        accettato per i duplicati veri degli eventi). `session.merge`
        controlla solo il DB, non gli altri oggetti gia' mersi in QUESTA
        sessione non ancora flushata - due righe con la stessa PK finiscono
        entrambe marcate per INSERT e il bulk insert va in UniqueViolation
        su `player_lineup_pkey`."""
        if not lineups:
            return
        deduped = list({(lineup.fixture_id, lineup.player_id): lineup for lineup in lineups}.values())
        with SessionLocal() as session:
            for lineup in deduped:
                session.merge(lineup)
            session.commit()

    def has_lineup_for_fixture(self, fixture_id: int) -> bool:
        with SessionLocal() as session:
            return (
                session.query(PlayerLineup.player_id)
                .filter(PlayerLineup.fixture_id == int(fixture_id))
                .first()
                is not None
            )

    def captured_lineup_fixture_ids(self, fixture_ids: list[int]) -> set[int]:
        """Stesso motivo/pattern di `captured_fixture_ids`: UNA query per
        scartare in blocco le fixture gia' catturate nel backfill, non una
        per fixture."""
        if not fixture_ids:
            return set()
        with SessionLocal() as session:
            rows = (
                session.query(PlayerLineup.fixture_id)
                .filter(PlayerLineup.fixture_id.in_(fixture_ids))
                .distinct()
                .all()
            )
        return {row[0] for row in rows}

    def list_lineup_for_fixture(self, fixture_id: int) -> list[PlayerLineup]:
        with SessionLocal() as session:
            return (
                session.query(PlayerLineup)
                .filter(PlayerLineup.fixture_id == int(fixture_id))
                .all()
            )
