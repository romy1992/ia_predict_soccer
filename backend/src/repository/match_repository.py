from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import insert as sa_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.repository.base.crud_repository import CrudRepository
from src.service_ia.model.match import Match

logger = logging.getLogger(__name__)


def _giorno_successivo(giorno_iso: str) -> str:
    """"2026-09-14" -> "2026-09-15". Serve a rendere INCLUSIVO l'estremo
    superiore di `search_by_date_window` con un confronto `<` su stringa."""
    return (date.fromisoformat(giorno_iso[:10]) + timedelta(days=1)).isoformat()


class MatchRepository(CrudRepository):  # Connessione base con i metodi crud

    def __init__(self):
        super().__init__(Match)

    def search_by_date_window(
        self,
        from_day: str,
        to_day: str,
        statuses: Optional[list[str]] = None,
        seasons: Optional[list[int]] = None,
        leagues: Optional[list[int]] = None,
        solo_con_id_fixture: bool = False,
    ) -> list[Match]:
        """Match con `date_match` nella finestra [from_day, to_day] INCLUSIVA,
        filtrando in SQL invece che in Python.

        Fix prestazioni 2026-09-15: `SettlementService.run_settlement`
        chiedeva a `search_filter` TUTTI i match con stato finale senza
        alcun vincolo di data - 44.983 righe su tutte le stagioni, ognuna
        con statistics/odds (e, prima del passaggio a `lazy="select"`, tutti
        gli odds_snapshot) - per poi scartarne in Python il 99% con
        `_match_in_window`, dato che la finestra utile e' di soli 4 giorni
        (oggi-3 -> oggi). Misurato: 1.331 s per esecuzione, su un job con
        intervallo di 60 minuti.

        `date_match` e' una colonna VARCHAR con ISO 8601 e offset fisso
        "+00:00" (es. "2026-09-14T16:30:00+00:00"), quindi il confronto
        lessicografico sul prefisso "YYYY-MM-DD" equivale a un confronto
        temporale e usa `ix_match_date_match` - stessa tecnica gia' adottata
        da `DashboardService._fetch_matches`. `to_day` e' reso inclusivo
        confrontando con il giorno DOPO in `<` (una stringa
        "2026-09-14T23:59:59+00:00" e' comunque < "2026-09-15").
        """
        session = self.session
        query = session.query(Match).filter(
            Match.date_match >= from_day,
            Match.date_match < _giorno_successivo(to_day),
        )
        if statuses:
            query = query.filter(Match.status.in_(statuses))
        if seasons:
            query = query.filter(Match.season.in_(seasons))
        if leagues:
            query = query.filter(Match.current_league.in_(leagues))
        if solo_con_id_fixture:
            # Esplicito e non implicito: una ricerca per finestra di date non
            # deve scartare righe da sola. Serve a chi poi usa `id_fixture`
            # come intero (es. `SettlementService`, che ci fa `int(...)`).
            query = query.filter(Match.id_fixture.is_not(None))
        return query.all()

    def upsert_base_by_fixture(self, dict_base: dict) -> tuple[str, bool]:
        """Scrive SUBITO la riga base di una fixture e ne restituisce
        l'`id_match_fk` DEFINITIVO, piu' un flag "e' stata inserita adesso".

        Bug fix 2026-09-15 (righe duplicate). `download_import_matches`
        decideva insert/update con `filter_by(...).first()` e poi accumulava
        le righe NUOVE in una lista Python, scritta con `save_all` solo alla
        FINE del job (che dura minuti). In quella finestra gli insert non
        erano visibili a nessuno: ne' alle iterazioni successive dello stesso
        job, ne' - soprattutto - a un'altra esecuzione in parallelo. Due giri
        sovrapposti sulla stessa finestra di date (misurato 2026-09-15: job
        `data_daily_refresh` nel container `scheduler` e bottone "Aggiorna
        tutto" nel container `api`, nessun lock condiviso) concludevano
        entrambi "questa fixture non esiste" e inserivano, ciascuno con il
        proprio `uuid4()`: 116 partite finite a DB in doppia copia, e gli
        update successivi - che passano da `.first()` - ne aggiornavano una
        sola, lasciando l'altra a `NS` per sempre.

        Con un vero `INSERT ... ON CONFLICT (id_fixture) DO UPDATE ...
        RETURNING id_match_fk` la riga e' persistita immediatamente e, in
        caso di conflitto, il RETURNING riporta l'id di chi ha vinto la
        corsa: le due esecuzioni convergono sulla STESSA riga invece di
        crearne due. Presuppone l'indice unico `uq_match_id_fixture`
        (migration f1a2b3c4d5e6).

        Aggiorna soltanto le colonne presenti in `dict_base` (quelle che
        `map_base_match` ricava dalla risposta del provider). `id_match_fk`
        e' escluso dal SET: e' la PK e ci puntano `statistics`/`odds`/
        `odds_snapshot`. Restano quindi intatte le colonne gestite da altri
        servizi (`is_settled`/`settlement_*` di `SettlementService`,
        `mean_statistics` di `calculate_mean`, `id_events`/`sport_key` degli
        import storici da odds-api).

        :return: (id_match_fk definitivo, True se inserita ORA da noi)
        """
        colonne = {k: v for k, v in dict_base.items() if k in Match.__table__.columns.keys()}
        id_fixture = colonne.get("id_fixture")
        id_proposto = colonne.get("id_match_fk")

        if id_fixture is None:
            # Nessun `id_fixture` su cui fare inferenza del conflitto (righe
            # legacy da odds-api). Si ricade sul comportamento storico: qui
            # non c'e' corsa da proteggere, perche' e' `id_fixture` la chiave
            # naturale che due import concorrenti si contendono.
            self.save(Match(**dict_base))
            return id_proposto, True

        session = self.session
        try:
            if session.get_bind().dialect.name == "postgresql":
                stmt = pg_insert(Match).values(**colonne)
                set_colonne = {k: getattr(stmt.excluded, k) for k in colonne if k != "id_match_fk"}
                stmt = stmt.on_conflict_do_update(
                    index_elements=["id_fixture"],
                    set_=set_colonne,
                ).returning(Match.id_match_fk)
                id_definitivo = session.execute(stmt).scalar_one()
            else:
                # Percorso portabile (SQLite dei test): stesso effetto in due
                # istruzioni. Non e' atomico come `ON CONFLICT`, ma i test
                # girano su una connessione sola e non hanno corse da gestire.
                esistente = (
                    session.query(Match.id_match_fk)
                    .filter(Match.id_fixture == id_fixture)
                    .first()
                )
                if esistente:
                    id_definitivo = esistente[0]
                    session.query(Match).filter(Match.id_fixture == id_fixture).update(
                        {k: v for k, v in colonne.items() if k != "id_match_fk"},
                        synchronize_session=False,
                    )
                else:
                    session.execute(sa_insert(Match).values(**colonne))
                    id_definitivo = id_proposto
            session.commit()
        except Exception as exc:
            logger.error("upsert_base_by_fixture id_fixture=%s: %s", id_fixture, exc)
            session.rollback()
            raise

        # Inserita da noi solo se la riga non c'era E il DB ci ha restituito
        # proprio l'id che avevamo proposto: se ne torna un altro, un'altra
        # esecuzione ha vinto la corsa e la nostra e' di fatto una update.
        return id_definitivo, id_definitivo == id_proposto
