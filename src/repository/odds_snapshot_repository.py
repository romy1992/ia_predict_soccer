from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.repository.base.repository_db import SessionLocal
from src.service_ia.model.match import OddsSnapshot

_UPSERT_COLUMNS = (
    "id_match",
    "fixture_id",
    "bookmaker",
    "market",
    "period",
    "line",
    "outcome",
    "odd",
    "captured_at",
    "source",
)


class OddsSnapshotRepository:
    def save_many(self, snapshots: list[OddsSnapshot]) -> None:
        """Upsert massivo per (id_snapshot).

        BUGFIX 2026-09-06 (performance): PRIMA faceva `session.merge(...)`
        in un loop Python, e ogni `merge()` esegue una query SELECT
        implicita per capire se la PK esiste gia' - con migliaia di
        snapshot per singolo job (es. 7196 osservati in un solo "Aggiorna
        tutto" da 55 fixture, vedi jobs_history.jsonl), questo significava
        migliaia di round-trip SEQUENZIALI verso il DB remoto (Railway), un
        contributo enorme alla lentezza osservata ("ancora in esecuzione
        dopo parecchio tempo"). Un vero bulk `INSERT ... ON CONFLICT
        (id_snapshot) DO UPDATE` fa lo stesso upsert idempotente in POCHE
        query di batch (non piu' una per riga) - stesso comportamento
        esterno (idempotente sulla PK `id_snapshot`, un hash deterministico
        - vedi `map_odds_snapshots`)."""
        if not snapshots:
            return

        # Deduplica per PK mantenendo l'ultimo valore: PostgreSQL rifiuta un
        # batch `ON CONFLICT DO UPDATE` che contenga la STESSA PK piu' volte
        # ("ON CONFLICT DO UPDATE command cannot affect row a second time").
        by_id: dict[str, OddsSnapshot] = {s.id_snapshot: s for s in snapshots}
        rows = [
            {
                "id_snapshot": s.id_snapshot,
                "id_match": s.id_match,
                "fixture_id": s.fixture_id,
                "bookmaker": s.bookmaker,
                "market": s.market,
                "period": s.period,
                "line": s.line,
                "outcome": s.outcome,
                "odd": s.odd,
                "captured_at": s.captured_at,
                "source": s.source,
            }
            for s in by_id.values()
        ]

        session = SessionLocal()
        try:
            # In batch (non un'unica query gigante) per restare entro
            # limiti ragionevoli di query size/parametri con migliaia di
            # righe - stesso ordine di grandezza gia' visto funzionare per
            # gli INSERT di `download_import_matches` (batch impliciti da
            # centinaia di parametri).
            batch_size = 500
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                stmt = pg_insert(OddsSnapshot).values(batch)
                update_cols = {col: getattr(stmt.excluded, col) for col in _UPSERT_COLUMNS}
                stmt = stmt.on_conflict_do_update(index_elements=["id_snapshot"], set_=update_cols)
                session.execute(stmt)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list_for_fixture(
        self,
        fixture_id: int,
        market: Optional[str] = None,
        period: Optional[str] = None,
        line: Optional[str] = None,
    ) -> list[OddsSnapshot]:
        with SessionLocal() as session:
            query = session.query(OddsSnapshot).filter(OddsSnapshot.fixture_id == int(fixture_id))
            if market:
                query = query.filter(OddsSnapshot.market == market)
            if period:
                query = query.filter(OddsSnapshot.period == period)
            if line is not None:
                query = query.filter(OddsSnapshot.line == line)
            return query.order_by(OddsSnapshot.captured_at.asc()).all()

    def list_for_fixture_until(
        self,
        fixture_id: int,
        prediction_at: datetime,
        market: Optional[str] = None,
        period: Optional[str] = None,
        line: Optional[str] = None,
    ) -> list[OddsSnapshot]:
        with SessionLocal() as session:
            query = session.query(OddsSnapshot).filter(OddsSnapshot.fixture_id == int(fixture_id))
            query = query.filter(OddsSnapshot.captured_at <= prediction_at)
            if market:
                query = query.filter(OddsSnapshot.market == market)
            if period:
                query = query.filter(OddsSnapshot.period == period)
            if line is not None:
                query = query.filter(OddsSnapshot.line == line)
            return query.order_by(OddsSnapshot.captured_at.asc()).all()

    def list_all(self) -> list[OddsSnapshot]:
        with SessionLocal() as session:
            return session.query(OddsSnapshot).order_by(OddsSnapshot.captured_at.asc()).all()

    def opening_latest_closing(self, fixture_id: int) -> list[dict]:
        rows = self.list_for_fixture(fixture_id=fixture_id)
        grouped: dict[tuple[str, str, str, str | None, str], list[OddsSnapshot]] = {}

        for row in rows:
            key = (row.bookmaker, row.market, row.period, row.line, row.outcome)
            grouped.setdefault(key, []).append(row)

        payload: list[dict] = []
        for key, values in grouped.items():
            values = sorted(values, key=lambda item: item.captured_at)
            opening = values[0]
            latest = values[-1]
            payload.append(
                {
                    "bookmaker": key[0],
                    "market": key[1],
                    "period": key[2],
                    "line": key[3],
                    "outcome": key[4],
                    "opening": {
                        "odd": opening.odd,
                        "captured_at": opening.captured_at.isoformat(),
                    },
                    "latest": {
                        "odd": latest.odd,
                        "captured_at": latest.captured_at.isoformat(),
                    },
                    # Per un flusso prematch il closing coincide con l'ultimo snapshot disponibile.
                    "closing": {
                        "odd": latest.odd,
                        "captured_at": latest.captured_at.isoformat(),
                    },
                }
            )

        payload.sort(key=lambda row: (row["market"], row["line"] or "", row["bookmaker"], row["outcome"]))
        return payload

