from __future__ import annotations

from typing import Optional

from sqlalchemy import func

from src.repository.base.repository_db import SessionLocal
from src.service_ia.model.match import MatchPredictionSnapshot


class MatchPredictionSnapshotRepository:
    """Persistenza del log append-only delle predizioni ML (2026-09-09).

    Stesso pattern "semplice" gia' in uso in `OddsSnapshotRepository`/
    `PredictionLedgerRepository`: una sessione per operazione, nessuno
    stato condiviso tra chiamate."""

    def save(self, snapshot: MatchPredictionSnapshot) -> MatchPredictionSnapshot:
        with SessionLocal() as session:
            session.add(snapshot)
            session.commit()
            session.refresh(snapshot)
            return snapshot

    def get_latest(self, fixture_id: int, market: str) -> Optional[MatchPredictionSnapshot]:
        with SessionLocal() as session:
            return (
                session.query(MatchPredictionSnapshot)
                .filter(MatchPredictionSnapshot.fixture_id == int(fixture_id))
                .filter(MatchPredictionSnapshot.market == market)
                .order_by(MatchPredictionSnapshot.computed_at.desc())
                .first()
            )

    def get_latest_bulk(
        self,
        fixture_ids: list[int],
        markets: Optional[list[str]] = None,
    ) -> dict[tuple[int, str], MatchPredictionSnapshot]:
        """Ultima riga per OGNI (fixture_id, market) presente tra
        `fixture_ids`, in UNA sola query (invece di `get_latest` in loop) -
        usata dal job schedulato di refresh, che considera decine/centinaia
        di fixture in un colpo solo. Le righe sono ordinate per
        `computed_at` crescente cosi' l'ULTIMA occorrenza di ciascuna chiave
        nel dict finale e' sempre la piu' recente (nessun ORDER BY/PARTITION
        BY specifico del dialetto necessario)."""
        if not fixture_ids:
            return {}

        with SessionLocal() as session:
            ranked_query = session.query(
                MatchPredictionSnapshot.id_snapshot.label("id_snapshot"),
                func.row_number()
                .over(
                    partition_by=(
                        MatchPredictionSnapshot.fixture_id,
                        MatchPredictionSnapshot.market,
                    ),
                    order_by=MatchPredictionSnapshot.computed_at.desc(),
                )
                .label("row_number"),
            ).filter(MatchPredictionSnapshot.fixture_id.in_([int(value) for value in fixture_ids]))
            if markets:
                ranked_query = ranked_query.filter(MatchPredictionSnapshot.market.in_(markets))
            ranked = ranked_query.subquery()
            rows = (
                session.query(MatchPredictionSnapshot)
                .join(ranked, MatchPredictionSnapshot.id_snapshot == ranked.c.id_snapshot)
                .filter(ranked.c.row_number == 1)
                .all()
            )

        return {(row.fixture_id, row.market): row for row in rows}

    def list_for_fixture(self, fixture_id: int, market: Optional[str] = None) -> list[MatchPredictionSnapshot]:
        with SessionLocal() as session:
            query = session.query(MatchPredictionSnapshot).filter(
                MatchPredictionSnapshot.fixture_id == int(fixture_id)
            )
            if market:
                query = query.filter(MatchPredictionSnapshot.market == market)
            return query.order_by(MatchPredictionSnapshot.computed_at.asc()).all()
