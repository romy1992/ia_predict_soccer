from __future__ import annotations

from typing import Optional

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

    def get_latest_bulk(self, fixture_ids: list[int]) -> dict[tuple[int, str], MatchPredictionSnapshot]:
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
            rows = (
                session.query(MatchPredictionSnapshot)
                .filter(MatchPredictionSnapshot.fixture_id.in_(fixture_ids))
                .order_by(MatchPredictionSnapshot.computed_at.asc())
                .all()
            )

        latest: dict[tuple[int, str], MatchPredictionSnapshot] = {}
        for row in rows:
            latest[(row.fixture_id, row.market)] = row
        return latest

    def list_for_fixture(self, fixture_id: int, market: Optional[str] = None) -> list[MatchPredictionSnapshot]:
        with SessionLocal() as session:
            query = session.query(MatchPredictionSnapshot).filter(
                MatchPredictionSnapshot.fixture_id == int(fixture_id)
            )
            if market:
                query = query.filter(MatchPredictionSnapshot.market == market)
            return query.order_by(MatchPredictionSnapshot.computed_at.asc()).all()
