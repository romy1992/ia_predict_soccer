from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.repository.base.repository_db import SessionLocal
from src.service_ia.model.match import PredictionLedger


class PredictionLedgerRepository:
    """Persistenza del Prediction Ledger (BET-06): stesso pattern "semplice"
    gia' in uso in `OddsSnapshotRepository` (sessione per operazione, nessuno
    stato condiviso tra chiamate)."""

    def save(self, prediction: PredictionLedger) -> PredictionLedger:
        with SessionLocal() as session:
            merged = session.merge(prediction)
            session.commit()
            session.refresh(merged)
            return merged

    def get_by_id(self, id_prediction: str) -> Optional[PredictionLedger]:
        with SessionLocal() as session:
            return session.get(PredictionLedger, id_prediction)

    def find_existing(
        self,
        fixture_id: int,
        market: str,
        outcome: str,
        model_run_id: Optional[str],
    ) -> Optional[PredictionLedger]:
        """Cerca una prediction gia' salvata per la STESSA chiave logica
        (fixture/market/outcome/model_run_id): usato per l'idempotenza di
        `log_prediction` (mai un duplicato silenzioso per lo stesso run)."""
        with SessionLocal() as session:
            query = session.query(PredictionLedger).filter(
                PredictionLedger.fixture_id == int(fixture_id),
                PredictionLedger.market == market,
                PredictionLedger.outcome == outcome,
            )
            if model_run_id is None:
                query = query.filter(PredictionLedger.model_run_id.is_(None))
            else:
                query = query.filter(PredictionLedger.model_run_id == model_run_id)
            return query.order_by(PredictionLedger.created_at.desc()).first()

    def list_for_fixture(self, fixture_id: int, market: Optional[str] = None) -> list[PredictionLedger]:
        with SessionLocal() as session:
            query = session.query(PredictionLedger).filter(PredictionLedger.fixture_id == int(fixture_id))
            if market:
                query = query.filter(PredictionLedger.market == market)
            return query.order_by(PredictionLedger.created_at.asc()).all()

    def list_pending_settlement(self, before: Optional[datetime] = None, limit: int = 500) -> list[PredictionLedger]:
        """Prediction NON ancora settled con kickoff gia' passato (rispetto a
        `before`, default now): candidate per `settle_pending` — mai una
        prediction settled prima del kickoff (acceptance criteria implicito,
        "Settlement dopo risultato")."""
        cutoff = before or datetime.now(timezone.utc)
        with SessionLocal() as session:
            query = (
                session.query(PredictionLedger)
                .filter(PredictionLedger.is_settled.is_(False))
                .filter(PredictionLedger.kickoff_at.is_not(None))
                .filter(PredictionLedger.kickoff_at <= cutoff)
                .order_by(PredictionLedger.kickoff_at.asc())
                .limit(limit)
            )
            return query.all()

    def list_all(
        self,
        market: Optional[str] = None,
        is_settled: Optional[bool] = None,
        since: Optional[datetime] = None,
        limit: int = 200,
    ) -> list[PredictionLedger]:
        """`since` (OPS-03, opzionale, default `None` = comportamento
        INVARIATO): filtra lato query `created_at >= since`, per finestre
        temporali (prediction volume/ROI rolling) senza dover caricare
        l'intera tabella e filtrare in memoria."""
        with SessionLocal() as session:
            query = session.query(PredictionLedger)
            if market:
                query = query.filter(PredictionLedger.market == market)
            if is_settled is not None:
                query = query.filter(PredictionLedger.is_settled.is_(bool(is_settled)))
            if since is not None:
                query = query.filter(PredictionLedger.created_at >= since)
            return query.order_by(PredictionLedger.created_at.desc()).limit(max(0, limit)).all()


