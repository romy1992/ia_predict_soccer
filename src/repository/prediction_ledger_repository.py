from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func

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

    def get_earliest_created_date(self) -> Optional[datetime]:
        """Prima data in assoluto in cui e' stata salvata una prediction
        (MIN(created_at)): usato per costruire l'elenco date accumulato di
        `DashboardService.get_available_dates` ("giorno 1 di previsioni" ->
        oggi), al posto di un calendario libero in UI."""
        with SessionLocal() as session:
            return session.query(func.min(PredictionLedger.created_at)).scalar()

    def find_existing(
        self,
        fixture_id: int,
        market: str,
        outcome: str,
        model_run_id: Optional[str],
        cohort: Optional[str] = None,
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
            if cohort is not None:
                query = query.filter(PredictionLedger.cohort == cohort)
            return query.order_by(PredictionLedger.created_at.desc()).first()

    def find_by_capture_key(self, capture_key: str) -> Optional[PredictionLedger]:
        with SessionLocal() as session:
            return (
                session.query(PredictionLedger)
                .filter(PredictionLedger.capture_key == capture_key)
                .first()
            )

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
        until: Optional[datetime] = None,
        cohort: Optional[str] = None,
        outcome: Optional[str] = None,
        league: Optional[int] = None,
        model_name: Optional[str] = None,
        model_run_id: Optional[str] = None,
        policy_version: Optional[str] = None,
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
            if until is not None:
                query = query.filter(PredictionLedger.created_at <= until)
            if cohort is not None:
                query = query.filter(PredictionLedger.cohort == cohort)
            if outcome is not None:
                query = query.filter(PredictionLedger.outcome == outcome)
            if league is not None:
                query = query.filter(PredictionLedger.league == int(league))
            if model_name is not None:
                query = query.filter(PredictionLedger.model_name == model_name)
            if model_run_id is not None:
                query = query.filter(PredictionLedger.model_run_id == model_run_id)
            if policy_version is not None:
                query = query.filter(PredictionLedger.policy_version == policy_version)
            return query.order_by(PredictionLedger.captured_at.desc()).limit(max(0, limit)).all()


