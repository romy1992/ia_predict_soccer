from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from src.repository.base.repository_db import SessionLocal
from src.service_ia.model.match import BettingSlipProposalSnapshot


class BettingSlipProposalRepository:
    """Persistenza append-only delle revisioni prodotte dal generatore."""

    def save_revision(
        self,
        snapshot: BettingSlipProposalSnapshot,
    ) -> tuple[BettingSlipProposalSnapshot, bool]:
        with SessionLocal() as session:
            existing = (
                session.query(BettingSlipProposalSnapshot)
                .filter(BettingSlipProposalSnapshot.snapshot_key == snapshot.snapshot_key)
                .first()
            )
            if existing is not None:
                # Payload invariato (stesso snapshot_key): nessuna nuova riga,
                # ma la lineage va comunque "ri-confermata" dal giro corrente
                # - altrimenti una combinazione stabile sparirebbe dalla vista
                # corrente (`list_current`) pur essendo ancora valida. MAI
                # toccare `is_latest`/`generated_at` qui: quei campi restano
                # il segnale "e' cambiato qualcosa", non "e' stata rivista".
                if snapshot.last_confirmed_at is not None:
                    existing.last_confirmed_at = snapshot.last_confirmed_at
                    session.commit()
                    session.refresh(existing)
                return existing, False

            previous = (
                session.query(BettingSlipProposalSnapshot)
                .filter(BettingSlipProposalSnapshot.logical_slip_id == snapshot.logical_slip_id)
                .filter(BettingSlipProposalSnapshot.is_latest.is_(True))
                .order_by(BettingSlipProposalSnapshot.generated_at.desc())
                .first()
            )
            if previous is not None:
                previous.is_latest = False
                snapshot.supersedes_id = previous.id

            session.add(snapshot)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                concurrent = (
                    session.query(BettingSlipProposalSnapshot)
                    .filter(BettingSlipProposalSnapshot.snapshot_key == snapshot.snapshot_key)
                    .one()
                )
                return concurrent, False
            session.refresh(snapshot)
            return snapshot, True

    def list_all(
        self,
        *,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        reference_date: Optional[str] = None,
        latest_only: bool = False,
        limit: int = 100_000,
    ) -> list[BettingSlipProposalSnapshot]:
        with SessionLocal() as session:
            query = session.query(BettingSlipProposalSnapshot)
            if since is not None:
                query = query.filter(BettingSlipProposalSnapshot.generated_at >= since)
            if until is not None:
                query = query.filter(BettingSlipProposalSnapshot.generated_at <= until)
            if reference_date is not None:
                query = query.filter(BettingSlipProposalSnapshot.reference_date == reference_date)
            if latest_only:
                query = query.filter(BettingSlipProposalSnapshot.is_latest.is_(True))
            return (
                query.order_by(
                    BettingSlipProposalSnapshot.generated_at.desc(),
                    BettingSlipProposalSnapshot.id.asc(),
                )
                .limit(max(0, limit))
                .all()
            )

    def list_current(
        self,
        *,
        reference_date: str,
        limit: int = 200,
    ) -> list[BettingSlipProposalSnapshot]:
        """Righe `is_latest=True` confermate dal giro di generazione PIU'
        RECENTE per `reference_date` (stesso `last_confirmed_at`, mai un
        giro precedente) - la "lista di oggi" limitata, a differenza di
        `list_all(latest_only=True)` che ritorna TUTTO l'accumulato del
        giorno (ogni lineage mai generata, anche se un giro successivo non
        la ripropone piu'). Non tocca liquidazione/ROI: quelli restano su
        `is_latest`/`list_pending_settlement`, invariati."""
        with SessionLocal() as session:
            max_confirmed_at = (
                session.query(func.max(BettingSlipProposalSnapshot.last_confirmed_at))
                .filter(BettingSlipProposalSnapshot.reference_date == reference_date)
                .filter(BettingSlipProposalSnapshot.is_latest.is_(True))
                .scalar()
            )
            if max_confirmed_at is None:
                return []
            return (
                session.query(BettingSlipProposalSnapshot)
                .filter(BettingSlipProposalSnapshot.reference_date == reference_date)
                .filter(BettingSlipProposalSnapshot.is_latest.is_(True))
                .filter(BettingSlipProposalSnapshot.last_confirmed_at == max_confirmed_at)
                .order_by(BettingSlipProposalSnapshot.id.asc())
                .limit(max(0, limit))
                .all()
            )

    def list_pending_settlement(
        self,
        *,
        before: datetime,
        limit: int = 500,
    ) -> list[BettingSlipProposalSnapshot]:
        with SessionLocal() as session:
            return (
                session.query(BettingSlipProposalSnapshot)
                .filter(BettingSlipProposalSnapshot.shadow_status == "PENDING")
                .filter(BettingSlipProposalSnapshot.generated_at < before)
                .filter(
                    BettingSlipProposalSnapshot.reference_date
                    <= before.date().isoformat()
                )
                .order_by(BettingSlipProposalSnapshot.generated_at.asc())
                .limit(max(0, limit))
                .all()
            )

    def save_settlement(
        self,
        snapshot: BettingSlipProposalSnapshot,
    ) -> BettingSlipProposalSnapshot:
        with SessionLocal() as session:
            merged = session.merge(snapshot)
            session.commit()
            session.refresh(merged)
            return merged
