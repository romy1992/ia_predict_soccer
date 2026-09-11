from __future__ import annotations

from datetime import datetime
from typing import Optional

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
