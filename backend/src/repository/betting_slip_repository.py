from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import selectinload

from src.repository.base.repository_db import SessionLocal
from src.service_ia.model.match import BettingSlip, BettingSlipPick


class BettingSlipRepository:
    """Persistenza atomica e letture eager delle schedine ufficiali."""

    def get_by_capture_key(self, capture_key: str) -> Optional[BettingSlip]:
        with SessionLocal() as session:
            return (
                session.query(BettingSlip)
                .options(selectinload(BettingSlip.picks))
                .filter(BettingSlip.capture_key == capture_key)
                .first()
            )

    def save_with_picks(self, slip: BettingSlip, picks: list[BettingSlipPick]) -> tuple[BettingSlip, bool]:
        with SessionLocal() as session:
            existing = (
                session.query(BettingSlip)
                .options(selectinload(BettingSlip.picks))
                .filter(BettingSlip.capture_key == slip.capture_key)
                .first()
            )
            if existing is not None:
                return existing, False
            slip.picks = picks
            session.add(slip)
            session.commit()
            session.refresh(slip)
            _ = list(slip.picks)
            return slip, True

    def list_all(
        self,
        *,
        reference_date: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
    ) -> list[BettingSlip]:
        with SessionLocal() as session:
            query = session.query(BettingSlip).options(selectinload(BettingSlip.picks))
            if reference_date:
                query = query.filter(BettingSlip.reference_date == reference_date)
            if status:
                query = query.filter(BettingSlip.status == status)
            rows = query.order_by(BettingSlip.created_at.desc(), BettingSlip.id.asc()).limit(max(0, limit)).all()
            for row in rows:
                _ = list(row.picks)
            return rows

    def list_pending(self, before: Optional[datetime] = None, limit: int = 500) -> list[BettingSlip]:
        cutoff = before or datetime.now(timezone.utc)
        with SessionLocal() as session:
            rows = (
                session.query(BettingSlip)
                .options(selectinload(BettingSlip.picks))
                .join(BettingSlipPick)
                .filter(BettingSlip.status == "PENDING")
                .filter(BettingSlipPick.kickoff_at <= cutoff)
                .distinct()
                .order_by(BettingSlip.created_at.asc())
                .limit(limit)
                .all()
            )
            for row in rows:
                _ = list(row.picks)
            return rows

    def save_settlement(self, slip: BettingSlip) -> BettingSlip:
        with SessionLocal() as session:
            merged = session.merge(slip)
            session.commit()
            session.refresh(merged)
            _ = list(merged.picks)
            return merged
