from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.repository.base.repository_db import SessionLocal
from src.service_ia.model.match import OddsSnapshot


class OddsSnapshotRepository:
    def save_many(self, snapshots: list[OddsSnapshot]) -> None:
        if not snapshots:
            return

        with SessionLocal() as session:
            for snapshot in snapshots:
                session.merge(snapshot)
            session.commit()

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


