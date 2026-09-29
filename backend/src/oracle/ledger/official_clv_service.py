"""CLV costruito esclusivamente dalle PLAY del Prediction Ledger ufficiale."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import Any, Optional

from src.oracle.backtest.clv import ClvResult, build_clv_result, compute_clv_report
from src.oracle.ledger.official_performance_service import OFFICIAL_COHORT, OFFICIAL_SOURCE
from src.repository.odds_snapshot_repository import OddsSnapshotRepository
from src.repository.prediction_ledger_repository import PredictionLedgerRepository


def _summary(results: list[ClvResult]) -> dict[str, Any]:
    report = compute_clv_report(results)
    payload = dataclasses.asdict(report.overall)
    payload.update(
        {
            "total_records": report.total_considered,
            "available_records": report.available_count,
            "unavailable_records": report.unavailable_count,
            "coverage": (
                report.available_count / report.total_considered
                if report.total_considered
                else None
            ),
        }
    )
    return payload


def _serialize(result: ClvResult, id_prediction: Optional[str] = None) -> dict[str, Any]:
    payload = dataclasses.asdict(result)
    payload["id_prediction"] = id_prediction
    payload["unavailable_reason"] = payload.pop("reason")
    return payload


class OfficialClvService:
    def __init__(
        self,
        ledger_repo: Optional[PredictionLedgerRepository] = None,
        snapshot_repo: Optional[OddsSnapshotRepository] = None,
    ):
        self.ledger_repo = ledger_repo or PredictionLedgerRepository()
        self.snapshot_repo = snapshot_repo or OddsSnapshotRepository()

    def _result_for_row(self, row: Any) -> ClvResult:
        return build_clv_result(
            market=row.market,
            outcome=row.outcome,
            odd_at_bet=row.odd,
            p_fair_at_bet=row.p_market_fair,
            fixture_id=row.fixture_id,
            kickoff_at=row.kickoff_at,
            period=row.period,
            line=row.line,
            model_name=row.model_name,
            snapshot_repo=self.snapshot_repo,
        )

    def report(
        self,
        *,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        market: Optional[str] = None,
        league: Optional[int] = None,
        model_name: Optional[str] = None,
        policy_version: Optional[str] = None,
    ) -> dict[str, Any]:
        rows = self.ledger_repo.list_all(
            market=market,
            since=since,
            until=until,
            cohort=OFFICIAL_COHORT,
            league=league,
            model_name=model_name,
            policy_version=policy_version,
            limit=1_000_000,
        )
        rows = [row for row in rows if row.decision == "PLAY" and row.odd is not None]
        paired = [(row, self._result_for_row(row)) for row in rows]
        results = [result for _, result in paired]

        breakdowns: dict[str, dict[str, Any]] = {}
        dimensions = {
            "market": lambda row: row.market,
            "model": lambda row: row.model_name or "unknown",
            "league": lambda row: str(row.league or "unknown"),
            "period": lambda row: row.period or "unknown",
        }
        for dimension, key_fn in dimensions.items():
            groups: dict[str, list[ClvResult]] = {}
            for row, result in paired:
                groups.setdefault(str(key_fn(row)), []).append(result)
            breakdowns[dimension] = {key: _summary(value) for key, value in sorted(groups.items())}

        return {
            "source": OFFICIAL_SOURCE,
            "cohort": OFFICIAL_COHORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "filters": {
                "since": since.isoformat() if since else None,
                "until": until.isoformat() if until else None,
                "market": market,
                "league": league,
                "model": model_name,
                "policy_version": policy_version,
            },
            "sample_size": len(rows),
            "settled_count": sum(1 for row in rows if row.is_settled),
            "void_count": sum(1 for row in rows if str(row.settlement_status or "").startswith("void_")),
            "pending_count": sum(1 for row in rows if not row.is_settled),
            "overall": _summary(results),
            "breakdowns": breakdowns,
            "rows": [_serialize(result, row.id_prediction) for row, result in paired],
        }

    def detail(self, id_prediction: str) -> Optional[dict[str, Any]]:
        row = self.ledger_repo.get_by_id(id_prediction)
        if row is None or row.cohort != OFFICIAL_COHORT:
            return None
        return {
            "source": OFFICIAL_SOURCE,
            "cohort": OFFICIAL_COHORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **_serialize(self._result_for_row(row), row.id_prediction),
        }
