"""Performance delle sole PLAY della coorte ufficiale/paper."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.oracle.backtest.betting_backtester import edge_bucket_label, max_drawdown_from_cumulative
from src.oracle.ledger.prediction_ledger import VOID_STATUSES
from src.repository.prediction_ledger_repository import PredictionLedgerRepository


OFFICIAL_SOURCE = "scheduled_official_capture"
OFFICIAL_COHORT = "official_paper"


def _value(row: Any, name: str, default: Any = None) -> Any:
    return row.get(name, default) if isinstance(row, dict) else getattr(row, name, default)


def _odd_bucket(odd: Optional[float]) -> str:
    if odd is None:
        return "unknown"
    value = float(odd)
    if value < 1.5:
        return "<1.50"
    if value < 2.0:
        return "[1.50,2.00)"
    if value < 3.0:
        return "[2.00,3.00)"
    return ">=3.00"


def _metric_row(rows: list[Any]) -> dict[str, Any]:
    placed = [
        row
        for row in rows
        if _value(row, "decision") == "PLAY"
        and _value(row, "odd") is not None
        and not str(_value(row, "settlement_status") or "").startswith(("not_placed", "skipped"))
    ]
    pending = [row for row in placed if not bool(_value(row, "is_settled"))]
    wins = [row for row in placed if _value(row, "settlement_status") == "settled_win"]
    losses = [row for row in placed if _value(row, "settlement_status") == "settled_loss"]
    voids = [
        row
        for row in placed
        if str(_value(row, "settlement_status") or "") in VOID_STATUSES
        or str(_value(row, "settlement_status") or "").startswith("void_")
    ]
    not_placed = [
        row
        for row in rows
        if str(_value(row, "settlement_status") or "").startswith(("not_placed", "skipped"))
        or _value(row, "decision") != "PLAY"
        or _value(row, "odd") is None
    ]

    gross_stake = float(sum(float(_value(row, "stake", 0.0) or 0.0) for row in placed))
    void_stake = float(sum(float(_value(row, "stake", 0.0) or 0.0) for row in voids))
    active_stake = gross_stake - void_stake
    total_profit = float(sum(float(_value(row, "pnl", 0.0) or 0.0) for row in placed))
    settled_decisions = wins + losses

    ordered = sorted(
        [row for row in placed if _value(row, "pnl") is not None],
        key=lambda row: (
            str(_value(row, "kickoff_at") or ""),
            int(_value(row, "fixture_id", 0) or 0),
        ),
    )
    cumulative: list[float] = []
    running = 0.0
    for row in ordered:
        running += float(_value(row, "pnl", 0.0) or 0.0)
        cumulative.append(running)

    def average(name: str) -> Optional[float]:
        values = [float(_value(row, name)) for row in placed if _value(row, name) is not None]
        return (sum(values) / len(values)) if values else None

    void_by_reason: dict[str, int] = {}
    for row in voids:
        reason = str(_value(row, "settlement_status") or "void_unknown")
        void_by_reason[reason] = void_by_reason.get(reason, 0) + 1

    return {
        "plays": len(placed),
        "pending": len(pending),
        "settled": len(placed) - len(pending),
        "wins": len(wins),
        "losses": len(losses),
        "void": len(voids),
        "void_by_reason": dict(sorted(void_by_reason.items())),
        "not_placed": len(not_placed),
        "gross_stake": gross_stake,
        "void_stake": void_stake,
        "active_stake": active_stake,
        "total_profit": total_profit,
        "roi": (total_profit / active_stake) if active_stake > 0 else None,
        "gross_roi": (total_profit / gross_stake) if gross_stake > 0 else None,
        "hit_rate": (len(wins) / len(settled_decisions)) if settled_decisions else None,
        "avg_odd": average("odd"),
        "avg_prob_edge": average("prob_edge"),
        "avg_ev": average("ev"),
        # Include il capitale iniziale (0): una prima perdita deve produrre
        # drawdown, non diventare artificialmente il primo "peak".
        "max_drawdown": max_drawdown_from_cumulative([0.0, *cumulative]),
        "sample_size": len(placed),
    }


def compute_official_performance(rows: Iterable[Any]) -> dict[str, Any]:
    official_rows = [
        row
        for row in rows
        if _value(row, "source") == OFFICIAL_SOURCE and _value(row, "cohort") == OFFICIAL_COHORT
    ]
    dimensions = {
        "day": lambda row: (
            _value(row, "kickoff_at").date().isoformat()
            if isinstance(_value(row, "kickoff_at"), datetime)
            else str(_value(row, "kickoff_at") or _value(row, "created_at") or "unknown")[:10]
        ),
        "market": lambda row: _value(row, "market") or "unknown",
        "outcome": lambda row: _value(row, "outcome") or "unknown",
        "league": lambda row: str(_value(row, "league") or "unknown"),
        "model": lambda row: _value(row, "model_name") or "unknown",
        "model_run_id": lambda row: _value(row, "model_run_id") or "unknown",
        "policy_version": lambda row: _value(row, "policy_version") or "unknown",
        "cohort": lambda row: _value(row, "cohort") or "unknown",
        "period": lambda row: _value(row, "period") or "unknown",
        "edge_bucket": lambda row: edge_bucket_label(_value(row, "prob_edge")),
        "odd_bucket": lambda row: _odd_bucket(_value(row, "odd")),
    }
    breakdowns: dict[str, dict[str, dict[str, Any]]] = {}
    for name, key_fn in dimensions.items():
        groups: dict[str, list[Any]] = {}
        for row in official_rows:
            groups.setdefault(str(key_fn(row)), []).append(row)
        breakdowns[name] = {key: _metric_row(value) for key, value in sorted(groups.items())}

    return {"overall": _metric_row(official_rows), "breakdowns": breakdowns}


class OfficialPerformanceService:
    def __init__(self, repo: Optional[PredictionLedgerRepository] = None):
        self.repo = repo or PredictionLedgerRepository()

    def report(
        self,
        *,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        market: Optional[str] = None,
        league: Optional[int] = None,
        model_name: Optional[str] = None,
        policy_version: Optional[str] = None,
        cohort: str = OFFICIAL_COHORT,
    ) -> dict[str, Any]:
        rows = self.repo.list_all(
            market=market,
            since=since,
            until=until,
            cohort=cohort,
            league=league,
            model_name=model_name,
            policy_version=policy_version,
            limit=1_000_000,
        )
        metrics = compute_official_performance(rows)
        overall = metrics["overall"]
        return {
            "source": OFFICIAL_SOURCE,
            "cohort": cohort,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "filters": {
                "since": since.isoformat() if since else None,
                "until": until.isoformat() if until else None,
                "market": market,
                "league": league,
                "model": model_name,
                "policy_version": policy_version,
            },
            "sample_size": overall["sample_size"],
            "settled_count": overall["settled"],
            "void_count": overall["void"],
            "pending_count": overall["pending"],
            **metrics,
        }
