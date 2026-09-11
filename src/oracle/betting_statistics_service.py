from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from src.oracle.betslip.official_betslip_service import OfficialBetslipService
from src.oracle.ledger.official_performance_service import (
    OFFICIAL_COHORT,
    OFFICIAL_SOURCE,
    _metric_row,
)
from src.repository.betting_slip_proposal_repository import BettingSlipProposalRepository
from src.repository.prediction_ledger_repository import PredictionLedgerRepository


def _proposal_bucket(rows: list[Any]) -> dict[str, Any]:
    expected = [
        float(row.combined_expected_roi)
        for row in rows
        if row.combined_expected_roi is not None
    ]
    return {
        "generated": len(rows),
        "play": sum(row.situation == "PLAY" for row in rows),
        "borderline": sum(row.situation == "BORDERLINE" for row in rows),
        "no_bet": sum(row.situation == "NO BET" for row in rows),
        "average_combined_odd": (
            sum(float(row.combined_odd) for row in rows) / len(rows) if rows else None
        ),
        "average_expected_roi": sum(expected) / len(expected) if expected else None,
    }


class BettingStatisticsService:
    """Vista unica; non mescola mai proposte e performance ufficiale."""

    def __init__(
        self,
        *,
        ledger_repo: Optional[PredictionLedgerRepository] = None,
        proposal_repo: Optional[BettingSlipProposalRepository] = None,
        official_betslip_service: Optional[OfficialBetslipService] = None,
    ):
        self.ledger_repo = ledger_repo or PredictionLedgerRepository()
        self.proposal_repo = proposal_repo or BettingSlipProposalRepository()
        self.official_betslip_service = official_betslip_service or OfficialBetslipService()

    def report(self, *, days: int = 30) -> dict[str, Any]:
        if days not in {7, 30, 90, 365}:
            raise ValueError("days deve essere 7, 30, 90 o 365")
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=days)
        since_date = since.date().isoformat()
        until_date = now.date().isoformat()

        official_predictions = [
            row
            for row in self.ledger_repo.list_all(
                since=since,
                until=now,
                cohort=OFFICIAL_COHORT,
                limit=1_000_000,
            )
            if row.source == OFFICIAL_SOURCE
        ]
        market_groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for row in official_predictions:
            kickoff = row.kickoff_at or row.created_at
            day = kickoff.date().isoformat() if kickoff else "unknown"
            market_groups[(day, row.market or "unknown")].append(row)
        market_daily = [
            {"date": day, "market": market, **_metric_row(rows)}
            for (day, market), rows in sorted(market_groups.items(), reverse=True)
        ]

        # La finestra riguarda QUANDO la proposta è stata generata, non la
        # data dell'evento: così una schedina futura salvata oggi compare
        # subito nelle statistiche di attività.
        proposals = self.proposal_repo.list_all(
            since=since,
            until=now,
            limit=1_000_000,
        )
        latest_proposals = [row for row in proposals if row.is_latest]
        proposal_daily_groups: dict[str, list[Any]] = defaultdict(list)
        proposal_profile_groups: dict[str, list[Any]] = defaultdict(list)
        for row in proposals:
            proposal_daily_groups[row.reference_date].append(row)
            proposal_profile_groups[row.profile].append(row)

        official_slips = self.official_betslip_service.statistics(
            since_date=since_date,
            until_date=until_date,
        )
        official_overall = _metric_row(official_predictions)
        return {
            "generated_at": now.isoformat(),
            "filters": {"days": days, "since": since_date, "until": until_date},
            "overview": {
                "official_predictions": official_overall,
                "proposals": {
                    **_proposal_bucket(proposals),
                    "latest": len(latest_proposals),
                    "revisions": len(proposals) - len(latest_proposals),
                },
                "official_slips": official_slips,
            },
            "markets": {"daily": market_daily},
            "slips": {
                "proposals_daily": [
                    {"date": key, **_proposal_bucket(value)}
                    for key, value in sorted(proposal_daily_groups.items(), reverse=True)
                ],
                "proposals_by_profile": {
                    key: _proposal_bucket(value)
                    for key, value in sorted(proposal_profile_groups.items())
                },
                "official_daily": official_slips.get("by_day", {}),
                "official_by_profile": official_slips.get("by_profile", {}),
                "official_by_event_count": official_slips.get("by_event_count", {}),
                "official_by_market_combination": official_slips.get(
                    "by_market_combination",
                    {},
                ),
                "bankroll_curve": official_slips.get("bankroll_curve", []),
            },
        }
