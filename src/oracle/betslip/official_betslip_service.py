from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import selectinload

from src.oracle.betslip.betslip_builder import GeneratedSlip, generate_betslips
from src.oracle.betslip.pick_pool import CandidatePick
from src.oracle.ledger.ledger_service import PredictionLedgerService
from src.oracle.ledger.settlement_rules import outcome_wins
from src.repository.base.repository_db import SessionLocal
from src.repository.betting_slip_repository import BettingSlipRepository
from src.service_ia.model.match import BettingSlip, BettingSlipPick, Match, PredictionLedger


OFFICIAL_COHORT = "official_paper"


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class OfficialBetslipService:
    """Cattura, settlement e statistiche delle sole schedine persistite."""

    def __init__(
        self,
        repo: Optional[BettingSlipRepository] = None,
        ledger_service: Optional[PredictionLedgerService] = None,
    ):
        self.repo = repo or BettingSlipRepository()
        self.ledger_service = ledger_service or PredictionLedgerService()

    @staticmethod
    def _capture_key(reference_date: str, slip: GeneratedSlip) -> str:
        selections = sorted(
            (leg.fixture_id, leg.market, leg.line or "", leg.outcome, leg.model_run_id or "") for leg in slip.legs
        )
        raw = json.dumps(
            {
                "date": reference_date,
                "profile": slip.profile_name,
                "profile_version": slip.profile_version,
                "policy": slip.decision_policy_version,
                "correlation": slip.correlation_ruleset_version,
                "selections": selections,
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:48]

    def _official_candidates(self, now: datetime) -> tuple[dict[str, list[CandidatePick]], dict[tuple, str]]:
        grouped: dict[str, list[CandidatePick]] = defaultdict(list)
        prediction_ids: dict[tuple, str] = {}
        with SessionLocal() as session:
            rows = (
                session.query(PredictionLedger, Match)
                .outerjoin(Match, Match.id_fixture == PredictionLedger.fixture_id)
                .filter(PredictionLedger.cohort == OFFICIAL_COHORT)
                .filter(PredictionLedger.decision == "PLAY")
                .filter(PredictionLedger.kickoff_at.is_not(None))
                .filter(PredictionLedger.kickoff_at > now)
                .order_by(PredictionLedger.kickoff_at.asc())
                .all()
            )
            for row, match in rows:
                kickoff = _aware(row.kickoff_at)
                key = (row.fixture_id, row.market, row.line or "", row.outcome, row.model_run_id or "")
                prediction_ids[key] = row.id_prediction
                grouped[kickoff.date().isoformat()].append(
                    CandidatePick(
                        fixture_id=row.fixture_id,
                        market=row.market,
                        outcome=row.outcome,
                        decision=row.value_label or row.decision,
                        p_model=row.p_model,
                        p_market_fair=row.p_market_fair,
                        odd=row.odd,
                        fair_odd=row.fair_odd,
                        prob_edge=row.prob_edge,
                        ev=row.ev,
                        samples=row.bookmaker_count,
                        model_run_id=row.model_run_id,
                        model_name=row.model_name,
                        policy_version=row.policy_version,
                        kickoff_at=kickoff.isoformat(),
                        competition=match.title_league if match else None,
                        home_team=match.name_home if match else None,
                        away_team=match.name_away if match else None,
                        line=row.line,
                        model_void_odd=row.model_void_odd,
                        market_fair_odd=row.market_fair_odd,
                        odds_edge_absolute=row.odds_edge_absolute,
                        odds_edge_percent=row.odds_edge_percent,
                        expected_roi_percent=row.expected_roi_percent,
                    )
                )
        return grouped, prediction_ids

    def capture_from_official_ledger(self, now: Optional[datetime] = None, stake: float = 1.0) -> dict[str, Any]:
        captured_at = _aware(now or datetime.now(timezone.utc))
        grouped, prediction_ids = self._official_candidates(captured_at)
        report: dict[str, Any] = {"betslips_created": 0, "betslips_duplicates": 0, "betslips_skipped": 0}
        for reference_date, candidates in grouped.items():
            generation = generate_betslips(candidates)
            for slips in generation.profiles.values():
                for generated in slips:
                    if generated.situation != "PLAY":
                        report["betslips_skipped"] += 1
                        continue
                    if any(
                        not leg.kickoff_at or _aware(datetime.fromisoformat(leg.kickoff_at)) <= captured_at
                        for leg in generated.legs
                    ):
                        report["betslips_skipped"] += 1
                        continue
                    model_versions = sorted({leg.model_run_id for leg in generated.legs if leg.model_run_id})
                    slip = BettingSlip(
                        capture_key=self._capture_key(reference_date, generated),
                        reference_date=reference_date,
                        profile=generated.profile_name,
                        initial_situation=generated.situation,
                        initial_reason=generated.situation_reason,
                        status="PENDING",
                        event_count=generated.n_legs,
                        combined_odd=generated.combined_odd,
                        naive_probability=generated.naive_probability,
                        adjusted_probability=generated.adjusted_probability,
                        combined_model_void_odd=generated.combined_model_void_odd,
                        combined_edge_absolute=generated.combined_edge_absolute,
                        combined_edge_percent=generated.combined_edge_percent,
                        combined_expected_roi=generated.combined_expected_roi,
                        risk_score=generated.risk_score,
                        combined_play_threshold=generated.combined_play_threshold,
                        slip_min_edge_percent=generated.slip_min_edge_percent,
                        stake=stake,
                        potential_return=stake * generated.combined_odd,
                        model_version=",".join(model_versions) or None,
                        policy_version=generated.decision_policy_version,
                        correlation_version=generated.correlation_ruleset_version,
                        created_at=captured_at,
                    )
                    picks: list[BettingSlipPick] = []
                    for position, leg in enumerate(generated.legs, start=1):
                        key = (leg.fixture_id, leg.market, leg.line or "", leg.outcome, leg.model_run_id or "")
                        picks.append(
                            BettingSlipPick(
                                prediction_id=prediction_ids.get(key),
                                position=position,
                                fixture_id=leg.fixture_id,
                                competition=leg.competition,
                                kickoff_at=_aware(datetime.fromisoformat(leg.kickoff_at)),
                                home_team=leg.home_team,
                                away_team=leg.away_team,
                                market=leg.market,
                                line=leg.line,
                                outcome=leg.outcome,
                                p_model=float(leg.p_model),
                                market_odd=float(leg.odd),
                                model_void_odd=leg.model_void_odd,
                                market_fair_odd=leg.market_fair_odd,
                                odds_edge_absolute=leg.odds_edge_absolute,
                                odds_edge_percent=leg.odds_edge_percent,
                                expected_roi=leg.ev,
                                situation=leg.decision,
                                bookmakers_count=leg.samples,
                                model_version=leg.model_run_id,
                                policy_version=leg.policy_version,
                                status="PENDING",
                            )
                        )
                    _, created = self.repo.save_with_picks(slip, picks)
                    report["betslips_created"] += int(created)
                    report["betslips_duplicates"] += int(not created)
        return report

    def settle_pending(self, before: Optional[datetime] = None, limit: int = 500) -> dict[str, Any]:
        now = _aware(before or datetime.now(timezone.utc))
        report = {"betslips_candidates": 0, "betslips_settled": 0, "betslips_pending": 0, "errors": []}
        for slip in self.repo.list_pending(before=now, limit=limit):
            report["betslips_candidates"] += 1
            try:
                for pick in slip.picks:
                    if pick.status != "PENDING" or _aware(pick.kickoff_at) > now:
                        continue
                    resolution = self.ledger_service._resolve_outcome_for_fixture(
                        fixture_id=pick.fixture_id,
                        market=pick.market,
                        line=pick.line,
                    )
                    if resolution.pending:
                        continue
                    pick.settled_at = now
                    if resolution.void_status or resolution.is_push:
                        pick.status = "VOID"
                        pick.void_reason = resolution.void_status or "void_push"
                    else:
                        won = outcome_wins(pick.market, pick.outcome, resolution.actual_outcome)
                        if won is None:
                            pick.status = "VOID"
                            pick.void_reason = "void_market_rule"
                        else:
                            pick.status = "WON" if won else "LOST"
                    with SessionLocal() as session:
                        match = session.query(Match).filter(Match.id_fixture == pick.fixture_id).first()
                        if match and match.score_home is not None and match.score_away is not None:
                            pick.final_score = f"{match.score_home}-{match.score_away}"

                statuses = [pick.status for pick in slip.picks]
                if "LOST" in statuses:
                    slip.status = "LOST"
                elif "PENDING" in statuses:
                    slip.status = "PENDING"
                elif statuses and all(status == "VOID" for status in statuses):
                    slip.status = "VOID"
                elif statuses and all(status in {"WON", "VOID"} for status in statuses):
                    slip.status = "WON"

                if slip.status != "PENDING":
                    active_odds = [pick.market_odd for pick in slip.picks if pick.status != "VOID"]
                    effective = 1.0
                    for odd in active_odds:
                        effective *= float(odd)
                    slip.effective_combined_odd = effective
                    slip.actual_return = (
                        slip.stake * effective if slip.status == "WON" else slip.stake if slip.status == "VOID" else 0.0
                    )
                    slip.realized_profit = slip.actual_return - slip.stake
                    slip.settled_at = now
                    report["betslips_settled"] += 1
                else:
                    report["betslips_pending"] += 1
                self.repo.save_settlement(slip)
            except Exception as exc:
                report["errors"].append({"slip_id": slip.id, "message": str(exc)})
        return report

    def list_official(self, reference_date: Optional[str] = None, status: Optional[str] = None, limit: int = 200):
        return [row.to_dict() for row in self.repo.list_all(reference_date=reference_date, status=status, limit=limit)]

    def statistics(self) -> dict[str, Any]:
        rows = sorted(self.repo.list_all(limit=1_000_000), key=lambda row: (row.created_at, row.id))
        settled = [row for row in rows if row.status in {"WON", "LOST", "VOID"}]
        active_settled = [row for row in settled if row.status != "VOID"]
        total_stake = sum(float(row.stake) for row in active_settled)
        total_return = sum(float(row.actual_return or 0.0) for row in settled)
        net_profit = sum(float(row.realized_profit or 0.0) for row in settled)
        bankroll = peak = drawdown = 0.0
        curve = []
        for row in settled:
            bankroll += float(row.realized_profit or 0.0)
            peak = max(peak, bankroll)
            drawdown = max(drawdown, peak - bankroll)
            curve.append({"settled_at": row.settled_at.isoformat() if row.settled_at else None, "bankroll": bankroll})

        def aggregate(key):
            buckets: dict[str, dict[str, Any]] = {}
            for row in rows:
                label = str(key(row))
                bucket = buckets.setdefault(label, {"total": 0, "won": 0, "lost": 0, "void": 0, "profit": 0.0})
                bucket["total"] += 1
                bucket[row.status.lower()] = bucket.get(row.status.lower(), 0) + 1
                bucket["profit"] += float(row.realized_profit or 0.0)
            return buckets

        return {
            "total": len(rows),
            "pending": sum(row.status == "PENDING" for row in rows),
            "won": sum(row.status == "WON" for row in rows),
            "lost": sum(row.status == "LOST" for row in rows),
            "void": sum(row.status == "VOID" for row in rows),
            "win_rate": (sum(row.status == "WON" for row in active_settled) / len(active_settled)) if active_settled else None,
            "total_stake": total_stake,
            "total_return": total_return,
            "net_profit": net_profit,
            "realized_roi": net_profit / total_stake if total_stake else None,
            "average_combined_odd": sum(row.combined_odd for row in rows) / len(rows) if rows else None,
            "average_expected_roi": (
                sum(row.combined_expected_roi for row in rows if row.combined_expected_roi is not None)
                / sum(row.combined_expected_roi is not None for row in rows)
                if any(row.combined_expected_roi is not None for row in rows)
                else None
            ),
            "average_adjusted_probability": (
                sum(row.adjusted_probability for row in rows if row.adjusted_probability is not None)
                / sum(row.adjusted_probability is not None for row in rows)
                if any(row.adjusted_probability is not None for row in rows)
                else None
            ),
            "max_drawdown": drawdown,
            "bankroll_curve": curve,
            "by_profile": aggregate(lambda row: row.profile),
            "by_event_count": aggregate(lambda row: row.event_count),
            "by_period": aggregate(lambda row: row.reference_date[:7]),
            "by_market_combination": aggregate(
                lambda row: "+".join(sorted(pick.market for pick in row.picks))
            ),
        }
