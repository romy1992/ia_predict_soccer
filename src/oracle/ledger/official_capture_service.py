"""Cattura server-side, idempotente e pre-kickoff delle sole decisioni PLAY."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy.orm import selectinload

from src.api.dashboard_service import DashboardService
from src.ml.baselines.bookmaker_baseline import build_fixture_baseline, get_market_outcome_baseline
from src.ml.markets.market_1x2 import (
    Market1x2Expert,
    build_1x2_prediction_frame,
)
from src.ml.markets.market_double_chance import derive_double_chance_from_dict
from src.ml.serving.prediction_snapshot_service import PredictionSnapshotService
from src.oracle.decision_engine.decision_policy import PLAY, evaluate_decision
from src.oracle.ledger.ledger_service import PredictionLedgerService
from src.oracle.ledger.official_performance_service import OFFICIAL_COHORT, OFFICIAL_SOURCE
from src.repository.base.repository_db import SessionLocal
from src.service_ia.model.match import Match
from src.service_ia.training.model_registry import ModelRegistry
from src.service_ia.utility.utils import convert_orm_match_to_dict


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _kickoff(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return _aware(value)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return _aware(parsed)


class OfficialPredictionCaptureService:
    def __init__(
        self,
        ledger_service: Optional[PredictionLedgerService] = None,
        snapshot_service: Optional[PredictionSnapshotService] = None,
        dashboard_service: Optional[DashboardService] = None,
        registry: Optional[ModelRegistry] = None,
    ):
        self.ledger_service = ledger_service or PredictionLedgerService()
        self.snapshot_service = snapshot_service or PredictionSnapshotService()
        self.dashboard = dashboard_service or DashboardService()
        self.registry = registry or ModelRegistry()

    def _eligible_matches(self, now: datetime, cutoff_minutes: int) -> list[Match]:
        # Il confronto esatto avviene in Python perché ``date_match`` è una
        # colonna storica stringa. La query resta circoscritta alle NS.
        with SessionLocal() as session:
            rows = (
                session.query(Match)
                .options(
                    selectinload(Match.statistics),
                    selectinload(Match.odds),
                    selectinload(Match.odds_snapshots),
                )
                .filter(Match.status == "NS")
                .all()
            )
            upper = now + timedelta(minutes=cutoff_minutes)
            return [row for row in rows if (kickoff := _kickoff(row.date_match)) and now < kickoff <= upper]

    def _odds_summary(self, match: Match, captured_at: datetime) -> dict[str, list[dict[str, Any]]]:
        latest: dict[tuple[str, str, str, Optional[str], str], Any] = {}
        for row in match.odds_snapshots or []:
            row_time = _aware(row.captured_at)
            if row_time > captured_at:
                continue
            outcome = self.dashboard._normalize_outcome_value(row.outcome)
            key = (row.bookmaker, row.market, row.period, row.line, outcome)
            if key not in latest or _aware(latest[key].captured_at) < row_time:
                latest[key] = row

        grouped: dict[tuple[str, str, Optional[str], str], list[Any]] = {}
        for (_bookmaker, market, period, line, outcome), row in latest.items():
            grouped.setdefault((market, period, line, outcome), []).append(row)

        summary: dict[str, list[dict[str, Any]]] = {}
        for (market, period, line, outcome), rows in grouped.items():
            summary.setdefault(market, []).append(
                {
                    "outcome": outcome,
                    "avg_odd": sum(float(row.odd) for row in rows) / len(rows),
                    "bookmakers": len(rows),
                    "period": period,
                    "line": line,
                    "captured_at": max(_aware(row.captured_at) for row in rows),
                }
            )
        return summary

    @staticmethod
    def _row_for_outcome(rows: list[dict[str, Any]], outcome: str) -> Optional[dict[str, Any]]:
        wanted = DashboardService._normalize_text(outcome)
        return next(
            (row for row in rows if DashboardService._normalize_text(str(row.get("outcome") or "")) == wanted),
            None,
        )

    def _save_decision(
        self,
        *,
        match: Match,
        kickoff_at: datetime,
        captured_at: datetime,
        decision: Any,
        model_run_id: Optional[str],
        model_name: Optional[str],
        odds_row: dict[str, Any],
        p_market_raw: Optional[float],
        cutoff_minutes: int,
    ) -> tuple[Any, bool]:
        capture_key = (
            f"{match.id_fixture}:{decision.market}:{decision.outcome}:"
            f"{kickoff_at.isoformat()}:{cutoff_minutes}"
        )
        existing = self.ledger_service.repo.find_by_capture_key(capture_key)
        if existing is not None:
            return existing, False
        row = self.ledger_service.log_prediction(
            fixture_id=int(match.id_fixture),
            decision=decision,
            model_run_id=model_run_id,
            model_name=model_name,
            kickoff_at=kickoff_at,
            stake=1.0,
            p_market_raw=p_market_raw,
            period=str(odds_row.get("period") or "full_time"),
            line=odds_row.get("line"),
            source=OFFICIAL_SOURCE,
            cohort=OFFICIAL_COHORT,
            captured_at=captured_at,
            odds_captured_at=odds_row.get("captured_at"),
            bookmaker_count=int(odds_row.get("bookmakers") or 0),
            league=match.current_league,
            capture_key=capture_key,
        )
        return row, True

    def _capture_true_1x2(
        self,
        match: Match,
        match_dict: dict[str, Any],
        odds_summary: dict[str, list[dict[str, Any]]],
        captured_at: datetime,
        kickoff_at: datetime,
        cutoff_minutes: int,
    ) -> tuple[int, int, list[str]]:
        try:
            expert = Market1x2Expert.load_production(registry=self.registry)
        except (LookupError, FileNotFoundError):
            return 0, 0, ["production_1x2_unavailable"]
        frame = build_1x2_prediction_frame(match_dict)
        if frame is None or frame.empty:
            return 0, 0, ["calibrated_1x2_probability_unavailable"]
        X = frame.drop(columns=["market", "id_fixture", "season", "league", "prediction_at"], errors="ignore")
        probabilities = expert.predict_proba_dict(X)[0]
        if abs(sum(probabilities.values()) - 1.0) > 1e-6:
            return 0, 0, ["incoherent_1x2_probabilities"]

        h2h_rows = odds_summary.get("h2h") or []
        baseline = build_fixture_baseline({"h2h": h2h_rows})
        captured = duplicates = 0
        errors: list[str] = []
        for model_outcome, outcome in (("HOME", "Home"), ("DRAW", "Draw"), ("AWAY", "Away")):
            odds_row = self._row_for_outcome(h2h_rows, outcome)
            baseline_row = get_market_outcome_baseline(baseline, "h2h", outcome)
            if odds_row is None or baseline_row is None:
                errors.append(f"skipped_missing_odd:1x2:{outcome}")
                continue
            decision = evaluate_decision(
                market="1x2",
                outcome=outcome,
                p_model=probabilities[model_outcome],
                p_market_fair=baseline_row.get("fair_probability"),
                odd=odds_row.get("avg_odd"),
                samples=int(odds_row.get("bookmakers") or 0),
            )
            if decision.decision != PLAY:
                continue
            _, created = self._save_decision(
                match=match,
                kickoff_at=kickoff_at,
                captured_at=captured_at,
                decision=decision,
                model_run_id=expert.run_id,
                model_name="market_1x2",
                odds_row=odds_row,
                p_market_raw=baseline_row.get("raw_probability"),
                cutoff_minutes=cutoff_minutes,
            )
            captured += int(created)
            duplicates += int(not created)

        dc_rows = odds_summary.get("dc") or []
        if dc_rows:
            dc_probabilities = derive_double_chance_from_dict(probabilities)
            fair = {
                "Home/Draw": sum(
                    (get_market_outcome_baseline(baseline, "h2h", item) or {}).get("fair_probability", 0.0)
                    for item in ("Home", "Draw")
                ),
                "Home/Away": sum(
                    (get_market_outcome_baseline(baseline, "h2h", item) or {}).get("fair_probability", 0.0)
                    for item in ("Home", "Away")
                ),
                "Draw/Away": sum(
                    (get_market_outcome_baseline(baseline, "h2h", item) or {}).get("fair_probability", 0.0)
                    for item in ("Draw", "Away")
                ),
            }
            for outcome, p_model in dc_probabilities.items():
                odds_row = self._row_for_outcome(dc_rows, outcome)
                if odds_row is None:
                    errors.append(f"skipped_missing_odd:dc:{outcome}")
                    continue
                decision = evaluate_decision(
                    market="dc",
                    outcome=outcome,
                    p_model=p_model,
                    p_market_fair=fair[outcome],
                    odd=odds_row.get("avg_odd"),
                    samples=int(odds_row.get("bookmakers") or 0),
                )
                if decision.decision != PLAY:
                    continue
                _, created = self._save_decision(
                    match=match,
                    kickoff_at=kickoff_at,
                    captured_at=captured_at,
                    decision=decision,
                    model_run_id=expert.run_id,
                    model_name="market_1x2_derived_dc",
                    odds_row=odds_row,
                    p_market_raw=(1.0 / float(odds_row["avg_odd"])) if odds_row.get("avg_odd") else None,
                    cutoff_minutes=cutoff_minutes,
                )
                captured += int(created)
                duplicates += int(not created)
        return captured, duplicates, errors

    def capture(
        self,
        *,
        now: Optional[datetime] = None,
        cutoff_minutes: int = 60,
        matches: Optional[list[Match]] = None,
    ) -> dict[str, Any]:
        captured_at = _aware(now or datetime.now(timezone.utc))
        candidates = matches if matches is not None else self._eligible_matches(captured_at, cutoff_minutes)
        report: dict[str, Any] = {
            "source": OFFICIAL_SOURCE,
            "cohort": OFFICIAL_COHORT,
            "captured_at": captured_at.isoformat(),
            "cutoff_minutes": cutoff_minutes,
            "fixtures_considered": 0,
            "plays_created": 0,
            "duplicates": 0,
            "skipped_missing_odd": 0,
            "errors": [],
        }
        standard_markets = [
            market
            for market in self.registry.list_markets()
            if market not in {"h2h", "dc", "1x2"}
        ]
        for match in candidates:
            kickoff_at = _kickoff(match.date_match)
            if kickoff_at is None or not captured_at < kickoff_at <= captured_at + timedelta(minutes=cutoff_minutes):
                continue
            report["fixtures_considered"] += 1
            try:
                odds_summary = self._odds_summary(match, captured_at)
                match_dict = convert_orm_match_to_dict([match])[0]
                created, duplicates, messages = self._capture_true_1x2(
                    match, match_dict, odds_summary, captured_at, kickoff_at, cutoff_minutes
                )
                report["plays_created"] += created
                report["duplicates"] += duplicates
                report["skipped_missing_odd"] += sum(message.startswith("skipped_missing_odd") for message in messages)

                predictions = self.snapshot_service.resolve_predictions(
                    fixture_id=int(match.id_fixture),
                    markets=standard_markets,
                    db_match=match,
                    status=match.status,
                )
                baseline = build_fixture_baseline(odds_summary)
                cards = self.dashboard._build_decision_cards(
                    row_context={"home": match.name_home, "away": match.name_away},
                    predictions=predictions,
                    odds_summary=odds_summary,
                    bookmaker_baseline=baseline,
                )
                for card in cards:
                    if card.get("value_label") != PLAY:
                        continue
                    outcome = card.get("outcome") or card.get("pick")
                    odds_row = self._row_for_outcome(odds_summary.get(card["market"]) or [], str(outcome))
                    if odds_row is None or card.get("odd") is None:
                        report["skipped_missing_odd"] += 1
                        continue
                    decision = evaluate_decision(
                        market=card["market"],
                        outcome=str(outcome),
                        p_model=card.get("predicted_probability"),
                        p_market_fair=card.get("bookmaker_fair_probability"),
                        odd=card.get("odd"),
                        samples=int(card.get("bookmakers_count") or 0),
                    )
                    _, was_created = self._save_decision(
                        match=match,
                        kickoff_at=kickoff_at,
                        captured_at=captured_at,
                        decision=decision,
                        model_run_id=card.get("run_id"),
                        model_name=card.get("model_name"),
                        odds_row=odds_row,
                        p_market_raw=card.get("bookmaker_implied_raw"),
                        cutoff_minutes=cutoff_minutes,
                    )
                    report["plays_created"] += int(was_created)
                    report["duplicates"] += int(not was_created)
            except Exception as exc:
                report["errors"].append({"fixture_id": match.id_fixture, "message": str(exc)})
        return report
