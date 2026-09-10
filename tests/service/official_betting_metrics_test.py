from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

from src.oracle.decision_engine.decision_policy import evaluate_decision
from src.oracle.ledger.ledger_service import PredictionLedgerService
from src.oracle.ledger.official_performance_service import (
    OFFICIAL_COHORT,
    OFFICIAL_SOURCE,
    compute_official_performance,
)
from src.oracle.ledger.prediction_ledger import (
    PredictionRecord,
    settle_prediction_record,
)
from src.oracle.ledger.settlement_rules import outcome_wins, resolve_settlement


def _row(**overrides):
    base = {
        "source": OFFICIAL_SOURCE,
        "cohort": OFFICIAL_COHORT,
        "decision": "PLAY",
        "odd": 2.0,
        "stake": 1.0,
        "is_settled": True,
        "settlement_status": "settled_win",
        "won": True,
        "pnl": 1.0,
        "prob_edge": 0.1,
        "ev": 0.2,
        "fixture_id": 1,
        "kickoff_at": datetime(2026, 9, 10, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return base


def test_1x2_draw_is_never_away():
    result = resolve_settlement(
        market="1x2", match_status="FT", home_score=1, away_score=1
    )
    assert result.actual_outcome == "Draw"
    assert outcome_wins("1x2", "Away", result.actual_outcome) is False
    assert outcome_wins("1x2", "Draw", result.actual_outcome) is True


def test_draw_wins_both_1x_and_x2_independently():
    result = resolve_settlement(
        market="dc", match_status="FT", home_score=2, away_score=2
    )
    assert outcome_wins("dc", "1X", result.actual_outcome) is True
    assert outcome_wins("dc", "X2", result.actual_outcome) is True
    assert outcome_wins("dc", "12", result.actual_outcome) is False


def test_integer_line_push_and_half_line_never_push():
    integer = resolve_settlement(
        market="under_over_2_5",
        match_status="FT",
        home_score=1,
        away_score=1,
        line="2.0",
    )
    half = resolve_settlement(
        market="under_over_2_5",
        match_status="FT",
        home_score=1,
        away_score=1,
        line="2.5",
    )
    assert integer.is_push is True
    assert half.is_push is False
    assert half.actual_outcome == "Under 2.5"


def test_cancelled_and_abandoned_are_explicit_voids():
    cancelled = resolve_settlement(
        market="1x2", match_status="CANC", home_score=None, away_score=None
    )
    abandoned = resolve_settlement(
        market="1x2", match_status="ABD", home_score=1, away_score=0
    )
    assert cancelled.void_status == "void_cancelled"
    assert abandoned.void_status == "void_abandoned"


def test_void_refunds_stake_and_does_not_set_won():
    record = PredictionRecord(
        fixture_id=1,
        market="1x2",
        outcome="Home",
        decision="PLAY",
        odd=2.0,
        stake=1.0,
    )
    settled = settle_prediction_record(
        record, actual_outcome=None, void_status="void_cancelled"
    )
    assert settled.pnl == 0.0
    assert settled.won is None
    assert settled.settlement_status == "void_cancelled"


def test_roi_uses_active_stake_and_void_not_hit_rate():
    report = compute_official_performance(
        [
            _row(),
            _row(
                fixture_id=2,
                settlement_status="settled_loss",
                won=False,
                pnl=-1.0,
            ),
            _row(
                fixture_id=3,
                settlement_status="void_cancelled",
                won=None,
                pnl=0.0,
            ),
        ]
    )["overall"]
    assert report["gross_stake"] == 3.0
    assert report["void_stake"] == 1.0
    assert report["active_stake"] == 2.0
    assert report["total_profit"] == 0.0
    assert report["roi"] == 0.0
    assert report["gross_roi"] == 0.0
    assert report["hit_rate"] == 0.5


def test_only_voids_have_null_roi_and_hit_rate():
    report = compute_official_performance(
        [_row(settlement_status="void_cancelled", won=None, pnl=0.0)]
    )["overall"]
    assert report["active_stake"] == 0.0
    assert report["roi"] is None
    assert report["hit_rate"] is None


class _Repo:
    def find_by_capture_key(self, _key):
        return None

    def find_existing(self, **_kwargs):
        return None

    def save(self, row):
        return row


def test_official_play_cannot_be_registered_after_kickoff():
    service = PredictionLedgerService(repo=_Repo())
    decision = evaluate_decision(
        market="under_over_2_5",
        outcome="Over 2.5",
        p_model=0.75,
        p_market_fair=0.5,
        odd=2.0,
        samples=5,
    )
    kickoff = datetime.now(timezone.utc) - timedelta(minutes=1)
    with pytest.raises(ValueError, match="kickoff"):
        service.log_prediction(
            fixture_id=1,
            decision=decision,
            kickoff_at=kickoff,
            source=OFFICIAL_SOURCE,
            cohort=OFFICIAL_COHORT,
        )


def test_missing_odd_is_not_a_void():
    record = PredictionRecord(
        fixture_id=1,
        market="1x2",
        outcome="Home",
        decision="NO BET",
        odd=None,
        stake=1.0,
    )
    settled = settle_prediction_record(record, actual_outcome="Home")
    assert settled.settlement_status == "not_placed_missing_odd"
    assert not settled.settlement_status.startswith("void_")
    assert settled.pnl == 0.0
