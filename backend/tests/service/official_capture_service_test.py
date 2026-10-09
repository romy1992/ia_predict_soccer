from __future__ import annotations

from datetime import datetime, timezone
from unittest import mock

from src.oracle.decision_engine.decision_policy import Decision
from src.oracle.ledger.official_capture_service import OfficialPredictionCaptureService, _best_decision
from src.service_ia.model.match import Match


def _decision(outcome: str, decision: str, expected_roi_percent=None, p_model=None) -> Decision:
    return Decision(
        market="1x2",
        outcome=outcome,
        p_model=p_model,
        p_market_fair=0.33,
        odd=2.0,
        samples=3,
        prob_edge=0.0,
        ev=0.0,
        decision=decision,
        reason="test",
        policy_version="test_v1",
        expected_roi_percent=expected_roi_percent,
    )


def test_best_decision_empty_returns_none():
    assert _best_decision([]) is None


def test_best_decision_play_beats_borderline_and_no_bet():
    play = _decision("Home", "PLAY", expected_roi_percent=1.0)
    borderline = _decision("Draw", "BORDERLINE", expected_roi_percent=50.0)
    no_bet = _decision("Away", "NO BET", p_model=0.9)
    assert _best_decision([borderline, no_bet, play]) is play


def test_best_decision_borderline_beats_no_bet():
    borderline = _decision("Draw", "BORDERLINE", expected_roi_percent=0.1)
    no_bet = _decision("Away", "NO BET", p_model=0.99)
    assert _best_decision([no_bet, borderline]) is borderline


def test_best_decision_tie_break_by_expected_roi_percent():
    low = _decision("Home", "PLAY", expected_roi_percent=1.0)
    high = _decision("Away", "PLAY", expected_roi_percent=5.0)
    assert _best_decision([low, high]) is high


def test_best_decision_no_bet_tie_break_by_p_model():
    low = _decision("Home", "NO BET", p_model=0.2)
    high = _decision("Away", "NO BET", p_model=0.4)
    assert _best_decision([low, high]) is high


def test_best_decision_none_metric_loses_tiebreak():
    missing = _decision("Home", "PLAY", expected_roi_percent=None)
    present = _decision("Away", "PLAY", expected_roi_percent=0.01)
    assert _best_decision([missing, present]) is present


class _FakeExpert:
    run_id = "run-fake"

    def __init__(self, probabilities):
        self._probabilities = probabilities

    def predict_proba_dict(self, _frame):
        return [self._probabilities]


def _match(fixture_id=12345):
    return Match(
        id_match_fk="m1",
        id_fixture=fixture_id,
        name_home="Home",
        name_away="Away",
        current_league=39,
        status="NS",
        date_match="2026-10-10T18:00:00+00:00",
    )


def test_capture_true_1x2_saves_only_best_decision_across_the_three_outcomes():
    """Home=BORDERLINE, Draw=NO BET, Away=PLAY: solo Away (PLAY) deve
    essere salvata, anche se la selezione precedente avrebbe scartato
    Home/Draw senza mai valutare se fossero la "migliore" disponibile."""
    service = OfficialPredictionCaptureService.__new__(OfficialPredictionCaptureService)
    service.registry = mock.Mock()
    match = _match()
    match_dict = {}
    odds_summary = {
        "h2h": [
            {"outcome": "Home", "avg_odd": 2.5, "bookmakers": 3, "period": "full_time", "line": None, "captured_at": datetime.now(timezone.utc)},
            {"outcome": "Draw", "avg_odd": 3.2, "bookmakers": 3, "period": "full_time", "line": None, "captured_at": datetime.now(timezone.utc)},
            {"outcome": "Away", "avg_odd": 4.0, "bookmakers": 3, "period": "full_time", "line": None, "captured_at": datetime.now(timezone.utc)},
        ]
    }
    captured_at = datetime(2026, 10, 10, 17, 0, tzinfo=timezone.utc)
    kickoff_at = datetime(2026, 10, 10, 18, 0, tzinfo=timezone.utc)

    decisions_by_outcome = {
        "Home": _decision("Home", "BORDERLINE", expected_roi_percent=2.0),
        "Draw": _decision("Draw", "NO BET", p_model=0.5),
        "Away": _decision("Away", "PLAY", expected_roi_percent=10.0),
    }

    saved_decisions = []

    def fake_save_decision(*, decision, **_kwargs):
        saved_decisions.append(decision)
        return mock.Mock(), True

    with (
        mock.patch(
            "src.oracle.ledger.official_capture_service.Market1x2Expert.load_production",
            return_value=_FakeExpert({"HOME": 0.4, "DRAW": 0.3, "AWAY": 0.3}),
        ),
        mock.patch(
            "src.oracle.ledger.official_capture_service.build_1x2_prediction_frame",
            return_value=mock.Mock(empty=False, drop=lambda columns, errors: "frame"),
        ),
        mock.patch(
            "src.oracle.ledger.official_capture_service.get_market_outcome_baseline",
            side_effect=lambda baseline, market, outcome: {"raw_probability": 0.33, "fair_probability": 0.33},
        ),
        mock.patch(
            "src.oracle.ledger.official_capture_service.evaluate_decision",
            side_effect=lambda market, outcome, **_kwargs: decisions_by_outcome[outcome],
        ),
        mock.patch.object(service, "_save_decision", side_effect=fake_save_decision),
    ):
        captured, duplicates, errors = service._capture_true_1x2(
            match, match_dict, odds_summary, captured_at, kickoff_at, cutoff_minutes=60
        )

    assert captured == 1
    assert duplicates == 0
    assert errors == []
    assert len(saved_decisions) == 1
    assert saved_decisions[0].outcome == "Away"
    assert saved_decisions[0].decision == "PLAY"
