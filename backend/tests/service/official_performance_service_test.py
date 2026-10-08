import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from src.oracle.ledger.official_performance_service import (
    OFFICIAL_COHORT,
    OFFICIAL_SOURCE,
    compute_official_performance,
)


def _row(decision, settlement_status, market="1x2", odd=2.0, is_settled=True, pnl=1.0, league=135):
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        source=OFFICIAL_SOURCE,
        cohort=OFFICIAL_COHORT,
        market=market,
        outcome="Home",
        kickoff_at=now,
        created_at=now,
        decision=decision,
        odd=odd,
        is_settled=is_settled,
        settlement_status=settlement_status,
        stake=1.0,
        pnl=pnl,
        prob_edge=0.1,
        ev=0.2,
        fixture_id=10,
        league=league,
        model_name="logistic",
        model_run_id="run-1",
        policy_version="policy-v1",
        period="full_time",
    )


class TestComputeOfficialPerformanceByDecision(unittest.TestCase):
    def test_overall_and_breakdowns_stay_play_only_for_backward_compatibility(self):
        """`overall`/`breakdowns` (le chiavi gia' usate da tutto il resto
        del progetto) devono restare SOLO-PLAY, identiche a prima
        dell'aggiunta di by_decision - nessun chiamante esistente deve
        accorgersi del cambiamento."""
        rows = [
            _row("PLAY", "settled_win", pnl=1.0),
            _row("BORDERLINE", "settled_win", pnl=1.0),
            _row("NO BET", "settled_loss", pnl=-1.0),
        ]
        metrics = compute_official_performance(rows)

        self.assertEqual(metrics["overall"]["plays"], 1)
        self.assertEqual(metrics["overall"]["wins"], 1)
        self.assertEqual(metrics["overall"]["losses"], 0)
        self.assertEqual(metrics["breakdowns"]["market"]["1x2"]["plays"], 1)

    def test_by_decision_play_matches_overall(self):
        rows = [_row("PLAY", "settled_win"), _row("BORDERLINE", "settled_loss")]
        metrics = compute_official_performance(rows)

        self.assertEqual(metrics["by_decision"]["PLAY"]["overall"], metrics["overall"])
        self.assertEqual(metrics["by_decision"]["PLAY"]["breakdowns"], metrics["breakdowns"])

    def test_by_decision_borderline_isolates_only_borderline_rows(self):
        rows = [
            _row("PLAY", "settled_win", pnl=1.0),
            _row("BORDERLINE", "settled_win", pnl=1.0),
            _row("BORDERLINE", "settled_loss", pnl=-1.0),
            _row("NO BET", "settled_win", pnl=1.0),
        ]
        metrics = compute_official_performance(rows)

        borderline = metrics["by_decision"]["BORDERLINE"]["overall"]
        self.assertEqual(borderline["plays"], 2)
        self.assertEqual(borderline["wins"], 1)
        self.assertEqual(borderline["losses"], 1)

    def test_by_decision_no_bet_isolates_only_no_bet_rows(self):
        rows = [
            _row("PLAY", "settled_win"),
            _row("NO BET", "settled_win", pnl=1.0),
            _row("NO BET", "settled_win", pnl=1.0),
        ]
        metrics = compute_official_performance(rows)

        no_bet = metrics["by_decision"]["NO BET"]["overall"]
        self.assertEqual(no_bet["plays"], 2)
        self.assertEqual(no_bet["wins"], 2)

    def test_by_decision_all_aggregates_every_decision(self):
        rows = [
            _row("PLAY", "settled_win"),
            _row("BORDERLINE", "settled_win"),
            _row("NO BET", "settled_loss"),
        ]
        metrics = compute_official_performance(rows)

        overall_all = metrics["by_decision"]["ALL"]["overall"]
        self.assertEqual(overall_all["plays"], 3)
        self.assertEqual(overall_all["wins"], 2)
        self.assertEqual(overall_all["losses"], 1)

    def test_by_decision_market_breakdown_isolated_per_decision(self):
        rows = [
            _row("PLAY", "settled_win", market="cards"),
            _row("BORDERLINE", "settled_win", market="cards"),
            _row("BORDERLINE", "settled_win", market="corners"),
        ]
        metrics = compute_official_performance(rows)

        borderline_markets = metrics["by_decision"]["BORDERLINE"]["breakdowns"]["market"]
        self.assertEqual(set(borderline_markets.keys()), {"cards", "corners"})
        self.assertEqual(borderline_markets["cards"]["plays"], 1)
        self.assertEqual(borderline_markets["corners"]["plays"], 1)

    def test_empty_rows_produce_zeroed_buckets_for_every_decision(self):
        metrics = compute_official_performance([])

        for label in ("PLAY", "BORDERLINE", "NO BET", "ALL"):
            self.assertEqual(metrics["by_decision"][label]["overall"]["plays"], 0)


if __name__ == "__main__":
    unittest.main()
