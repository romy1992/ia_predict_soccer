import copy
import dataclasses
import random
import tempfile
import unittest
from pathlib import Path

from src.oracle.backtest.betting_backtester import (
    DEFAULT_EDGE_BUCKET_EDGES,
    BacktestBet,
    bet_profit,
    build_backtest_bet,
    compute_backtest_report,
    edge_bucket_label,
    max_drawdown_from_cumulative,
    persist_backtest_report,
)
from src.oracle.value_engine.value_engine import BORDERLINE, NO_BET, PLAY


def _bet(
    market="h2h",
    outcome="Home",
    odd=2.0,
    won=True,
    prob_edge=0.05,
    ev=0.10,
    decision=PLAY,
    league="A",
    fixture_id=1,
    kickoff_at="2026-01-01T00:00:00+00:00",
) -> BacktestBet:
    return BacktestBet(
        market=market,
        outcome=outcome,
        p_model=0.6,
        p_market_fair=0.55,
        odd=odd,
        prob_edge=prob_edge,
        ev=ev,
        decision=decision,
        policy_version="value_policy_v1",
        won=won,
        fixture_id=fixture_id,
        league=league,
        season=2025,
        kickoff_at=kickoff_at,
    )


class TestBuildBacktestBet(unittest.TestCase):
    def test_wires_edge_and_ev_from_value_engine(self):
        bet = build_backtest_bet(
            market="h2h", outcome="Home", p_model=0.65, p_market_fair=0.55, odd=2.0, won=True, fixture_id=42
        )
        self.assertAlmostEqual(bet.prob_edge, 0.10, places=9)
        self.assertAlmostEqual(bet.ev, 0.65 * 2.0 - 1.0, places=9)
        self.assertEqual(bet.fixture_id, 42)

    def test_missing_odd_yields_no_bet_and_no_ev(self):
        bet = build_backtest_bet(market="h2h", outcome="Home", p_model=0.9, p_market_fair=0.55, odd=None, won=None)
        self.assertEqual(bet.decision, NO_BET)
        self.assertIsNone(bet.ev)
        self.assertIsNotNone(bet.prob_edge)


class TestBetProfit(unittest.TestCase):
    def test_profit_when_won(self):
        self.assertAlmostEqual(bet_profit(_bet(odd=2.0, won=True), stake=1.0), 1.0, places=9)

    def test_loss_when_lost(self):
        self.assertAlmostEqual(bet_profit(_bet(odd=2.0, won=False), stake=1.0), -1.0, places=9)

    def test_scales_with_stake(self):
        self.assertAlmostEqual(bet_profit(_bet(odd=3.0, won=True), stake=10.0), 20.0, places=9)
        self.assertAlmostEqual(bet_profit(_bet(odd=3.0, won=False), stake=10.0), -10.0, places=9)

    def test_none_when_odd_missing(self):
        bet = _bet()
        bet.odd = None
        self.assertIsNone(bet_profit(bet))

    def test_none_when_won_unknown(self):
        bet = _bet()
        bet.won = None
        self.assertIsNone(bet_profit(bet))

    def test_none_when_odd_not_positive(self):
        bet = _bet()
        bet.odd = 0.0
        self.assertIsNone(bet_profit(bet))


class TestMaxDrawdownFromCumulative(unittest.TestCase):
    def test_zero_drawdown_on_monotonic_increase(self):
        self.assertEqual(max_drawdown_from_cumulative([1.0, 2.0, 3.0, 4.0]), 0.0)

    def test_known_sequence(self):
        # peak: 1,2,2,2,2,3 -> drawdown: 0,0,1,3,2,0 -> max = 3
        self.assertAlmostEqual(max_drawdown_from_cumulative([1.0, 2.0, 1.0, -1.0, 0.0, 3.0]), 3.0, places=9)

    def test_empty_curve_is_zero(self):
        self.assertEqual(max_drawdown_from_cumulative([]), 0.0)


class TestEdgeBucketLabel(unittest.TestCase):
    def test_none_is_unknown(self):
        self.assertEqual(edge_bucket_label(None), "unknown")

    def test_below_first_edge(self):
        self.assertEqual(edge_bucket_label(-0.05, DEFAULT_EDGE_BUCKET_EDGES), "<0.00")

    def test_middle_buckets(self):
        self.assertEqual(edge_bucket_label(0.01, DEFAULT_EDGE_BUCKET_EDGES), "[0.00,0.03)")
        self.assertEqual(edge_bucket_label(0.04, DEFAULT_EDGE_BUCKET_EDGES), "[0.03,0.06)")
        self.assertEqual(edge_bucket_label(0.08, DEFAULT_EDGE_BUCKET_EDGES), "[0.06,0.10)")

    def test_boundary_is_inclusive_on_lower_bound(self):
        self.assertEqual(edge_bucket_label(0.03, DEFAULT_EDGE_BUCKET_EDGES), "[0.03,0.06)")

    def test_above_last_edge(self):
        self.assertEqual(edge_bucket_label(0.15, DEFAULT_EDGE_BUCKET_EDGES), ">=0.10")


class TestComputeBacktestReport(unittest.TestCase):
    def _scenario(self) -> list[BacktestBet]:
        return [
            _bet(market="h2h", league="A", odd=2.0, won=True, prob_edge=0.10, fixture_id=1, kickoff_at="2026-01-01"),
            _bet(market="h2h", league="A", odd=3.0, won=False, prob_edge=0.02, fixture_id=2, kickoff_at="2026-01-02"),
            _bet(market="dc", league="B", odd=1.5, won=True, prob_edge=0.07, fixture_id=3, kickoff_at="2026-01-03"),
            # BORDERLINE: esclusa di default (include_decisions={PLAY}).
            _bet(market="h2h", league="A", odd=2.5, won=True, prob_edge=0.01, decision=BORDERLINE, fixture_id=4),
            # NO_BET: esclusa (quota/esito mancanti, come da Value Engine).
            _bet(market="h2h", league="A", odd=None, won=None, prob_edge=None, ev=None, decision=NO_BET, fixture_id=5),
        ]

    def test_overall_stats_are_numerically_correct(self):
        report = compute_backtest_report(self._scenario())

        self.assertEqual(report.total_bets_considered, 5)
        self.assertEqual(report.placed_bets, 3)
        self.assertEqual(report.skipped_bets, 2)

        overall = report.overall
        self.assertEqual(overall.bets, 3)
        self.assertEqual(overall.wins, 2)
        self.assertEqual(overall.losses, 1)
        self.assertAlmostEqual(overall.profit, 0.5, places=9)  # +1.0 -1.0 +0.5
        self.assertAlmostEqual(overall.total_staked, 3.0, places=9)
        self.assertAlmostEqual(overall.roi, 0.5 / 3.0, places=9)
        self.assertAlmostEqual(overall.yield_pct, (0.5 / 3.0) * 100.0, places=9)
        self.assertAlmostEqual(overall.hit_rate, 2.0 / 3.0, places=9)
        self.assertAlmostEqual(overall.avg_odds, (2.0 + 3.0 + 1.5) / 3.0, places=9)
        self.assertAlmostEqual(overall.max_drawdown, 1.0, places=9)  # cumulative 1.0,0.0,0.5

    def test_breakdown_by_market(self):
        report = compute_backtest_report(self._scenario())

        self.assertEqual(set(report.by_market.keys()), {"h2h", "dc"})
        h2h = report.by_market["h2h"]
        self.assertEqual(h2h.bets, 2)
        self.assertAlmostEqual(h2h.profit, 0.0, places=9)
        self.assertAlmostEqual(h2h.hit_rate, 0.5, places=9)

        dc = report.by_market["dc"]
        self.assertEqual(dc.bets, 1)
        self.assertAlmostEqual(dc.profit, 0.5, places=9)
        self.assertAlmostEqual(dc.hit_rate, 1.0, places=9)

    def test_breakdown_by_league(self):
        report = compute_backtest_report(self._scenario())

        self.assertEqual(set(report.by_league.keys()), {"A", "B"})
        self.assertEqual(report.by_league["A"].bets, 2)
        self.assertEqual(report.by_league["B"].bets, 1)

    def test_breakdown_by_edge_bucket(self):
        report = compute_backtest_report(self._scenario())

        self.assertEqual(report.by_edge_bucket["[0.00,0.03)"].bets, 1)
        self.assertAlmostEqual(report.by_edge_bucket["[0.00,0.03)"].profit, -1.0, places=9)
        self.assertEqual(report.by_edge_bucket["[0.06,0.10)"].bets, 1)
        self.assertAlmostEqual(report.by_edge_bucket["[0.06,0.10)"].profit, 0.5, places=9)
        self.assertEqual(report.by_edge_bucket[">=0.10"].bets, 1)
        self.assertAlmostEqual(report.by_edge_bucket[">=0.10"].profit, 1.0, places=9)

    def test_widening_include_decisions_adds_borderline_bet(self):
        report = compute_backtest_report(self._scenario(), include_decisions=frozenset({PLAY, BORDERLINE}))
        self.assertEqual(report.placed_bets, 4)
        self.assertEqual(report.overall.bets, 4)

    def test_report_is_reproducible_regardless_of_input_order(self):
        scenario = self._scenario()
        shuffled = copy.deepcopy(scenario)
        random.Random(7).shuffle(shuffled)

        report_a = compute_backtest_report(scenario)
        report_b = compute_backtest_report(shuffled)

        self.assertEqual(dataclasses.asdict(report_a), dataclasses.asdict(report_b))

    def test_custom_stake_scales_profit_and_staked(self):
        report = compute_backtest_report(self._scenario(), stake=10.0)
        self.assertAlmostEqual(report.overall.profit, 5.0, places=9)
        self.assertAlmostEqual(report.overall.total_staked, 30.0, places=9)
        # ROI invariante rispetto allo stake (a parita' di esiti/quote).
        self.assertAlmostEqual(report.overall.roi, 0.5 / 3.0, places=9)


class TestComputeBacktestReportEdgeCases(unittest.TestCase):
    def test_empty_bets_list(self):
        report = compute_backtest_report([])
        self.assertEqual(report.total_bets_considered, 0)
        self.assertEqual(report.placed_bets, 0)
        self.assertEqual(report.overall.bets, 0)
        self.assertIsNone(report.overall.roi)
        self.assertIsNone(report.overall.hit_rate)
        self.assertEqual(report.overall.max_drawdown, 0.0)

    def test_all_bets_excluded_when_no_decision_matches(self):
        bets = [_bet(decision=NO_BET, odd=None, won=None, prob_edge=None, ev=None)]
        report = compute_backtest_report(bets)
        self.assertEqual(report.placed_bets, 0)
        self.assertEqual(report.skipped_bets, 1)


class TestPersistBacktestReport(unittest.TestCase):
    def test_persists_valid_json_report(self):
        report = compute_backtest_report([_bet()])
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = persist_backtest_report(market="h2h", report=report, output_dir=tmp_dir)
            self.assertTrue(Path(path).exists())
            content = Path(path).read_text(encoding="utf-8")
            self.assertIn("overall", content)
            self.assertIn("by_market", content)


if __name__ == "__main__":
    unittest.main()
