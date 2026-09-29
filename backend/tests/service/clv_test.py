import unittest
from datetime import datetime, timedelta, timezone

from src.oracle.backtest.clv import (
    ClvReport,
    ClvResult,
    build_clv_result,
    compute_clv_odd_pct,
    compute_clv_prob,
    compute_clv_report,
    fetch_closing_market_baseline,
)

KICKOFF = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)


class _FakeSnapshot:
    def __init__(self, bookmaker, market, outcome, odd, captured_at, period="full_time", line=None):
        self.bookmaker = bookmaker
        self.market = market
        self.period = period
        self.line = line
        self.outcome = outcome
        self.odd = odd
        self.captured_at = captured_at

    def to_dict(self):
        return {
            "bookmaker": self.bookmaker,
            "market": self.market,
            "period": self.period,
            "line": self.line,
            "outcome": self.outcome,
            "odd": self.odd,
            "captured_at": self.captured_at.isoformat(),
        }


class _FakeRepo:
    def __init__(self, snapshots):
        self._snapshots = snapshots

    def list_for_fixture(self, fixture_id, market=None, period=None, line=None):
        rows = self._snapshots
        if market:
            rows = [s for s in rows if s.market == market]
        if period:
            rows = [s for s in rows if s.period == period]
        if line is not None:
            rows = [s for s in rows if s.line == line]
        return rows


class TestComputeClvOddPct(unittest.TestCase):
    def test_positive_when_bet_odd_higher_than_closing(self):
        # Ho preso 2.10, la chiusura è scesa a 1.90: la mia quota era migliore.
        self.assertAlmostEqual(compute_clv_odd_pct(2.10, 1.90), 2.10 / 1.90 - 1.0, places=9)
        self.assertGreater(compute_clv_odd_pct(2.10, 1.90), 0)

    def test_negative_when_bet_odd_lower_than_closing(self):
        self.assertLess(compute_clv_odd_pct(1.80, 2.00), 0)

    def test_none_when_odd_at_bet_missing(self):
        self.assertIsNone(compute_clv_odd_pct(None, 1.90))

    def test_none_when_closing_odd_missing(self):
        self.assertIsNone(compute_clv_odd_pct(2.10, None))

    def test_none_when_not_positive(self):
        self.assertIsNone(compute_clv_odd_pct(0.0, 1.9))
        self.assertIsNone(compute_clv_odd_pct(2.1, -1.0))

    def test_none_when_not_numeric(self):
        self.assertIsNone(compute_clv_odd_pct("not-a-number", 1.9))


class TestComputeClvProb(unittest.TestCase):
    def test_numeric_value(self):
        self.assertAlmostEqual(compute_clv_prob(0.55, 0.60), -0.05, places=9)

    def test_none_when_missing(self):
        self.assertIsNone(compute_clv_prob(None, 0.6))
        self.assertIsNone(compute_clv_prob(0.55, None))


class TestFetchClosingMarketBaseline(unittest.TestCase):
    def test_none_when_kickoff_unknown(self):
        repo = _FakeRepo([_FakeSnapshot("bookA", "h2h", "Home", 1.9, KICKOFF - timedelta(days=1))])
        result = fetch_closing_market_baseline(
            fixture_id=1, market="h2h", kickoff_at=None, snapshot_repo=repo
        )
        self.assertIsNone(result)

    def test_none_when_no_snapshots(self):
        repo = _FakeRepo([])
        result = fetch_closing_market_baseline(
            fixture_id=1, market="h2h", kickoff_at=KICKOFF, snapshot_repo=repo
        )
        self.assertIsNone(result)

    def test_none_when_only_post_kickoff_snapshots_exist(self):
        # Nessuno snapshot PRIMA del kickoff: non esiste una vera "chiusura".
        repo = _FakeRepo(
            [
                _FakeSnapshot("bookA", "h2h", "Home", 1.5, KICKOFF + timedelta(hours=1)),
            ]
        )
        result = fetch_closing_market_baseline(
            fixture_id=1, market="h2h", kickoff_at=KICKOFF, snapshot_repo=repo
        )
        self.assertIsNone(result)

    def test_uses_last_snapshot_before_kickoff_ignoring_later_ones(self):
        repo = _FakeRepo(
            [
                _FakeSnapshot("bookA", "h2h", "Home", 2.10, KICKOFF - timedelta(days=5)),
                _FakeSnapshot("bookA", "h2h", "Home", 1.90, KICKOFF - timedelta(minutes=5)),  # vera closing
                _FakeSnapshot("bookA", "h2h", "Home", 1.50, KICKOFF + timedelta(hours=1)),  # post-kickoff, ignorata
                _FakeSnapshot("bookA", "h2h", "Draw", 3.40, KICKOFF - timedelta(minutes=5)),
                _FakeSnapshot("bookA", "h2h", "Away", 4.20, KICKOFF - timedelta(minutes=5)),
            ]
        )
        baseline = fetch_closing_market_baseline(fixture_id=1, market="h2h", kickoff_at=KICKOFF, snapshot_repo=repo)
        self.assertIsNotNone(baseline)
        home_row = next(o for o in baseline["outcomes"] if o["outcome"] == "Home")
        self.assertAlmostEqual(home_row["avg_odd"], 1.90, places=9)

    def test_averages_across_bookmakers(self):
        repo = _FakeRepo(
            [
                _FakeSnapshot("bookA", "h2h", "Home", 2.00, KICKOFF - timedelta(minutes=5)),
                _FakeSnapshot("bookB", "h2h", "Home", 1.80, KICKOFF - timedelta(minutes=5)),
                _FakeSnapshot("bookA", "h2h", "Draw", 3.0, KICKOFF - timedelta(minutes=5)),
                _FakeSnapshot("bookA", "h2h", "Away", 4.0, KICKOFF - timedelta(minutes=5)),
            ]
        )
        baseline = fetch_closing_market_baseline(fixture_id=1, market="h2h", kickoff_at=KICKOFF, snapshot_repo=repo)
        home_row = next(o for o in baseline["outcomes"] if o["outcome"] == "Home")
        self.assertAlmostEqual(home_row["avg_odd"], 1.90, places=9)
        self.assertEqual(home_row["bookmakers"], 2)


class TestBuildClvResult(unittest.TestCase):
    def test_available_result_with_full_data(self):
        repo = _FakeRepo(
            [
                _FakeSnapshot("bookA", "h2h", "Home", 1.90, KICKOFF - timedelta(minutes=5)),
                _FakeSnapshot("bookA", "h2h", "Draw", 3.40, KICKOFF - timedelta(minutes=5)),
                _FakeSnapshot("bookA", "h2h", "Away", 4.20, KICKOFF - timedelta(minutes=5)),
            ]
        )
        result = build_clv_result(
            market="h2h", outcome="Home", odd_at_bet=2.10, p_fair_at_bet=0.50,
            fixture_id=1, kickoff_at=KICKOFF, model_name="direct_h2h", snapshot_repo=repo,
        )
        self.assertIsInstance(result, ClvResult)
        self.assertTrue(result.available)
        self.assertAlmostEqual(result.closing_odd, 1.90, places=9)
        self.assertAlmostEqual(result.clv_odd_pct, 2.10 / 1.90 - 1.0, places=9)
        self.assertIsNotNone(result.clv_prob)
        self.assertEqual(result.model_name, "direct_h2h")

    def test_unavailable_when_no_snapshots(self):
        repo = _FakeRepo([])
        result = build_clv_result(
            market="h2h", outcome="Home", odd_at_bet=2.10, p_fair_at_bet=0.50,
            fixture_id=1, kickoff_at=KICKOFF, snapshot_repo=repo,
        )
        self.assertFalse(result.available)
        self.assertIn("non disponibile", result.reason)
        self.assertIsNone(result.closing_odd)
        self.assertIsNone(result.clv_odd_pct)

    def test_unavailable_when_outcome_not_quoted_at_closing(self):
        repo = _FakeRepo(
            [
                _FakeSnapshot("bookA", "h2h", "Draw", 3.40, KICKOFF - timedelta(minutes=5)),
            ]
        )
        result = build_clv_result(
            market="h2h", outcome="Home", odd_at_bet=2.10, p_fair_at_bet=0.50,
            fixture_id=1, kickoff_at=KICKOFF, snapshot_repo=repo,
        )
        self.assertFalse(result.available)
        self.assertIn("non quotato", result.reason)

    def test_never_raises_when_kickoff_missing(self):
        repo = _FakeRepo([_FakeSnapshot("bookA", "h2h", "Home", 1.9, KICKOFF - timedelta(days=1))])
        result = build_clv_result(
            market="h2h", outcome="Home", odd_at_bet=2.10, p_fair_at_bet=0.50,
            fixture_id=1, kickoff_at=None, snapshot_repo=repo,
        )
        self.assertFalse(result.available)


class TestComputeClvReport(unittest.TestCase):
    def _result(self, market, model_name, clv_odd_pct, available=True):
        return ClvResult(
            market=market, outcome="Home", fixture_id=1, model_name=model_name,
            odd_at_bet=2.0, p_fair_at_bet=0.5, closing_odd=1.9, p_fair_closing=0.52,
            clv_odd_pct=clv_odd_pct, clv_prob=-0.02, available=available, reason="",
        )

    def test_overall_average_and_positive_rate(self):
        results = [
            self._result("h2h", "model_a", 0.10),
            self._result("h2h", "model_a", -0.05),
            self._result("corners", "model_b", 0.02),
        ]
        report = compute_clv_report(results)
        self.assertIsInstance(report, ClvReport)
        self.assertEqual(report.overall.count, 3)
        self.assertAlmostEqual(report.overall.avg_clv_odd_pct, (0.10 - 0.05 + 0.02) / 3, places=9)
        self.assertAlmostEqual(report.overall.positive_clv_rate, 2 / 3, places=9)

    def test_breakdown_by_market_and_model(self):
        results = [
            self._result("h2h", "model_a", 0.10),
            self._result("h2h", "model_a", -0.05),
            self._result("corners", "model_b", 0.02),
        ]
        report = compute_clv_report(results)
        self.assertEqual(report.by_market["h2h"].count, 2)
        self.assertEqual(report.by_market["corners"].count, 1)
        self.assertEqual(report.by_model["model_a"].count, 2)
        self.assertEqual(report.by_model["model_b"].count, 1)

    def test_unavailable_results_excluded_from_averages_but_counted(self):
        results = [
            self._result("h2h", "model_a", 0.10),
            ClvResult(
                market="h2h", outcome="Away", fixture_id=2, model_name="model_a",
                odd_at_bet=1.8, p_fair_at_bet=0.5, closing_odd=None, p_fair_closing=None,
                clv_odd_pct=None, clv_prob=None, available=False, reason="non disponibile",
            ),
        ]
        report = compute_clv_report(results)
        self.assertEqual(report.total_considered, 2)
        self.assertEqual(report.available_count, 1)
        self.assertEqual(report.unavailable_count, 1)
        self.assertEqual(report.overall.count, 1)

    def test_missing_model_name_grouped_as_unknown(self):
        results = [self._result("h2h", None, 0.05)]
        report = compute_clv_report(results)
        self.assertIn("unknown", report.by_model)

    def test_empty_input_returns_empty_report(self):
        report = compute_clv_report([])
        self.assertEqual(report.total_considered, 0)
        self.assertEqual(report.available_count, 0)
        self.assertIsNone(report.overall.avg_clv_odd_pct)


if __name__ == "__main__":
    unittest.main()

