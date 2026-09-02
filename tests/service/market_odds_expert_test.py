import unittest
from datetime import datetime, timedelta, timezone

from src.ml.experts.market.market_odds_expert import (
    MarketOddsExpert,
    build_opening_latest_closing,
    compute_bookmaker_dispersion,
    compute_fair_probabilities,
    compute_odds_movement,
)

KICKOFF = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)


def _snapshot(bookmaker, market, outcome, odd, captured_at, period="full_time", line=None):
    return {
        "fixture_id": 1001,
        "bookmaker": bookmaker,
        "market": market,
        "period": period,
        "line": line,
        "outcome": outcome,
        "odd": odd,
        "captured_at": captured_at.isoformat(),
        "source": "api_sports",
    }


class TestMarketOddsExpert(unittest.TestCase):
    def setUp(self):
        opening_time = KICKOFF - timedelta(days=5)
        mid_time = KICKOFF - timedelta(days=1)
        future_time = KICKOFF + timedelta(hours=1)  # NON deve mai essere usato se as_of < kickoff

        self.snapshots = [
            _snapshot("bookA", "h2h", "Home", 2.00, opening_time),
            _snapshot("bookA", "h2h", "Draw", 3.40, opening_time),
            _snapshot("bookA", "h2h", "Away", 4.20, opening_time),
            _snapshot("bookB", "h2h", "Home", 1.95, opening_time),
            _snapshot("bookB", "h2h", "Draw", 3.30, opening_time),
            _snapshot("bookB", "h2h", "Away", 4.00, opening_time),
            # Movimento verso Home: la quota scende (probabilita' implicita sale).
            _snapshot("bookA", "h2h", "Home", 1.70, mid_time),
            _snapshot("bookB", "h2h", "Home", 1.72, mid_time),
            # Snapshot "dal futuro" rispetto a un as_of pre-match: deve essere sempre escluso.
            _snapshot("bookA", "h2h", "Home", 1.50, future_time),
        ]

    def test_fair_probabilities_sum_to_one_and_ignore_future_snapshots(self):
        as_of = KICKOFF - timedelta(hours=2)
        result = compute_fair_probabilities(self.snapshots, market="h2h", as_of=as_of)

        self.assertTrue(result["is_exclusive"])
        self.assertAlmostEqual(result["sum_fair_probability"], 1.0, places=6)
        home_row = next(o for o in result["outcomes"] if o["outcome"] == "Home")
        # Deve riflettere l'ultima quota nota PRIMA del cutoff (1.70/1.72), non quella futura (1.50).
        self.assertGreater(home_row["avg_odd"], 1.6)

    def test_bookmaker_dispersion_reports_std_across_books(self):
        as_of = KICKOFF - timedelta(days=4)  # solo gli snapshot di apertura sono disponibili
        dispersion = compute_bookmaker_dispersion(self.snapshots, as_of=as_of)

        self.assertIn("Home", dispersion)
        self.assertEqual(dispersion["Home"]["bookmaker_count"], 2)
        self.assertGreaterEqual(dispersion["Home"]["std_implied_probability"], 0.0)

    def test_odds_movement_detects_drift_towards_home(self):
        as_of = KICKOFF - timedelta(hours=2)
        movement = compute_odds_movement(self.snapshots, as_of=as_of)

        self.assertIn("Home", movement)
        self.assertGreater(movement["Home"]["drift"], 0.0)  # probabilita' implicita salita nel tempo

    def test_closing_is_none_when_as_of_is_well_before_kickoff(self):
        as_of = KICKOFF - timedelta(hours=2)
        payload = build_opening_latest_closing(self.snapshots, as_of=as_of, kickoff_at=KICKOFF)

        self.assertTrue(len(payload) > 0)
        for row in payload:
            self.assertIsNone(row["closing"], msg="closing non deve mai essere popolato prima del kickoff")
            self.assertIsNotNone(row["latest"])

    def test_closing_is_available_at_or_after_kickoff(self):
        as_of = KICKOFF + timedelta(minutes=5)
        payload = build_opening_latest_closing(self.snapshots, as_of=as_of, kickoff_at=KICKOFF)

        home_book_a = next(r for r in payload if r["bookmaker"] == "bookA" and r["outcome"] == "Home")
        self.assertIsNotNone(home_book_a["closing"])
        self.assertEqual(home_book_a["closing"]["odd"], home_book_a["latest"]["odd"])

    def test_closing_never_reported_without_kickoff_info(self):
        as_of = KICKOFF + timedelta(days=1)  # anche molto "dopo", ma senza kickoff noto
        payload = build_opening_latest_closing(self.snapshots, as_of=as_of, kickoff_at=None)

        for row in payload:
            self.assertIsNone(row["closing"])

    def test_future_snapshot_never_leaks_into_opening_latest(self):
        as_of = KICKOFF - timedelta(hours=2)
        payload = build_opening_latest_closing(self.snapshots, as_of=as_of, kickoff_at=KICKOFF)

        home_book_a = next(r for r in payload if r["bookmaker"] == "bookA" and r["outcome"] == "Home")
        self.assertNotEqual(home_book_a["latest"]["odd"], 1.50)  # quota "futura", mai visibile

    def test_expert_version_is_deterministic(self):
        expert_a = MarketOddsExpert()
        expert_b = MarketOddsExpert()
        self.assertEqual(expert_a.VERSION, expert_b.VERSION)
        self.assertEqual(MarketOddsExpert.VERSION, expert_a.VERSION)


if __name__ == "__main__":
    unittest.main()

