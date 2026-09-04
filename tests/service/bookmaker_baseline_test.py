import os
import tempfile
import unittest

from src.ml.baselines.bookmaker_baseline import (
    build_fixture_baseline,
    compute_market_baseline,
    get_market_outcome_baseline,
    persist_fixture_baseline,
)


class TestBookmakerBaseline(unittest.TestCase):
    def test_exclusive_market_fair_probabilities_sum_to_one(self):
        rows = [
            {"outcome": "Home", "avg_odd": 2.0, "bookmakers": 8},
            {"outcome": "Draw", "avg_odd": 3.5, "bookmakers": 8},
            {"outcome": "Away", "avg_odd": 4.0, "bookmakers": 8},
        ]

        payload = compute_market_baseline(market="h2h", odds_rows=rows)

        self.assertTrue(payload["is_exclusive"])
        self.assertAlmostEqual(payload["sum_fair_probability"], 1.0, places=6)
        self.assertGreater(payload["sum_implied_raw"], 1.0)

    def test_fixture_baseline_and_persist(self):
        baseline = build_fixture_baseline(
            {
                "under_over_2_5": [
                    {"outcome": "Over 2.5", "avg_odd": 1.90, "bookmakers": 4},
                    {"outcome": "Under 2.5", "avg_odd": 1.95, "bookmakers": 4},
                ],
                "goal_no_goal": [
                    {"outcome": "Yes", "avg_odd": 1.80, "bookmakers": 4},
                    {"outcome": "No", "avg_odd": 2.00, "bookmakers": 4},
                ],
            }
        )

        row = get_market_outcome_baseline(baseline, market="under_over_2_5", outcome="Over 2.5")
        self.assertIsNotNone(row)
        self.assertIn("fair_probability", row)

        with tempfile.TemporaryDirectory() as tmp:
            path = persist_fixture_baseline(fixture_id=1234, fixture_baseline=baseline, output_dir=tmp)
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
