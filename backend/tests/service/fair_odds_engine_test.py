import unittest

from src.oracle.fair_odds.fair_odds_engine import (
    FairOddsOutcome,
    build_fair_odds_for_fixture,
    build_fair_odds_for_market,
    build_fair_odds_outcome,
    fair_odd_from_probability,
)


class TestFairOddFromProbability(unittest.TestCase):
    def test_fair_odd_is_reciprocal_of_probability(self):
        self.assertAlmostEqual(fair_odd_from_probability(0.5), 2.0)
        self.assertAlmostEqual(fair_odd_from_probability(0.25), 4.0)

    def test_none_when_probability_missing(self):
        self.assertIsNone(fair_odd_from_probability(None))

    def test_none_when_probability_not_positive(self):
        self.assertIsNone(fair_odd_from_probability(0.0))
        self.assertIsNone(fair_odd_from_probability(-0.1))

    def test_none_when_probability_not_numeric(self):
        self.assertIsNone(fair_odd_from_probability("not-a-number"))


class TestBuildFairOddsOutcome(unittest.TestCase):
    def test_standard_output_fields_from_market_baseline_row(self):
        row = {"outcome": "Home", "avg_odd": 2.0, "implied_raw": 0.5, "fair_probability": 0.55, "bookmakers": 3}
        result = build_fair_odds_outcome(market="h2h", outcome="Home", market_baseline_row=row, p_model=0.6)

        self.assertIsInstance(result, FairOddsOutcome)
        self.assertEqual(result.market, "h2h")
        self.assertEqual(result.outcome, "Home")
        self.assertEqual(result.odd, 2.0)
        self.assertEqual(result.p_market_raw, 0.5)
        self.assertEqual(result.p_market_fair, 0.55)
        self.assertAlmostEqual(result.fair_odd, 1.0 / 0.55)
        self.assertEqual(result.p_model, 0.6)
        self.assertEqual(result.bookmakers, 3)

    def test_none_market_row_still_returns_standard_shape(self):
        result = build_fair_odds_outcome(market="h2h", outcome="Home", market_baseline_row=None, p_model=0.6)

        self.assertIsNone(result.odd)
        self.assertIsNone(result.p_market_raw)
        self.assertIsNone(result.p_market_fair)
        self.assertIsNone(result.fair_odd)
        self.assertEqual(result.p_model, 0.6)
        self.assertEqual(result.bookmakers, 0)


class TestBuildFairOddsForMarket(unittest.TestCase):
    def _h2h_odds_rows(self):
        return [
            {"outcome": "Home", "avg_odd": 2.0, "bookmakers": 3},
            {"outcome": "Draw", "avg_odd": 3.4, "bookmakers": 3},
            {"outcome": "Away", "avg_odd": 4.0, "bookmakers": 3},
        ]

    def test_overround_is_removed_from_fair_probability(self):
        # implied raw: 0.5 + 0.294 + 0.25 = 1.044 -> overround ~4.4%
        outcomes = build_fair_odds_for_market(market="h2h", odds_rows=self._h2h_odds_rows())
        total_fair = sum(o.p_market_fair for o in outcomes)
        self.assertAlmostEqual(total_fair, 1.0, places=6)
        # p_market_fair per Home deve essere < p_market_raw (overround rimosso).
        home = next(o for o in outcomes if o.outcome == "Home")
        self.assertLess(home.p_market_fair, home.p_market_raw)

    def test_fair_odd_matches_1_over_p_fair(self):
        outcomes = build_fair_odds_for_market(market="h2h", odds_rows=self._h2h_odds_rows())
        for item in outcomes:
            self.assertAlmostEqual(item.fair_odd, 1.0 / item.p_market_fair, places=9)

    def test_oracle_vs_market_comparison_attaches_p_model_per_outcome(self):
        outcomes = build_fair_odds_for_market(
            market="h2h", odds_rows=self._h2h_odds_rows(), p_model_by_outcome={"Home": 0.58, "Draw": 0.22}
        )
        by_outcome = {o.outcome: o for o in outcomes}
        self.assertAlmostEqual(by_outcome["Home"].p_model, 0.58)
        self.assertAlmostEqual(by_outcome["Draw"].p_model, 0.22)
        self.assertIsNone(by_outcome["Away"].p_model)

    def test_p_model_outcome_without_market_quote_is_still_included(self):
        outcomes = build_fair_odds_for_market(
            market="h2h", odds_rows=self._h2h_odds_rows(), p_model_by_outcome={"Home": 0.58, "Push": 0.02}
        )
        push = next(o for o in outcomes if o.outcome == "Push")
        self.assertAlmostEqual(push.p_model, 0.02)
        self.assertIsNone(push.p_market_fair)
        self.assertIsNone(push.fair_odd)

    def test_empty_odds_rows_returns_empty_list_without_error(self):
        self.assertEqual(build_fair_odds_for_market(market="h2h", odds_rows=[]), [])


class TestBuildFairOddsForFixture(unittest.TestCase):
    def test_builds_per_market_outcomes_from_odds_summary(self):
        odds_summary = {
            "h2h": [
                {"outcome": "Home", "avg_odd": 2.0, "bookmakers": 3},
                {"outcome": "Draw", "avg_odd": 3.4, "bookmakers": 3},
                {"outcome": "Away", "avg_odd": 4.0, "bookmakers": 3},
            ],
            "goal_no_goal": [
                {"outcome": "Yes", "avg_odd": 1.9, "bookmakers": 2},
                {"outcome": "No", "avg_odd": 1.9, "bookmakers": 2},
            ],
        }
        p_model_by_market_outcome = {"h2h": {"Home": 0.58}, "goal_no_goal": {"Yes": 0.52}}

        result = build_fair_odds_for_fixture(odds_summary, p_model_by_market_outcome)

        self.assertEqual(set(result.keys()), {"h2h", "goal_no_goal"})
        self.assertEqual(len(result["h2h"]), 3)
        self.assertEqual(len(result["goal_no_goal"]), 2)
        home = next(o for o in result["h2h"] if o.outcome == "Home")
        self.assertAlmostEqual(home.p_model, 0.58)

    def test_market_with_only_oracle_and_no_quotes_is_included(self):
        result = build_fair_odds_for_fixture(odds_summary={}, p_model_by_market_outcome={"corners": {"Over 9.5": 0.6}})
        self.assertIn("corners", result)
        self.assertEqual(len(result["corners"]), 1)
        self.assertAlmostEqual(result["corners"][0].p_model, 0.6)
        self.assertIsNone(result["corners"][0].p_market_fair)

    def test_empty_inputs_return_empty_dict(self):
        self.assertEqual(build_fair_odds_for_fixture(odds_summary={}, p_model_by_market_outcome={}), {})


if __name__ == "__main__":
    unittest.main()
