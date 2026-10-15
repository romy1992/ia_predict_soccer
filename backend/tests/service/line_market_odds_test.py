"""Le linee Cards/Corners prendono la quota della linea esatta dal mercato base."""

from __future__ import annotations

import unittest

from src.api.dashboard_service import DashboardService

ODDS = {"cards": [
    {"outcome": "Over 5.5", "avg_odd": 2.4}, {"outcome": "Under 5.5", "avg_odd": 1.55},
    {"outcome": "Over 6.5", "avg_odd": 3.6},
]}


class LineMarketOddsTest(unittest.TestCase):
    def pick(self, market, prediction, odds=ODDS):
        return DashboardService()._pick_and_odd_for_prediction(market, prediction, {}, odds)

    def test_exact_line_is_priced(self):
        self.assertEqual(self.pick("cards_line_5_5", 0), ("Under 5.5", 1.55))
        self.assertEqual(self.pick("cards_line_6_5", 1), ("Over 6.5", 3.6))

    def test_missing_exact_outcome_stays_without_quote(self):
        # nessuna quota "Under 6.5": non si usa quella di un'altra linea
        self.assertEqual(self.pick("cards_line_6_5", 0), ("Under 6.5", None))
        self.assertEqual(self.pick("cards_line_4_5", 1), ("Over 4.5", None))

    def test_no_odds_at_all(self):
        self.assertEqual(self.pick("cards_line_6_5", 1, {}), ("Over 6.5", None))

    def test_base_and_goal_markets_unchanged(self):
        goals = {"under_over_3_5": [{"outcome": "Under 3.5", "avg_odd": 1.4}]}
        self.assertEqual(self.pick("under_over_3_5", 0, goals), ("Under 3.5", 1.4))
        self.assertEqual(self.pick("cards", 1)[0].startswith("Over"), True)
