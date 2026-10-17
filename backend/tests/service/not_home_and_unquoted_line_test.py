"""\"Non casa\" si prezza con la X2; le linee non quotate spiegano il motivo."""

from __future__ import annotations

import unittest

from src.api.dashboard_service import DashboardService
from src.ml.baselines.bookmaker_baseline import build_fixture_baseline

ODDS = {
    "h2h": [
        {"outcome": "Home", "avg_odd": 2.0, "bookmakers": 5},
        {"outcome": "Draw", "avg_odd": 3.4, "bookmakers": 5},
        {"outcome": "Away", "avg_odd": 4.0, "bookmakers": 5},
    ],
    "dc": [{"outcome": "Draw/Away", "avg_odd": 1.8, "bookmakers": 5}],
}


def cards(market, prediction, odds, probability=0.6):
    svc = DashboardService()
    baseline = build_fixture_baseline(odds)
    payload = {market: {"prediction": prediction, "probability": probability}}
    return svc._build_decision_cards({"home": "Casa"}, payload, odds, baseline)


class NotHomeTest(unittest.TestCase):
    def test_not_home_uses_x2_quote(self):
        pick, odd = DashboardService()._pick_and_odd_for_prediction("h2h", 0, {}, ODDS)
        self.assertEqual((pick, odd), ("Non casa", 1.8))
        card = cards("h2h", 0, ODDS, probability=0.4)[0]  # P(Home)=0.4 -> P(non casa)=0.6
        self.assertNotEqual(card["value_label"], "SENZA QUOTA")
        self.assertEqual(card["odd"], 1.8)

    def test_not_home_without_dc_stays_without_quote(self):
        card = cards("h2h", 0, {"h2h": ODDS["h2h"]}, probability=0.4)[0]
        self.assertEqual(card["value_label"], "SENZA QUOTA")


class UnquotedLineTest(unittest.TestCase):
    def test_reason_and_reliability(self):
        c65 = cards("cards_line_6_5", 1, {})[0]
        self.assertEqual(c65["value_label"], "SENZA QUOTA")
        self.assertTrue(c65["low_reliability"])
        self.assertIn("6.5", c65["value_reason"])
        c45 = cards("cards_line_4_5", 1, {})[0]
        self.assertFalse(c45["low_reliability"])
        self.assertIn("non quotata", c45["value_reason"])


if __name__ == "__main__":
    unittest.main()
