import unittest
from datetime import date

from src.api.dashboard_service import DashboardService


class TestDashboardService(unittest.TestCase):
    def _fixture(self, fixture_id: int, day: str, status: str, home: str, away: str):
        return {
            "fixture": {
                "id": fixture_id,
                "date": f"{day}T18:45:00+00:00",
                "status": {"short": status},
            },
            "league": {"id": 135, "name": "Serie A", "round": "Regular Season - 1"},
            "teams": {"home": {"name": home}, "away": {"name": away}},
            "goals": {"home": 1, "away": 0},
        }

    def test_get_day_matches_from_api_feed(self):
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h"]
        service._fetch_api_day_fixtures = lambda target_date: [
            self._fixture(1001, "2026-09-01", "NS", "Inter", "Milan"),
            self._fixture(1002, "2026-09-01", "1H", "Roma", "Lazio"),
        ]
        service._fetch_matches = lambda: []
        service._predict_fixture = lambda fixture_id, markets: {
            "h2h": {
                "prediction": 1,
                "probability": 0.72,
                "model_name": "logistic",
                "run_id": "run-test",
            }
        }

        payload = service.get_day_matches(
            target_date=date(2026, 9, 1),
            limit=50,
            with_predictions=True,
        )

        self.assertEqual(payload.total, 2)
        self.assertEqual(payload.returned, 2)
        self.assertEqual(payload.rows[0]["source"], "api_sports")
        self.assertIn("h2h", payload.rows[0]["predictions"])

    def test_get_live_matches_filter(self):
        service = DashboardService()
        service.registry.list_markets = lambda: []
        service._fetch_api_live_fixtures = lambda: [
            self._fixture(2001, "2026-09-01", "1H", "Napoli", "Atalanta"),
            self._fixture(2002, "2026-09-01", "NS", "Juventus", "Bologna"),
        ]
        service._fetch_api_day_fixtures = lambda target_date: []
        service._fetch_matches = lambda: []

        payload = service.get_live_matches(target_date=date(2026, 9, 1), limit=50)

        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["returned"], 1)
        self.assertEqual(payload["rows"][0]["home"], "Napoli")

    def test_get_match_detail_cards_and_timeline(self):
        service = DashboardService()
        service.registry.list_markets = lambda: ["under_over_2_5", "goal_no_goal"]

        service._fetch_api_fixture_detail = lambda fixture_id: self._fixture(
            fixture_id=fixture_id,
            day="2026-09-01",
            status="1H",
            home="Inter",
            away="Roma",
        )
        service._fetch_db_match_by_fixture = lambda fixture_id: None
        service._fetch_api_events = lambda fixture_id: [
            {
                "time": {"elapsed": 22, "extra": None},
                "team": {"name": "Inter"},
                "type": "Goal",
                "detail": "Normal Goal",
                "player": {"name": "Lautaro"},
                "assist": {"name": "Barella"},
                "comments": None,
            }
        ]
        service._fetch_api_odds = lambda fixture_id: {
            "update": "2026-09-01T15:00:00+00:00",
            "bookmakers": [
                {
                    "name": "BookA",
                    "bets": [
                        {
                            "name": "Goals Over/Under",
                            "values": [
                                {"value": "Over 2.5", "odd": "1.90"},
                                {"value": "Under 2.5", "odd": "1.80"},
                            ],
                        },
                        {
                            "name": "Both Teams Score",
                            "values": [
                                {"value": "Yes", "odd": "1.75"},
                                {"value": "No", "odd": "2.00"},
                            ],
                        },
                    ],
                }
            ],
        }
        service._predict_fixture = lambda fixture_id, markets: {
            "under_over_2_5": {
                "prediction": 1,
                "probability": 0.72,
                "model_name": "logistic",
                "run_id": "uo-run",
            },
            "goal_no_goal": {
                "prediction": 0,
                "probability": 0.40,
                "model_name": "rf",
                "run_id": "gg-run",
            },
        }

        payload = service.get_match_detail(fixture_id=1234, with_predictions=True)

        self.assertIsNotNone(payload["fixture"])
        self.assertEqual(payload["fixture"]["home"], "Inter")
        self.assertEqual(len(payload["timeline"]), 1)
        self.assertEqual(payload["timeline"][0]["team"], "Inter")
        self.assertIn("under_over_2_5", payload["odds_summary"])
        self.assertGreaterEqual(len(payload["decision_cards"]), 2)
        labels = {c["value_label"] for c in payload["decision_cards"]}
        self.assertTrue(labels.issubset({"PLAY", "BORDERLINE", "NO BET"}))


if __name__ == "__main__":
    unittest.main()


