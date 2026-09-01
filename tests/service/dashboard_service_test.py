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


if __name__ == "__main__":
    unittest.main()

