import unittest
import uuid
from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.api.dashboard_service as dashboard_service_module
from src.api.dashboard_service import DashboardService
from src.service_ia.model.match import Base, Match


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
        service._fetch_matches = lambda target_date=None, day_margin=1: []
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
            self._fixture(2001, "2099-09-01", "1H", "Napoli", "Atalanta"),
            self._fixture(2002, "2099-09-01", "NS", "Juventus", "Bologna"),
        ]
        service._fetch_api_day_fixtures = lambda target_date: []
        service._fetch_matches = lambda target_date=None, day_margin=1: []

        payload = service.get_live_matches(target_date=date(2099, 9, 1), limit=50)

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
        self.assertIn("bookmaker_baseline", payload)
        self.assertTrue(payload["bookmaker_baseline"].get("generated"))
        self.assertGreaterEqual(len(payload["decision_cards"]), 2)
        labels = {c["value_label"] for c in payload["decision_cards"]}
        self.assertTrue(labels.issubset({"PLAY", "BORDERLINE", "NO BET"}))
        self.assertIn("bookmaker_fair_probability", payload["decision_cards"][0])
        self.assertIn("fair_odd", payload["decision_cards"][0])

    def test_get_day_matches_enriches_db_rows_with_decision_cards(self):
        """MATCH-01: quando la fixture e' gia' nel DB locale (`match.odds`
        gia' caricato dalla query unica di `_fetch_matches`), la riga di
        LISTA (Match Center) deve includere badge decision/quote/edge/EV
        senza alcuna fetch odds aggiuntiva verso l'API esterna."""
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h"]
        service._fetch_api_day_fixtures = lambda target_date: []

        match = Match(
            id_match_fk=str(uuid.uuid4()),
            id_fixture=5001,
            date_match="2026-09-01T18:00:00+00:00",
            status="NS",
            name_home="Inter",
            name_away="Milan",
            title_league="Serie A",
        )
        service._fetch_matches = lambda target_date=None, day_margin=1: [match]
        service._predict_fixture = lambda fixture_id, markets: {
            "h2h": {"prediction": 1, "probability": 0.75, "model_name": "logistic", "run_id": "run-x"}
        }
        service._aggregate_odds_from_db = lambda m: {
            "h2h": [
                {"outcome": "Home", "avg_odd": 1.5, "min_odd": 1.4, "max_odd": 1.6, "bookmakers": 5},
                {"outcome": "Away", "avg_odd": 3.0, "min_odd": 2.8, "max_odd": 3.2, "bookmakers": 5},
            ]
        }

        payload = service.get_day_matches(target_date=date(2026, 9, 1), limit=50, with_predictions=True)

        self.assertEqual(payload.returned, 1)
        row = payload.rows[0]
        self.assertTrue(row["decision_cards"])
        self.assertIsNotNone(row["best_decision"])
        self.assertIn(row["best_decision"]["value_label"], {"PLAY", "BORDERLINE", "NO BET"})
        self.assertIn("fair_odd", row["best_decision"])
        self.assertIsNotNone(row["best_decision"]["odd"])

    def test_get_day_matches_without_db_match_has_no_decision(self):
        """Fixture nota solo dal feed API (non ancora nel DB locale): niente
        badge decision (nessuna quota disponibile senza una fetch odds
        dedicata per riga, deliberatamente evitata per non consumare la
        quota API-Sports su centinaia di righe) - mai un dato inventato."""
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h"]
        service._fetch_api_day_fixtures = lambda target_date: [
            self._fixture(1001, "2026-09-01", "NS", "Inter", "Milan"),
        ]
        service._fetch_matches = lambda target_date=None, day_margin=1: []
        service._predict_fixture = lambda fixture_id, markets: {
            "h2h": {"prediction": 1, "probability": 0.72, "model_name": "logistic", "run_id": "run-test"}
        }

        payload = service.get_day_matches(target_date=date(2026, 9, 1), limit=50, with_predictions=True)

        self.assertEqual(payload.rows[0]["decision_cards"], [])
        self.assertIsNone(payload.rows[0]["best_decision"])


class TestFetchMatchesDateFilter(unittest.TestCase):
    """Bug fix performance: `_fetch_matches` caricava l'INTERO storico
    (nessun filtro SQL sulla data) ad ogni richiesta dashboard, impiegando
    20-40+ secondi con un DB reale di decine di migliaia di righe. Usa un
    vero engine SQLite in-memory (stesso pattern di crud_repository_test.py)
    per esercitare DAVVERO la query generata, non un mock."""

    def _make_session_factory(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def _add_match(self, session, *, fixture_id: int, date_match: str):
        session.add(
            Match(
                id_match_fk=str(uuid.uuid4()),
                id_fixture=fixture_id,
                date_match=date_match,
                status="NS",
            )
        )
        session.commit()

    def test_only_matches_within_target_range_are_loaded(self):
        session_factory = self._make_session_factory()
        with session_factory() as session:
            self._add_match(session, fixture_id=1, date_match="2026-09-03T18:00:00+00:00")  # target
            self._add_match(session, fixture_id=2, date_match="2026-09-02T10:00:00+00:00")  # margine -1gg
            self._add_match(session, fixture_id=3, date_match="2026-09-04T21:00:00+00:00")  # margine +1gg
            self._add_match(session, fixture_id=4, date_match="2020-01-01T10:00:00+00:00")  # fuori range (storico)
            self._add_match(session, fixture_id=5, date_match="2030-01-01T10:00:00+00:00")  # fuori range (futuro)

        original_session_local = dashboard_service_module.SessionLocal
        dashboard_service_module.SessionLocal = session_factory
        try:
            service = DashboardService.__new__(DashboardService)  # evita __init__ (registry/DB reale)
            rows = service._fetch_matches(target_date=date(2026, 9, 3), day_margin=1)
        finally:
            dashboard_service_module.SessionLocal = original_session_local

        fixture_ids = {row.id_fixture for row in rows}
        self.assertEqual(fixture_ids, {1, 2, 3})

    def test_empty_db_returns_empty_list(self):
        session_factory = self._make_session_factory()

        original_session_local = dashboard_service_module.SessionLocal
        dashboard_service_module.SessionLocal = session_factory
        try:
            service = DashboardService.__new__(DashboardService)
            rows = service._fetch_matches(target_date=date(2026, 9, 3))
        finally:
            dashboard_service_module.SessionLocal = original_session_local

        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()








