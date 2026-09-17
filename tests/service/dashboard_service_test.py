import unittest
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, time
from types import SimpleNamespace
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.api.dashboard_service as dashboard_service_module
from src.api.dashboard_service import CARDS_LINE_MARKETS, CARDS_LINES, CORNERS_LINE_MARKETS, CORNERS_LINES, DashboardService
from src.service_ia.model.match import Base, Match, Statistics


class TestDashboardService(unittest.TestCase):
    def setUp(self):
        snapshot_patch = mock.patch.object(DashboardService, "_load_snapshot_map", return_value={})
        snapshot_patch.start()
        self.addCleanup(snapshot_patch.stop)

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

    def test_available_dates_include_prediction_refresh_future_window(self):
        service = DashboardService.__new__(DashboardService)
        service.cfg = SimpleNamespace(daily_refresh_days_ahead=3)
        service.ledger_repo = SimpleNamespace(get_earliest_created_date=lambda: None)
        service._first_seen_date = lambda today: date(2026, 9, 9)

        payload = service.get_available_dates(today=date(2026, 9, 11))

        self.assertEqual(payload["dates"], [
            "2026-09-14",
            "2026-09-13",
            "2026-09-12",
            "2026-09-11",
            "2026-09-10",
            "2026-09-09",
        ])
        self.assertEqual(payload["first_date"], "2026-09-09")
        self.assertEqual(payload["last_date"], "2026-09-14")

    def test_get_day_matches_from_api_feed(self):
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h"]
        service._fetch_api_day_fixtures = lambda target_date, force_refresh=False: [
            self._fixture(1001, "2026-09-01", "NS", "Inter", "Milan"),
            self._fixture(1002, "2026-09-01", "1H", "Roma", "Lazio"),
        ]
        service._fetch_matches = lambda target_date=None, day_margin=1: []
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None, allow_compute=True: {
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
            force_refresh=True,
        )

        self.assertEqual(payload.total, 2)
        self.assertEqual(payload.returned, 2)
        self.assertEqual(payload.rows[0]["source"], "api_sports")
        self.assertIn("h2h", payload.rows[0]["predictions"])

    def test_get_live_matches_filter(self):
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h"]
        service.get_day_matches = mock.Mock(
            return_value=dashboard_service_module.DashboardDayData(
                date="2099-09-01",
                total=1,
                returned=1,
                model_markets=["h2h"],
                rows=[
                    {"fixture_id": 2001, "phase": "live", "datetime": "2099-09-01T18:45:00+00:00", "league": "Serie A", "home": "Napoli"},
                ],
            )
        )

        payload = service.get_live_matches(target_date=date(2099, 9, 1), limit=50)

        service.get_day_matches.assert_called_once_with(
            target_date=date(2099, 9, 1),
            limit=0,
            with_predictions=True,
            markets=["h2h"],
            phase="live",
        )
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
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None, allow_compute=True: {
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
        self.assertIn("model_void_odd", payload["decision_cards"][0])
        self.assertIn("market_fair_odd", payload["decision_cards"][0])
        self.assertIn("odds_edge_absolute", payload["decision_cards"][0])
        self.assertIn("expected_roi_percent", payload["decision_cards"][0])

    def test_get_match_detail_adds_correct_for_finished_match(self):
        """`get_match_detail` (2026-09-10, colorazione badge per esito
        reale): a differenza della vista lista, qui il `db_match` con le
        statistiche finali arriva da `_fetch_db_match_by_fixture`."""
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h"]

        service._fetch_api_fixture_detail = lambda fixture_id: self._fixture(
            fixture_id=fixture_id, day="2026-09-01", status="FT", home="Inter", away="Roma"
        )
        db_match = Match(id_match_fk=str(uuid.uuid4()), id_fixture=1234, id_team_home=10, id_team_away=20)
        db_match.statistics = [
            Statistics(id_statistics_fk=str(uuid.uuid4()), statistics_team_id=10, score_ft=1),
            Statistics(id_statistics_fk=str(uuid.uuid4()), statistics_team_id=20, score_ft=0),
        ]
        service._fetch_db_match_by_fixture = lambda fixture_id: db_match
        service._fetch_api_events = lambda fixture_id: []
        service._fetch_api_odds = lambda fixture_id: None
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None, allow_compute=True: {
            "h2h": {"prediction": 1, "probability": 0.72, "model_name": "logistic", "run_id": "run-1"}
        }

        payload = service.get_match_detail(fixture_id=1234, with_predictions=True)

        self.assertTrue(payload["predictions"]["h2h"]["correct"])

    def test_recompute_predictions_calls_predict_fixture_with_force_true(self):
        """Bottone "Ricalcola previsione" (2026-09-10, punto 4/4): deve
        SEMPRE passare `force=True` a `_predict_fixture`, indipendentemente
        da eventuali righe gia' salvate - il chiamante ha chiesto
        esplicitamente un ricalcolo, non un fast-path."""
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h", "goal_no_goal"]
        service._fetch_api_fixture_detail = lambda fixture_id: self._fixture(
            fixture_id=fixture_id, day="2026-09-01", status="FT", home="Inter", away="Roma"
        )
        service._fetch_db_match_by_fixture = lambda fixture_id: None

        captured = []

        def _capturing_predict(fixture_id, markets, db_match=None, status=None, allow_compute=True, force=False):
            captured.append({"fixture_id": fixture_id, "markets": markets, "status": status, "force": force})
            return {"h2h": {"prediction": 1, "probability": 0.9, "model_name": "m", "run_id": "r1"}}

        service._predict_fixture = _capturing_predict

        result = service.recompute_predictions(fixture_id=555)

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["fixture_id"], 555)
        self.assertEqual(sorted(captured[0]["markets"]), ["goal_no_goal", "h2h"])
        self.assertEqual(captured[0]["status"], "FT")
        self.assertTrue(captured[0]["force"])
        self.assertTrue(result["found"])
        self.assertIn("h2h", result["predictions"])

    def test_recompute_predictions_respects_explicit_markets(self):
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h", "goal_no_goal", "under_over_2_5"]
        service._fetch_api_fixture_detail = lambda fixture_id: self._fixture(
            fixture_id=fixture_id, day="2026-09-01", status="FT", home="Inter", away="Roma"
        )
        service._fetch_db_match_by_fixture = lambda fixture_id: None

        captured = []

        def _capturing_predict(fixture_id, markets, db_match=None, status=None, allow_compute=True, force=False):
            captured.append(markets)
            return {}

        service._predict_fixture = _capturing_predict

        service.recompute_predictions(fixture_id=555, markets=["goal_no_goal"])

        self.assertEqual(captured[0], ["goal_no_goal"])

    def test_recompute_predictions_fixture_not_found(self):
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h"]
        service._fetch_api_fixture_detail = lambda fixture_id: None
        service._fetch_db_match_by_fixture = lambda fixture_id: None

        def _boom(*args, **kwargs):
            raise AssertionError("_predict_fixture NON deve essere chiamato per una fixture inesistente")

        service._predict_fixture = _boom

        result = service.recompute_predictions(fixture_id=999)

        self.assertFalse(result["found"])
        self.assertEqual(result["predictions"], {})

    def test_get_day_matches_enriches_db_rows_with_decision_cards(self):
        """MATCH-01: quando la fixture e' gia' nel DB locale (`match.odds`
        gia' caricato dalla query unica di `_fetch_matches`), la riga di
        LISTA (Match Center) deve includere badge decision/quote/edge/EV
        senza alcuna fetch odds aggiuntiva verso l'API esterna."""
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h"]
        service._fetch_api_day_fixtures = lambda target_date, force_refresh=False: []

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
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None, allow_compute=True: {
            "h2h": {"prediction": 1, "probability": 0.75, "model_name": "logistic", "run_id": "run-x"}
        }
        service._aggregate_odds_from_db = lambda m: {
            "h2h": [
                {"outcome": "Home", "avg_odd": 1.5, "min_odd": 1.4, "max_odd": 1.6, "bookmakers": 5},
                {"outcome": "Away", "avg_odd": 3.0, "min_odd": 2.8, "max_odd": 3.2, "bookmakers": 5},
            ]
        }

        payload = service.get_day_matches(
            target_date=date(2026, 9, 1),
            limit=50,
            with_predictions=True,
            force_refresh=True,
        )

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
        service._fetch_api_day_fixtures = lambda target_date, force_refresh=False: [
            self._fixture(1001, "2026-09-01", "NS", "Inter", "Milan"),
        ]
        service._fetch_matches = lambda target_date=None, day_margin=1: []
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None, allow_compute=True: {
            "h2h": {"prediction": 1, "probability": 0.72, "model_name": "logistic", "run_id": "run-test"}
        }

        payload = service.get_day_matches(
            target_date=date(2026, 9, 1),
            limit=50,
            with_predictions=True,
            force_refresh=True,
        )

        self.assertEqual(payload.rows[0]["decision_cards"], [])
        self.assertIsNone(payload.rows[0]["best_decision"])

    def test_missing_odd_card_keeps_model_void_and_is_not_play(self):
        service = DashboardService()
        cards = service._build_decision_cards(
            row_context={"home": "Inter", "away": "Milan"},
            predictions={"under_over_2_5": {"prediction": 1, "probability": 0.60}},
            odds_summary={},
            bookmaker_baseline={},
        )
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["value_label"], "SENZA QUOTA")
        self.assertAlmostEqual(cards[0]["model_void_odd"], 1.6666667, places=6)
        self.assertIsNone(cards[0]["odds_edge_absolute"])
        self.assertIsNone(cards[0]["ev"])

    def test_multiclass_1x2_prices_each_matching_outcome_and_derives_dc(self):
        service = DashboardService()
        odds = {
            "h2h": [
                {"outcome": "Home", "avg_odd": 2.1, "bookmakers": 3},
                {"outcome": "Draw", "avg_odd": 3.2, "bookmakers": 3},
                {"outcome": "Away", "avg_odd": 3.8, "bookmakers": 3},
            ],
            "dc": [
                {"outcome": "Home/Draw", "avg_odd": 1.3, "bookmakers": 3},
                {"outcome": "Draw/Away", "avg_odd": 1.7, "bookmakers": 3},
                {"outcome": "Home/Away", "avg_odd": 1.4, "bookmakers": 3},
            ],
        }
        cards = service._build_decision_cards(
            row_context={"home": "Inter", "away": "Milan"},
            predictions={
                "1x2": {
                    "probabilities": {"HOME": 0.50, "DRAW": 0.30, "AWAY": 0.20},
                    "model_name": "multiclass",
                    "run_id": "run-1x2",
                }
            },
            odds_summary=odds,
            bookmaker_baseline=dashboard_service_module.build_fixture_baseline(odds),
        )
        one_x_two = {card["outcome"]: card for card in cards if card["market"] == "1x2"}
        double_chance = {card["outcome"]: card for card in cards if card["market"] == "dc"}
        self.assertEqual(set(one_x_two), {"Home", "Draw", "Away"})
        self.assertEqual(one_x_two["Draw"]["market_odd"], 3.2)
        self.assertEqual(one_x_two["Away"]["market_odd"], 3.8)
        self.assertAlmostEqual(double_chance["Home/Draw"]["predicted_probability"], 0.8)
        self.assertAlmostEqual(double_chance["Draw/Away"]["predicted_probability"], 0.5)
        self.assertAlmostEqual(double_chance["Home/Away"]["predicted_probability"], 0.7)

    def test_official_card_uses_frozen_ledger_values(self):
        service = DashboardService()
        frozen = SimpleNamespace(
            market="under_over_2_5",
            outcome="Over 2.5",
            line="2.5",
            p_model=0.64,
            odd=1.80,
            model_void_odd=1.5625,
            market_fair_odd=1.91,
            fair_odd=1.91,
            odds_edge_absolute=0.2375,
            odds_edge_percent=15.2,
            prob_edge=0.08,
            ev=0.152,
            expected_roi_percent=15.2,
            play_threshold_odd=1.59375,
            min_edge_percent=2.0,
            value_label="PLAY",
            value_reason="Quota sopra la soglia PLAY",
            decision="PLAY",
            policy_version="decision_policy_v2_model_break_even",
            bookmaker_count=4,
            model_name="frozen-model",
            model_run_id="frozen-run",
            is_settled=True,
            settlement_status="settled_win",
            pnl=0.8,
            captured_at=None,
        )
        card = service._official_card(frozen)
        self.assertTrue(card["is_official"])
        self.assertEqual(card["official_outcome"], "WON")
        self.assertEqual(card["model_void_odd"], 1.5625)
        self.assertEqual(card["policy_version"], "decision_policy_v2_model_break_even")

    def test_predict_fixture_delegates_to_snapshot_service_with_status(self):
        """`_predict_fixture` (2026-09-09, refactor cache persistita) non
        carica piu' modelli/frame da solo - delega interamente a
        `PredictionSnapshotService.resolve_predictions`, passando
        `db_match`/`status` cosi' com'e' (il regime finale/non-finale e'
        deciso li', non qui). Il ROUTING db_match-presente-vs-assente e'
        ora testato in `prediction_snapshot_service_test.py`, dove vive
        davvero questa logica."""
        service = DashboardService.__new__(DashboardService)
        service._prediction_cache = {}

        class _FakeSnapshotService:
            def __init__(self):
                self.calls = []

            def resolve_predictions(self, fixture_id, markets, db_match=None, status=None, allow_compute=True, force=False):
                self.calls.append(
                    {
                        "fixture_id": fixture_id,
                        "markets": markets,
                        "db_match": db_match,
                        "status": status,
                        "allow_compute": allow_compute,
                    }
                )
                return {"h2h": {"prediction": 1, "probability": 0.6, "model_name": "m", "run_id": "r1"}}

        service._snapshot_service = _FakeSnapshotService()
        sentinel_match = object()

        payload = service._predict_fixture(
            fixture_id=123, markets=["h2h"], db_match=sentinel_match, status="FT"
        )

        self.assertEqual(len(service._snapshot_service.calls), 1)
        call = service._snapshot_service.calls[0]
        self.assertEqual(call["fixture_id"], 123)
        self.assertEqual(call["markets"], ["h2h"])
        self.assertIs(call["db_match"], sentinel_match)
        self.assertEqual(call["status"], "FT")
        self.assertTrue(call["allow_compute"])
        self.assertEqual(payload["h2h"]["prediction"], 1)

    def test_predict_fixture_caches_within_same_instance(self):
        """Una seconda chiamata con la STESSA chiave (fixture+mercati) non
        deve richiamare di nuovo `resolve_predictions` - stesso principio
        di cache "per richiesta" gia' in uso prima del refactor."""
        service = DashboardService.__new__(DashboardService)
        service._prediction_cache = {}

        class _FakeSnapshotService:
            def __init__(self):
                self.call_count = 0

            def resolve_predictions(self, fixture_id, markets, db_match=None, status=None, allow_compute=True, force=False):
                self.call_count += 1
                return {"h2h": {"prediction": 1, "probability": 0.6, "model_name": "m", "run_id": "r1"}}

        service._snapshot_service = _FakeSnapshotService()

        service._predict_fixture(fixture_id=123, markets=["h2h"], status="NS")
        service._predict_fixture(fixture_id=123, markets=["h2h"], status="NS")

        self.assertEqual(service._snapshot_service.call_count, 1)

    def test_predict_fixture_cache_key_distinguishes_allow_compute(self):
        """`allow_compute` fa parte della chiave di cache (2026-09-10): la
        stessa fixture+mercati vista prima con `allow_compute=False` (lista
        storica) e poi con `allow_compute=True` (es. dettaglio) non deve
        MAI riusare la voce di cache dell'altra modalita' - altrimenti un
        payload "solo cio' che e' salvato" potrebbe restare incollato anche
        quando sarebbe stato lecito calcolare."""
        service = DashboardService.__new__(DashboardService)
        service._prediction_cache = {}

        class _FakeSnapshotService:
            def __init__(self):
                self.call_count = 0

            def resolve_predictions(self, fixture_id, markets, db_match=None, status=None, allow_compute=True, force=False):
                self.call_count += 1
                return {"h2h": {"prediction": 1, "probability": 0.6, "model_name": "m", "run_id": "r1"}}

        service._snapshot_service = _FakeSnapshotService()

        service._predict_fixture(fixture_id=123, markets=["h2h"], status="FT", allow_compute=False)
        service._predict_fixture(fixture_id=123, markets=["h2h"], status="FT", allow_compute=True)

        self.assertEqual(service._snapshot_service.call_count, 2)

    def test_get_overview_never_computes_predictions(self):
        """Fix performance: `get_overview` (usato dal frontend SOLO per i
        counts/badge di riepilogo, MAI per i valori di predizione - nessun
        componente legge `day_highlights`/`live_preview`/predictions) non
        deve piu' scatenare alcun calcolo ML/DB per le predizioni - prima
        duplicava ESATTAMENTE lo stesso lavoro gia' fatto da `/dashboard/day`
        (stessa data), chiamato in parallelo dal frontend ad ogni cambio
        giorno."""
        service = DashboardService()
        service.get_day_matches = mock.Mock(
            return_value=dashboard_service_module.DashboardDayData(
                date="2099-09-01",
                total=1,
                returned=1,
                model_markets=["h2h"],
                rows=[{"phase": "to_play", "predictions": {}}],
            )
        )

        overview = service.get_overview(target_date=date(2099, 9, 1))

        service.get_day_matches.assert_called_once_with(
            target_date=date(2099, 9, 1), limit=0, with_predictions=False
        )
        self.assertEqual(overview["counts"]["total"], 1)
        self.assertEqual(overview["counts"]["to_play"], 1)
        self.assertEqual(overview["counts"]["with_prediction"], 0)

    def test_get_day_matches_skips_api_when_db_already_synced_for_that_date(self):
        """Fix quota API-Sports (2026-09-07): il DB locale e' sincronizzato
        quotidianamente dai job schedulati (`data_daily_refresh`,
        `data_sync_today`, `future_sync`) per la finestra ieri -> oggi+N
        giorni, su TUTTI i campionati censiti. Se il DB ha GIA' almeno una
        riga per `target_date`, NON deve piu' scattare alcuna fetch verso il
        provider esterno (prima: sempre chiamata, fino a 18 richieste HTTP
        per cambio data, una per campionato configurato)."""
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h"]

        def _boom(target_date, force_refresh=False):
            raise AssertionError("_fetch_api_day_fixtures NON deve essere chiamato: il DB ha gia' dati per la data")

        service._fetch_api_day_fixtures = _boom

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
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None, allow_compute=True: {}

        payload = service.get_day_matches(target_date=date(2026, 9, 1), limit=50, with_predictions=True)

        self.assertEqual(payload.returned, 1)
        self.assertEqual(payload.rows[0]["source"], "db")
        self.assertEqual(payload.rows[0]["home"], "Inter")

    def test_get_day_matches_does_not_call_api_when_db_is_empty(self):
        """La vista passiva resta DB-first anche quando il DB è vuoto."""
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h"]
        call_count = {"n": 0}

        def _counting_fetch(target_date, force_refresh=False):
            call_count["n"] += 1
            return [self._fixture(1001, "2026-09-01", "NS", "Inter", "Milan")]

        service._fetch_api_day_fixtures = _counting_fetch
        service._fetch_matches = lambda target_date=None, day_margin=1: []
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None, allow_compute=True: {}

        payload = service.get_day_matches(target_date=date(2026, 9, 1), limit=50, with_predictions=True)

        self.assertEqual(call_count["n"], 0)
        self.assertEqual(payload.returned, 0)

    def test_get_day_matches_does_not_call_api_for_rows_from_other_dates(self):
        """Righe fuori data non riattivano implicitamente il provider."""
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h"]
        call_count = {"n": 0}

        def _counting_fetch(target_date, force_refresh=False):
            call_count["n"] += 1
            return []

        service._fetch_api_day_fixtures = _counting_fetch

        other_day_match = Match(
            id_match_fk=str(uuid.uuid4()),
            id_fixture=6001,
            date_match="2026-08-31T18:00:00+00:00",  # giorno precedente, non target_date
            status="FT",
        )
        service._fetch_matches = lambda target_date=None, day_margin=1: [other_day_match]
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None, allow_compute=True: {}

        service.get_day_matches(target_date=date(2026, 9, 1), limit=50, with_predictions=True)

        self.assertEqual(call_count["n"], 0)

    def test_get_day_matches_is_db_first_for_future_date(self):
        """Anche le date future leggono gli snapshot prodotti dai job."""
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h"]
        call_count = {"n": 0}

        def _counting_fetch(target_date, force_refresh=False):
            call_count["n"] += 1
            return []

        service._fetch_api_day_fixtures = _counting_fetch

        future_match = Match(
            id_match_fk=str(uuid.uuid4()),
            id_fixture=7001,
            date_match="2099-09-01T18:00:00+00:00",  # ben nel futuro rispetto a "oggi"
            status="NS",
        )
        service._fetch_matches = lambda target_date=None, day_margin=1: [future_match]
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None, allow_compute=True: {}

        service.get_day_matches(target_date=date(2099, 9, 1), limit=50, with_predictions=True)

        self.assertEqual(call_count["n"], 0)

    def test_get_day_matches_is_db_first_for_today(self):
        """Anche oggi la vista non interroga il provider: lo fanno i job."""
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h"]
        call_count = {"n": 0}

        def _counting_fetch(target_date, force_refresh=False):
            call_count["n"] += 1
            return []

        service._fetch_api_day_fixtures = _counting_fetch

        today = dashboard_service_module.datetime.now(dashboard_service_module.timezone.utc).date()
        today_match = Match(
            id_match_fk=str(uuid.uuid4()),
            id_fixture=7002,
            date_match=f"{today.isoformat()}T18:00:00+00:00",
            status="NS",
        )
        service._fetch_matches = lambda target_date=None, day_margin=1: [today_match]
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None, allow_compute=True: {}

        service.get_day_matches(target_date=today, limit=50, with_predictions=True)

        self.assertEqual(call_count["n"], 0)

    def test_get_day_matches_force_refresh_bypasses_historical_skip(self):
        """Bottone "Forza aggiornamento": anche per una data STORICA gia'
        completamente sincronizzata a DB, `force_refresh=True` deve
        comunque richiamare il provider esterno (caso eccezionale: dato
        importato errato, correzione tardiva del provider)."""
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h"]
        call_count = {"n": 0}

        def _counting_fetch(target_date, force_refresh=False):
            call_count["n"] += 1
            self.assertTrue(force_refresh)
            return []

        service._fetch_api_day_fixtures = _counting_fetch

        historical_match = Match(
            id_match_fk=str(uuid.uuid4()),
            id_fixture=8001,
            date_match="2026-09-01T18:00:00+00:00",  # storica rispetto a "oggi" (2026-09-07)
            status="FT",
        )
        service._fetch_matches = lambda target_date=None, day_margin=1: [historical_match]
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None, allow_compute=True: {}

        service.get_day_matches(
            target_date=date(2026, 9, 1), limit=50, with_predictions=True, force_refresh=True
        )

        self.assertEqual(call_count["n"], 1)

    def test_get_day_matches_force_refresh_still_blocked_when_quota_exhausted(self):
        """`force_refresh` bypassa SOLO lo skip DB-first/la cache, MAI il
        guard "quota esaurita" (`is_quota_exhausted_today`, verificato
        dentro `_fetch_api_day_fixtures` REALE, non mockata qui): nessun
        bottone puo' forzare una chiamata quando la quota e' al 100%."""
        service = DashboardService()
        service.cfg = type("Cfg", (), {"leagues": [135], "seasons": [2026]})()
        service.registry.list_markets = lambda: ["h2h"]
        service._fetch_matches = lambda target_date=None, day_margin=1: []
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None, allow_compute=True: {}
        DashboardService._api_cache = {}

        with mock.patch.object(
            dashboard_service_module, "is_quota_exhausted_today", return_value=True
        ), mock.patch.object(dashboard_service_module, "base_api_statistics") as mocked_call:
            payload = service.get_day_matches(
                target_date=date(2026, 9, 1), limit=50, with_predictions=True, force_refresh=True
            )

        mocked_call.assert_not_called()
        self.assertEqual(payload.returned, 0)
        DashboardService._api_cache = {}

    def test_get_day_matches_historical_date_passes_allow_compute_false(self):
        """Vista storica sempre veloce (2026-09-10): per una data passata,
        `_predict_fixture` deve sempre ricevere `allow_compute=False`, sia
        per le righe DB-only sia (quando `force_refresh` bypassa lo skip
        API) per quelle arricchite dal feed esterno."""
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h"]

        historical_match = Match(
            id_match_fk=str(uuid.uuid4()),
            id_fixture=9001,
            date_match="2026-09-01T18:00:00+00:00",
            status="FT",
            name_home="Inter",
            name_away="Milan",
        )
        service._fetch_matches = lambda target_date=None, day_margin=1: [historical_match]

        captured = []

        def _capturing_predict(fixture_id, markets, db_match=None, status=None, allow_compute=True):
            captured.append(allow_compute)
            return {}

        service._predict_fixture = _capturing_predict

        service.get_day_matches(target_date=date(2026, 9, 1), limit=50, with_predictions=True)

        self.assertEqual(captured, [False])

    def test_get_day_matches_today_reads_only_saved_predictions(self):
        """Oggi/date future leggono snapshot job senza calcolo inline."""
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h"]

        today = dashboard_service_module.datetime.now(dashboard_service_module.timezone.utc).date()
        today_match = Match(
            id_match_fk=str(uuid.uuid4()),
            id_fixture=9101,
            date_match=f"{today.isoformat()}T18:00:00+00:00",
            status="NS",
        )
        service._fetch_matches = lambda target_date=None, day_margin=1: [today_match]

        captured = []

        def _capturing_predict(fixture_id, markets, db_match=None, status=None, allow_compute=True):
            captured.append(allow_compute)
            return {}

        service._predict_fixture = _capturing_predict

        service.get_day_matches(target_date=today, limit=50, with_predictions=True)

        self.assertEqual(captured, [False])

    def test_get_day_matches_loads_snapshots_once_for_all_fixtures(self):
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h", "goal_no_goal"]
        matches = [
            Match(
                id_match_fk=str(uuid.uuid4()),
                id_fixture=fixture_id,
                date_match="2026-09-01T18:00:00+00:00",
                status="FT",
            )
            for fixture_id in (101, 102, 103)
        ]
        service._fetch_matches = lambda target_date=None, day_margin=0: matches
        service._load_snapshot_map = mock.Mock(return_value={})
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None, allow_compute=True: {}

        service.get_day_matches(target_date=date(2026, 9, 1), with_predictions=True)

        service._load_snapshot_map.assert_called_once_with(
            [101, 102, 103], markets=["h2h", "goal_no_goal"]
        )

    def test_dashboard_bundle_builds_all_sections_from_one_day_load(self):
        service = DashboardService()
        complete_day = dashboard_service_module.DashboardDayData(
            date="2026-09-01",
            total=3,
            returned=3,
            model_markets=["h2h"],
            rows=[
                {"fixture_id": 1, "phase": "live", "predictions": {"h2h": {}}, "datetime": "a"},
                {"fixture_id": 2, "phase": "to_play", "predictions": {}, "datetime": "b"},
                {"fixture_id": 3, "phase": "finished", "predictions": {}, "datetime": "c"},
            ],
        )
        service.get_day_matches = mock.Mock(return_value=complete_day)

        payload = service.get_dashboard_bundle(
            target_date=date(2026, 9, 1),
            limit=2,
            search_text="Inter",
        )

        service.get_day_matches.assert_called_once_with(
            target_date=date(2026, 9, 1),
            limit=0,
            with_predictions=True,
            search_text="Inter",
            force_refresh=False,
        )
        self.assertEqual(payload["overview"]["counts"]["live"], 1)
        self.assertEqual(payload["live"]["returned"], 1)
        self.assertEqual(payload["day"]["returned"], 2)
        self.assertEqual(len(payload["day"]["rows"]), 2)


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

    def test_finished_match_correctness_uses_only_eager_statistics_columns(self):
        session_factory = self._make_session_factory()
        match_id = str(uuid.uuid4())
        with session_factory() as session:
            session.add(
                Match(
                    id_match_fk=match_id,
                    id_fixture=10,
                    id_team_home=100,
                    id_team_away=200,
                    date_match="2026-09-03T18:00:00+00:00",
                    status="FT",
                )
            )
            session.add_all(
                [
                    Statistics(
                        id_match=match_id,
                        statistics_team_id=100,
                        score_ht=1,
                        score_ft=2,
                        corners=7,
                        yellow_cards=2,
                        red_cards=0,
                    ),
                    Statistics(
                        id_match=match_id,
                        statistics_team_id=200,
                        score_ht=0,
                        score_ft=1,
                        corners=4,
                        yellow_cards=3,
                        red_cards=1,
                    ),
                ]
            )
            session.commit()

        original_session_local = dashboard_service_module.SessionLocal
        dashboard_service_module.SessionLocal = session_factory
        try:
            service = DashboardService.__new__(DashboardService)
            match = service._fetch_matches(target_date=date(2026, 9, 3))[0]
        finally:
            dashboard_service_module.SessionLocal = original_session_local

        home, away, complete = service._resolve_final_stat_dicts(match)
        self.assertTrue(complete)
        self.assertEqual(home, {"score_ft": 2, "score_ht": 1, "corners": 7, "yellow_cards": 2, "red_cards": 0})
        self.assertEqual(away, {"score_ft": 1, "score_ht": 0, "corners": 4, "yellow_cards": 3, "red_cards": 1})


class TestQuotaExhaustedGuard(unittest.TestCase):
    """Fix (2026-09-07): quando la quota API-Sports e' gia' segnalata
    esaurita OGGI (check autoritativo `is_quota_exhausted_today`), NESSUNA
    delle funzioni di fetch diretto verso il provider esterno deve tentare
    la chiamata HTTP - protegge ogni punto di ingresso (polling frontend
    ogni 60s, job schedulati, richieste manuali) in un colpo solo, a
    prescindere da chi chiama questi metodi."""

    def setUp(self):
        self.service = DashboardService.__new__(DashboardService)
        self.service.cfg = type("Cfg", (), {"leagues": [135, 140], "seasons": [2026]})()
        # `_api_cache` e' un attributo di CLASSE condiviso tra istanze/test:
        # azzerato esplicitamente per non trovare una cache-hit "vecchia"
        # lasciata da un altro test e non esercitare davvero la guardia.
        DashboardService._api_cache = {}

    def tearDown(self):
        DashboardService._api_cache = {}
        DashboardService._api_cache_locks = {}

    def test_day_fetch_single_flight_calls_provider_once_per_cache_key(self):
        self.service.cfg = type("Cfg", (), {"leagues": [135], "seasons": [2026]})()
        entered = threading.Event()
        release = threading.Event()

        def slow_call(path, params):
            entered.set()
            release.wait(timeout=2)
            return [{"fixture": {"id": 1}}]

        with mock.patch.object(
            dashboard_service_module, "is_quota_exhausted_today", return_value=False
        ), mock.patch.object(
            dashboard_service_module, "base_api_statistics", side_effect=slow_call
        ) as mocked_call, ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(self.service._fetch_api_day_fixtures, date(2026, 9, 7))
            self.assertTrue(entered.wait(timeout=2))
            second = executor.submit(self.service._fetch_api_day_fixtures, date(2026, 9, 7))
            release.set()
            first_result = first.result(timeout=2)
            second_result = second.result(timeout=2)

        self.assertEqual(mocked_call.call_count, 1)
        self.assertEqual(first_result, second_result)

    def test_fetch_api_day_fixtures_skips_call_when_quota_exhausted(self):
        with mock.patch.object(dashboard_service_module, "is_quota_exhausted_today", return_value=True), mock.patch.object(
            dashboard_service_module, "base_api_statistics"
        ) as mocked_call:
            result = self.service._fetch_api_day_fixtures(date(2026, 9, 7))

        mocked_call.assert_not_called()
        self.assertEqual(result, [])

    def test_fetch_api_live_fixtures_skips_call_when_quota_exhausted(self):
        with mock.patch.object(dashboard_service_module, "is_quota_exhausted_today", return_value=True), mock.patch.object(
            dashboard_service_module, "base_api_statistics"
        ) as mocked_call:
            result = self.service._fetch_api_live_fixtures()

        mocked_call.assert_not_called()
        self.assertEqual(result, [])

    def test_fetch_api_fixture_detail_skips_call_when_quota_exhausted(self):
        with mock.patch.object(dashboard_service_module, "is_quota_exhausted_today", return_value=True), mock.patch.object(
            dashboard_service_module, "base_api_statistics"
        ) as mocked_call:
            result = self.service._fetch_api_fixture_detail(999)

        mocked_call.assert_not_called()
        self.assertIsNone(result)

    def test_fetch_api_events_skips_call_when_quota_exhausted(self):
        with mock.patch.object(dashboard_service_module, "is_quota_exhausted_today", return_value=True), mock.patch.object(
            dashboard_service_module, "base_api_statistics"
        ) as mocked_call:
            result = self.service._fetch_api_events(999)

        mocked_call.assert_not_called()
        self.assertEqual(result, [])

    def test_fetch_api_odds_skips_call_when_quota_exhausted(self):
        with mock.patch.object(dashboard_service_module, "is_quota_exhausted_today", return_value=True), mock.patch.object(
            dashboard_service_module, "base_api_statistics"
        ) as mocked_call:
            result = self.service._fetch_api_odds(999)

        mocked_call.assert_not_called()
        self.assertIsNone(result)

    def test_fetch_api_day_fixtures_still_calls_when_quota_not_exhausted(self):
        """Regressione: la guardia non deve bloccare il normale
        funzionamento quando la quota NON e' esaurita."""
        with mock.patch.object(dashboard_service_module, "is_quota_exhausted_today", return_value=False), mock.patch.object(
            dashboard_service_module, "base_api_statistics", return_value=[{"fixture": {"id": 1}}]
        ) as mocked_call:
            result = self.service._fetch_api_day_fixtures(date(2026, 9, 7))

        self.assertTrue(mocked_call.called)
        # 2 campionati configurati, stesso fixture id=1 in entrambi -> dedup a 1.
        self.assertEqual(len(result), 1)

    def test_fetch_api_day_fixtures_league_fetch_is_parallel_and_order_preserved(self):
        """Fix performance (2026-09-09): i campionati vengono interrogati in
        parallelo (ThreadPoolExecutor), non piu' uno alla volta - qui si
        verifica che TUTTI vengano comunque interrogati (nessuno saltato) e
        che l'ordine del risultato aggregato resti deterministico (stesso
        ordine di `cfg.leagues`, non l'ordine di completamento dei thread -
        `executor.map` lo garantisce)."""
        self.service.cfg = type("Cfg", (), {"leagues": [1, 2, 3, 4, 5], "seasons": [2026]})()

        def fake_call(path, params):
            return [{"fixture": {"id": params["league"]}}]

        with mock.patch.object(dashboard_service_module, "is_quota_exhausted_today", return_value=False), mock.patch.object(
            dashboard_service_module, "base_api_statistics", side_effect=fake_call
        ) as mocked_call:
            result = self.service._fetch_api_day_fixtures(date(2026, 9, 7))

        self.assertEqual(mocked_call.call_count, 5)
        self.assertEqual([r["fixture"]["id"] for r in result], [1, 2, 3, 4, 5])

    def test_fetch_api_day_fixtures_one_league_failing_does_not_block_others(self):
        self.service.cfg = type("Cfg", (), {"leagues": [1, 2, 3], "seasons": [2026]})()

        def fake_call(path, params):
            if params["league"] == 2:
                raise RuntimeError("boom")
            return [{"fixture": {"id": params["league"]}}]

        with mock.patch.object(dashboard_service_module, "is_quota_exhausted_today", return_value=False), mock.patch.object(
            dashboard_service_module, "base_api_statistics", side_effect=fake_call
        ):
            result = self.service._fetch_api_day_fixtures(date(2026, 9, 7))

        self.assertEqual([r["fixture"]["id"] for r in result], [1, 3])

    def test_fetch_api_live_fixtures_still_calls_when_quota_not_exhausted(self):
        with mock.patch.object(dashboard_service_module, "is_quota_exhausted_today", return_value=False), mock.patch.object(
            DashboardService, "_is_within_dashboard_api_window", return_value=True
        ), mock.patch.object(dashboard_service_module, "base_api_statistics", return_value=[]) as mocked_call:
            self.service._fetch_api_live_fixtures()

        self.assertTrue(mocked_call.called)


class TestApplyMonotonicProjection(unittest.TestCase):
    """Coerenza monotona tra le 4 soglie Under/Over collegata al serving
    (2026-09-09, richiesto esplicitamente dall'operatore): P(Over1.5) >=
    P(Over2.5) >= P(Over3.5) >= P(Over4.5) per la stessa fixture."""

    def _payload(self, **overrides):
        payload = {
            "under_over_1_5": {"prediction": 1, "probability": 0.6, "model_name": "m", "run_id": "r1"},
            "under_over_2_5": {"prediction": 1, "probability": 0.7, "model_name": "m", "run_id": "r2"},
            "under_over_3_5": {"prediction": 0, "probability": 0.3, "model_name": "m", "run_id": "r3"},
            "under_over_4_5": {"prediction": 0, "probability": 0.1, "model_name": "m", "run_id": "r4"},
        }
        payload.update(overrides)
        return payload

    def test_projects_violating_probabilities_to_monotone(self):
        payload = self._payload()
        DashboardService._apply_monotonic_projection(payload)

        probs = [payload[m]["probability"] for m in
                 ("under_over_1_5", "under_over_2_5", "under_over_3_5", "under_over_4_5")]
        self.assertEqual(probs, sorted(probs, reverse=True))
        # 2.5 (0.7) violava 1.5 (0.6): deve essere abbassata al minimo cumulativo.
        self.assertEqual(payload["under_over_2_5"]["probability"], 0.6)

    def test_prediction_recomputed_from_projected_probability(self):
        payload = self._payload()
        DashboardService._apply_monotonic_projection(payload)
        # 2.5 proiettata a 0.6 (>=0.5): prediction resta 1, ma ricalcolata
        # dalla probabilita' proiettata, non piu' quella grezza.
        self.assertEqual(payload["under_over_2_5"]["prediction"], 1)

    def test_already_monotone_payload_is_unchanged(self):
        payload = self._payload(under_over_2_5={"prediction": 0, "probability": 0.5, "model_name": "m", "run_id": "r2"})
        DashboardService._apply_monotonic_projection(payload)
        self.assertEqual(payload["under_over_2_5"]["probability"], 0.5)

    def test_skipped_when_a_threshold_is_missing(self):
        payload = self._payload()
        del payload["under_over_3_5"]
        original_2_5 = dict(payload["under_over_2_5"])
        DashboardService._apply_monotonic_projection(payload)
        self.assertEqual(payload["under_over_2_5"], original_2_5)

    def test_other_markets_untouched(self):
        payload = self._payload(goal_no_goal={"prediction": 1, "probability": 0.55, "model_name": "m", "run_id": "r5"})
        DashboardService._apply_monotonic_projection(payload)
        self.assertEqual(payload["goal_no_goal"]["probability"], 0.55)


class TestApplyLineMarketsMonotonicProjection(unittest.TestCase):
    """2026-09-13: stesso principio di `TestApplyMonotonicProjection` ma
    per Corners/Cards a linea configurabile (MARKET-05/06) - Over 10.5
    implica Over 9.5, che implica Over 8.5."""

    def _corners_payload(self, **overrides):
        payload = {
            "corners_line_8_5": {"prediction": 1, "probability": 0.6, "model_name": "m", "run_id": "r1"},
            "corners_line_9_5": {"prediction": 1, "probability": 0.7, "model_name": "m", "run_id": "r2"},
            "corners_line_10_5": {"prediction": 0, "probability": 0.3, "model_name": "m", "run_id": "r3"},
            "corners_line_11_5": {"prediction": 0, "probability": 0.1, "model_name": "m", "run_id": "r4"},
        }
        payload.update(overrides)
        return payload

    def test_projects_violating_probabilities_to_monotone(self):
        payload = self._corners_payload()
        DashboardService._apply_line_markets_monotonic_projection(payload, CORNERS_LINE_MARKETS, CORNERS_LINES)

        probs = [payload[m]["probability"] for m in CORNERS_LINE_MARKETS]
        self.assertEqual(probs, sorted(probs, reverse=True))
        # 9.5 (0.7) violava 8.5 (0.6): deve essere abbassata al minimo cumulativo.
        self.assertEqual(payload["corners_line_9_5"]["probability"], 0.6)

    def test_skipped_when_a_line_is_missing(self):
        payload = self._corners_payload()
        del payload["corners_line_10_5"]
        original = dict(payload["corners_line_9_5"])
        DashboardService._apply_line_markets_monotonic_projection(payload, CORNERS_LINE_MARKETS, CORNERS_LINES)
        self.assertEqual(payload["corners_line_9_5"], original)

    def test_cards_lines_projected_independently_from_corners(self):
        payload = {
            "cards_line_3_5": {"prediction": 1, "probability": 0.5, "model_name": "m", "run_id": "r1"},
            "cards_line_4_5": {"prediction": 1, "probability": 0.65, "model_name": "m", "run_id": "r2"},
            "cards_line_5_5": {"prediction": 0, "probability": 0.2, "model_name": "m", "run_id": "r3"},
            "cards_line_6_5": {"prediction": 0, "probability": 0.05, "model_name": "m", "run_id": "r4"},
        }
        DashboardService._apply_line_markets_monotonic_projection(payload, CARDS_LINE_MARKETS, CARDS_LINES)
        probs = [payload[m]["probability"] for m in CARDS_LINE_MARKETS]
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertEqual(payload["cards_line_4_5"]["probability"], 0.5)


class TestApplyLineMarketSignals(unittest.TestCase):
    def test_injects_signal_for_every_present_line_market(self):
        payload = {
            "corners_line_8_5": {"prediction": 1, "probability": 0.65},
            "cards_line_6_5": {"prediction": 0, "probability": 0.20},
            "h2h": {"prediction": 1, "probability": 0.55},
        }
        DashboardService._apply_line_market_signals(payload)

        self.assertIsNotNone(payload["corners_line_8_5"]["line_market_signal"])
        self.assertTrue(payload["cards_line_6_5"]["line_market_signal"]["signal"])  # 0.20 >= soglia 0.1686
        self.assertNotIn("line_market_signal", payload["h2h"])


class TestDashboardApiWindow(unittest.TestCase):
    """Finestra oraria 12:30-00:30 Europe/Rome per il risparmio quota
    (2026-09-09, richiesto esplicitamente dall'operatore): fuori da questa
    finestra, per la data ODIERNA, _fetch_api_day_fixtures/
    _fetch_api_live_fixtures non devono interrogare l'API esterna."""

    def setUp(self):
        self.service = DashboardService.__new__(DashboardService)
        self.service.cfg = type("Cfg", (), {"leagues": [135], "seasons": [2026]})()
        DashboardService._api_cache = {}

    def tearDown(self):
        DashboardService._api_cache = {}

    def test_window_boundaries(self):
        self.assertTrue(DashboardService._is_within_dashboard_api_window(time(12, 30)))
        self.assertFalse(DashboardService._is_within_dashboard_api_window(time(12, 29)))
        self.assertTrue(DashboardService._is_within_dashboard_api_window(time(23, 59)))
        self.assertTrue(DashboardService._is_within_dashboard_api_window(time(0, 0)))
        self.assertTrue(DashboardService._is_within_dashboard_api_window(time(0, 29)))
        self.assertFalse(DashboardService._is_within_dashboard_api_window(time(0, 30)))
        self.assertFalse(DashboardService._is_within_dashboard_api_window(time(6, 0)))

    def test_day_fixtures_skipped_for_today_outside_window(self):
        today = date(2026, 9, 9)
        with mock.patch.object(DashboardService, "_today_in_dashboard_timezone", return_value=today), mock.patch.object(
            DashboardService, "_is_within_dashboard_api_window", return_value=False
        ), mock.patch.object(dashboard_service_module, "is_quota_exhausted_today", return_value=False), mock.patch.object(
            dashboard_service_module, "base_api_statistics"
        ) as mocked_call:
            result = self.service._fetch_api_day_fixtures(today)

        mocked_call.assert_not_called()
        self.assertEqual(result, [])

    def test_day_fixtures_force_refresh_bypasses_window(self):
        """`force_refresh` (bottone "Forza aggiornamento", azione esplicita
        dell'operatore) bypassa la finestra oraria - a differenza del guard
        quota esaurita, che nessun bottone puo' bypassare."""
        today = date(2026, 9, 9)
        with mock.patch.object(DashboardService, "_today_in_dashboard_timezone", return_value=today), mock.patch.object(
            DashboardService, "_is_within_dashboard_api_window", return_value=False
        ), mock.patch.object(dashboard_service_module, "is_quota_exhausted_today", return_value=False), mock.patch.object(
            dashboard_service_module, "base_api_statistics", return_value=[]
        ) as mocked_call:
            self.service._fetch_api_day_fixtures(today, force_refresh=True)

        self.assertTrue(mocked_call.called)

    def test_day_fixtures_not_blocked_for_other_dates_outside_window(self):
        """La finestra oraria riguarda SOLO la data odierna - una data
        diversa (storica/futura) non e' toccata da questo guard."""
        today = date(2026, 9, 9)
        other_date = date(2026, 9, 10)
        with mock.patch.object(DashboardService, "_today_in_dashboard_timezone", return_value=today), mock.patch.object(
            DashboardService, "_is_within_dashboard_api_window", return_value=False
        ), mock.patch.object(dashboard_service_module, "is_quota_exhausted_today", return_value=False), mock.patch.object(
            dashboard_service_module, "base_api_statistics", return_value=[]
        ) as mocked_call:
            self.service._fetch_api_day_fixtures(other_date)

        self.assertTrue(mocked_call.called)

    def test_day_fixtures_called_for_today_inside_window(self):
        today = date(2026, 9, 9)
        with mock.patch.object(DashboardService, "_today_in_dashboard_timezone", return_value=today), mock.patch.object(
            DashboardService, "_is_within_dashboard_api_window", return_value=True
        ), mock.patch.object(dashboard_service_module, "is_quota_exhausted_today", return_value=False), mock.patch.object(
            dashboard_service_module, "base_api_statistics", return_value=[]
        ) as mocked_call:
            self.service._fetch_api_day_fixtures(today)

        self.assertTrue(mocked_call.called)

    def test_live_fixtures_skipped_outside_window(self):
        with mock.patch.object(DashboardService, "_is_within_dashboard_api_window", return_value=False), mock.patch.object(
            dashboard_service_module, "is_quota_exhausted_today", return_value=False
        ), mock.patch.object(dashboard_service_module, "base_api_statistics") as mocked_call:
            result = self.service._fetch_api_live_fixtures()

        mocked_call.assert_not_called()
        self.assertEqual(result, [])

    def test_live_fixtures_called_inside_window(self):
        with mock.patch.object(DashboardService, "_is_within_dashboard_api_window", return_value=True), mock.patch.object(
            dashboard_service_module, "is_quota_exhausted_today", return_value=False
        ), mock.patch.object(dashboard_service_module, "base_api_statistics", return_value=[]) as mocked_call:
            self.service._fetch_api_live_fixtures()

        self.assertTrue(mocked_call.called)


class TestExtractScores(unittest.TestCase):
    """`_extract_scores` (2026-09-10): preferisce SEMPRE `Match.score_home`/
    `score_away` (sempre disponibili, indipendenti dalle statistiche
    dettagliate), con fallback su `Statistics.score_ft`/`score_ht` SOLO per
    righe importate PRIMA di questo fix."""

    def test_prefers_match_score_over_statistics(self):
        match = Match(
            id_match_fk=str(uuid.uuid4()), id_fixture=1, id_team_home=10, id_team_away=20,
            score_home=3, score_away=1,
        )
        match.statistics = [
            Statistics(id_statistics_fk=str(uuid.uuid4()), statistics_team_id=10, score_ft=99),
            Statistics(id_statistics_fk=str(uuid.uuid4()), statistics_team_id=20, score_ft=99),
        ]

        self.assertEqual(DashboardService._extract_scores(match), {"home": 3, "away": 1})

    def test_falls_back_to_statistics_when_match_score_missing(self):
        match = Match(id_match_fk=str(uuid.uuid4()), id_fixture=1, id_team_home=10, id_team_away=20)
        match.statistics = [
            Statistics(id_statistics_fk=str(uuid.uuid4()), statistics_team_id=10, score_ft=2),
            Statistics(id_statistics_fk=str(uuid.uuid4()), statistics_team_id=20, score_ft=1),
        ]

        self.assertEqual(DashboardService._extract_scores(match), {"home": 2, "away": 1})

    def test_none_none_when_neither_source_available(self):
        match = Match(id_match_fk=str(uuid.uuid4()), id_fixture=1, id_team_home=10, id_team_away=20)
        match.statistics = []

        self.assertEqual(DashboardService._extract_scores(match), {"home": None, "away": None})


class TestResolveFinalStatDicts(unittest.TestCase):
    """`_resolve_final_stat_dicts` (2026-09-10, colorazione badge per esito
    reale): deve costruire i dict home/away SOLO da `Statistics` ORM reali
    quando disponibili (mai un dict parziale a mano - vedi il commento nel
    metodo sul perche' una chiave assente verrebbe letta come 0 da
    `_label_by_market`), con fallback "solo punteggio" da
    `Match.score_home`/`score_away` (2026-09-10: sempre disponibili
    indipendentemente dalle statistiche dettagliate, vedi
    `download_match_service.map_base_match`) quando `Statistics` manca del
    tutto - segnalato dal terzo elemento `has_full_stats`."""

    def test_none_when_match_is_none(self):
        self.assertEqual(DashboardService._resolve_final_stat_dicts(None), (None, None, False))

    def test_none_when_match_has_no_statistics_and_no_score(self):
        match = Match(id_match_fk=str(uuid.uuid4()), id_fixture=1, id_team_home=10, id_team_away=20)
        match.statistics = []
        self.assertEqual(DashboardService._resolve_final_stat_dicts(match), (None, None, False))

    def test_maps_home_away_by_team_id_regardless_of_list_order(self):
        match = Match(id_match_fk=str(uuid.uuid4()), id_fixture=1, id_team_home=10, id_team_away=20)
        stat_home = Statistics(
            id_statistics_fk=str(uuid.uuid4()), statistics_team_id=10, score_ft=2, corners=5, yellow_cards=1
        )
        stat_away = Statistics(
            id_statistics_fk=str(uuid.uuid4()), statistics_team_id=20, score_ft=1, corners=4, yellow_cards=2
        )
        match.statistics = [stat_away, stat_home]  # ordine invertito apposta

        home_dict, away_dict, has_full_stats = DashboardService._resolve_final_stat_dicts(match)

        self.assertTrue(has_full_stats)
        self.assertEqual(home_dict["score_ft"], 2)
        self.assertEqual(home_dict["corners"], 5)
        self.assertEqual(away_dict["score_ft"], 1)
        self.assertEqual(away_dict["corners"], 4)

    def test_falls_back_to_score_only_when_no_statistics_but_match_score_present(self):
        """Il gap reale segnalato dall'operatore (2026-09-10): leghe minori
        dove l'endpoint statistiche dedicato non ha dati, ma il punteggio
        finale e' comunque noto dalla risposta 'fixtures' leggera."""
        match = Match(
            id_match_fk=str(uuid.uuid4()), id_fixture=1, id_team_home=10, id_team_away=20,
            score_home=2, score_away=1,
        )
        match.statistics = []

        home_dict, away_dict, has_full_stats = DashboardService._resolve_final_stat_dicts(match)

        self.assertFalse(has_full_stats)
        self.assertEqual(home_dict, {"score_ft": 2})
        self.assertEqual(away_dict, {"score_ft": 1})

    def test_no_fallback_when_only_one_side_of_score_is_known(self):
        match = Match(
            id_match_fk=str(uuid.uuid4()), id_fixture=1, id_team_home=10, id_team_away=20,
            score_home=2, score_away=None,
        )
        match.statistics = []

        self.assertEqual(DashboardService._resolve_final_stat_dicts(match), (None, None, False))


class TestAnnotatePredictionCorrectness(unittest.TestCase):
    """`_annotate_prediction_correctness` (2026-09-10, richiesto
    esplicitamente dall'operatore: "quando una partita e' finita, colorami
    di verde le odds prese e in rosso quelle non prese")."""

    def test_marks_correct_when_prediction_matches_real_result(self):
        predictions = {"h2h": {"prediction": 1, "probability": 0.7}}  # prevista vittoria home
        DashboardService._annotate_prediction_correctness(predictions, {"score_ft": 2}, {"score_ft": 1})
        self.assertTrue(predictions["h2h"]["correct"])

    def test_marks_wrong_when_prediction_does_not_match_real_result(self):
        predictions = {"h2h": {"prediction": 1, "probability": 0.7}}  # prevista vittoria home
        DashboardService._annotate_prediction_correctness(predictions, {"score_ft": 0}, {"score_ft": 2})
        self.assertFalse(predictions["h2h"]["correct"])

    def test_none_when_result_not_determinable(self):
        predictions = {"h2h": {"prediction": 1, "probability": 0.7}}
        DashboardService._annotate_prediction_correctness(predictions, None, None)
        self.assertIsNone(predictions["h2h"]["correct"])

    def test_each_market_evaluated_independently(self):
        predictions = {
            "h2h": {"prediction": 1, "probability": 0.7},  # home vince -> corretto
            "under_over_2_5": {"prediction": 0, "probability": 0.6},  # under, ma 3 gol totali -> sbagliato
        }
        DashboardService._annotate_prediction_correctness(predictions, {"score_ft": 2}, {"score_ft": 1})
        self.assertTrue(predictions["h2h"]["correct"])
        self.assertFalse(predictions["under_over_2_5"]["correct"])

    def test_double_chance_1x_is_correct_on_draw(self):
        predictions = {"dc": {"prediction": 1, "probability": 0.7}}  # prevista 1X
        DashboardService._annotate_prediction_correctness(
            predictions, {"score_ft": 2}, {"score_ft": 2}
        )
        self.assertTrue(predictions["dc"]["correct"])

    def test_double_chance_1x_is_wrong_on_away_win(self):
        predictions = {"dc": {"prediction": 1, "probability": 0.7}}  # prevista 1X
        DashboardService._annotate_prediction_correctness(
            predictions, {"score_ft": 1}, {"score_ft": 2}
        )
        self.assertFalse(predictions["dc"]["correct"])

    def test_double_chance_away_is_correct_on_away_win(self):
        predictions = {"dc": {"prediction": 0, "probability": 0.7}}  # prevista vittoria ospite
        DashboardService._annotate_prediction_correctness(
            predictions, {"score_ft": 0}, {"score_ft": 1}
        )
        self.assertTrue(predictions["dc"]["correct"])

    def test_has_full_stats_false_still_resolves_score_only_markets(self):
        predictions = {"h2h": {"prediction": 1, "probability": 0.7}}
        DashboardService._annotate_prediction_correctness(
            predictions, {"score_ft": 2}, {"score_ft": 1}, has_full_stats=False
        )
        self.assertTrue(predictions["h2h"]["correct"])

    def test_has_full_stats_false_never_resolves_corners_or_cards(self):
        """Un dict 'solo punteggio' non ha ne' 'corners' ne'
        'yellow_cards'/'red_cards' - senza questo guard, `_label_by_market`
        li leggerebbe come 0 (FALSO, non sconosciuto)."""
        predictions = {
            "corners": {"prediction": 1, "probability": 0.6},
            "cards": {"prediction": 0, "probability": 0.55},
        }
        DashboardService._annotate_prediction_correctness(
            predictions, {"score_ft": 2}, {"score_ft": 1}, has_full_stats=False
        )
        self.assertIsNone(predictions["corners"]["correct"])
        self.assertIsNone(predictions["cards"]["correct"])


class TestSerializeRowsAnnotateCorrectnessOnlyWhenFinished(unittest.TestCase):
    """Integrazione: `_serialize_match`/`_serialize_api_fixture` devono
    aggiungere `correct` SOLO per partite concluse - mai per NS/live (nessun
    risultato reale su cui basarsi)."""

    def test_serialize_match_uses_score_only_fallback_when_no_statistics(self):
        """Il gap reale segnalato dall'operatore (2026-09-10, leghe minori
        senza statistiche dettagliate): `score` in riga E `correct` per h2h
        devono comunque risolversi dal solo `Match.score_home`/`score_away`."""
        service = DashboardService()
        match = Match(
            id_match_fk=str(uuid.uuid4()),
            id_fixture=1,
            date_match="2026-09-01T18:00:00+00:00",
            status="FT",
            id_team_home=10,
            id_team_away=20,
            name_home="Alcione",
            name_away="Treviso",
            score_home=1,
            score_away=0,
        )
        match.statistics = []
        match.odds = []
        service._predict_fixture = lambda **kwargs: {"h2h": {"prediction": 1, "probability": 0.7}}

        row = service._serialize_match(match, with_predictions=True, markets=["h2h"])

        self.assertEqual(row["score"], {"home": 1, "away": 0})
        self.assertTrue(row["predictions"]["h2h"]["correct"])

    def test_extract_actual_totals_sums_cards_and_corners(self):
        match = Match(
            id_match_fk=str(uuid.uuid4()), id_fixture=1, id_team_home=10, id_team_away=20,
        )
        match.statistics = [
            Statistics(statistics_team_id=10, yellow_cards=2, red_cards=1, corners=4),
            Statistics(statistics_team_id=20, yellow_cards=3, red_cards=0, corners=6),
        ]

        self.assertEqual(
            DashboardService._extract_actual_totals(match),
            {"yellow_cards": 5, "red_cards": 1, "cards": 6, "corners": 10},
        )

    def test_serialize_match_adds_correct_for_finished_match(self):
        service = DashboardService()
        match = Match(
            id_match_fk=str(uuid.uuid4()),
            id_fixture=1,
            date_match="2026-09-01T18:00:00+00:00",
            status="FT",
            id_team_home=10,
            id_team_away=20,
            name_home="Inter",
            name_away="Milan",
        )
        match.statistics = [
            Statistics(id_statistics_fk=str(uuid.uuid4()), statistics_team_id=10, score_ft=2),
            Statistics(id_statistics_fk=str(uuid.uuid4()), statistics_team_id=20, score_ft=1),
        ]
        match.odds = []
        service._predict_fixture = lambda **kwargs: {"h2h": {"prediction": 1, "probability": 0.7}}

        row = service._serialize_match(match, with_predictions=True, markets=["h2h"])

        self.assertTrue(row["predictions"]["h2h"]["correct"])

    def test_serialize_match_no_correct_key_when_not_finished(self):
        service = DashboardService()
        match = Match(
            id_match_fk=str(uuid.uuid4()),
            id_fixture=1,
            date_match="2099-09-01T18:00:00+00:00",
            status="NS",
            id_team_home=10,
            id_team_away=20,
        )
        match.statistics = []
        match.odds = []
        service._predict_fixture = lambda **kwargs: {"h2h": {"prediction": 1, "probability": 0.7}}

        row = service._serialize_match(match, with_predictions=True, markets=["h2h"])

        self.assertNotIn("correct", row["predictions"]["h2h"])

    def test_serialize_api_fixture_adds_correct_for_finished_match_using_db_match_stats(self):
        service = DashboardService()
        fixture = {
            "fixture": {"id": 1, "date": "2026-09-01T18:45:00+00:00", "status": {"short": "FT"}},
            "league": {"id": 135, "name": "Serie A"},
            "teams": {"home": {"name": "Inter"}, "away": {"name": "Milan"}},
            "goals": {"home": 2, "away": 1},
        }
        db_match = Match(id_match_fk=str(uuid.uuid4()), id_fixture=1, id_team_home=10, id_team_away=20)
        db_match.statistics = [
            Statistics(id_statistics_fk=str(uuid.uuid4()), statistics_team_id=10, score_ft=2),
            Statistics(id_statistics_fk=str(uuid.uuid4()), statistics_team_id=20, score_ft=1),
        ]
        service._predict_fixture = lambda **kwargs: {"h2h": {"prediction": 1, "probability": 0.7}}

        row = service._serialize_api_fixture(fixture, with_predictions=True, markets=["h2h"], db_match=db_match)

        self.assertTrue(row["predictions"]["h2h"]["correct"])

    def test_serialize_api_fixture_correct_is_none_without_db_match(self):
        service = DashboardService()
        fixture = {
            "fixture": {"id": 1, "date": "2026-09-01T18:45:00+00:00", "status": {"short": "FT"}},
            "league": {"id": 135, "name": "Serie A"},
            "teams": {"home": {"name": "Inter"}, "away": {"name": "Milan"}},
            "goals": {"home": 2, "away": 1},
        }
        service._predict_fixture = lambda **kwargs: {"h2h": {"prediction": 1, "probability": 0.7}}

        row = service._serialize_api_fixture(fixture, with_predictions=True, markets=["h2h"], db_match=None)

        self.assertIsNone(row["predictions"]["h2h"]["correct"])


if __name__ == "__main__":
    unittest.main()





