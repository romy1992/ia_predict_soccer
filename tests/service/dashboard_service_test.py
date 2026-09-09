import unittest
import uuid
from datetime import date, time
from unittest import mock

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
        service._fetch_api_day_fixtures = lambda target_date, force_refresh=False: [
            self._fixture(1001, "2026-09-01", "NS", "Inter", "Milan"),
            self._fixture(1002, "2026-09-01", "1H", "Roma", "Lazio"),
        ]
        service._fetch_matches = lambda target_date=None, day_margin=1: []
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None: {
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
        service._fetch_api_day_fixtures = lambda target_date, force_refresh=False: []
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
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None: {
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
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None: {
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
        service._fetch_api_day_fixtures = lambda target_date, force_refresh=False: [
            self._fixture(1001, "2026-09-01", "NS", "Inter", "Milan"),
        ]
        service._fetch_matches = lambda target_date=None, day_margin=1: []
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None: {
            "h2h": {"prediction": 1, "probability": 0.72, "model_name": "logistic", "run_id": "run-test"}
        }

        payload = service.get_day_matches(target_date=date(2026, 9, 1), limit=50, with_predictions=True)

        self.assertEqual(payload.rows[0]["decision_cards"], [])
        self.assertIsNone(payload.rows[0]["best_decision"])

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

            def resolve_predictions(self, fixture_id, markets, db_match=None, status=None):
                self.calls.append(
                    {"fixture_id": fixture_id, "markets": markets, "db_match": db_match, "status": status}
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

            def resolve_predictions(self, fixture_id, markets, db_match=None, status=None):
                self.call_count += 1
                return {"h2h": {"prediction": 1, "probability": 0.6, "model_name": "m", "run_id": "r1"}}

        service._snapshot_service = _FakeSnapshotService()

        service._predict_fixture(fixture_id=123, markets=["h2h"], status="NS")
        service._predict_fixture(fixture_id=123, markets=["h2h"], status="NS")

        self.assertEqual(service._snapshot_service.call_count, 1)

    def test_get_overview_never_computes_predictions(self):
        """Fix performance: `get_overview` (usato dal frontend SOLO per i
        counts/badge di riepilogo, MAI per i valori di predizione - nessun
        componente legge `day_highlights`/`live_preview`/predictions) non
        deve piu' scatenare alcun calcolo ML/DB per le predizioni - prima
        duplicava ESATTAMENTE lo stesso lavoro gia' fatto da `/dashboard/day`
        (stessa data), chiamato in parallelo dal frontend ad ogni cambio
        giorno."""
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h"]
        service._fetch_api_day_fixtures = lambda target_date, force_refresh=False: [
            self._fixture(1001, "2099-09-01", "NS", "Inter", "Milan"),
        ]
        service._fetch_matches = lambda target_date=None, day_margin=1: []

        def _boom(*args, **kwargs):
            raise AssertionError("_predict_fixture NON deve essere chiamato da get_overview")

        service._predict_fixture = _boom

        overview = service.get_overview(target_date=date(2099, 9, 1))

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
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None: {}

        payload = service.get_day_matches(target_date=date(2026, 9, 1), limit=50, with_predictions=True)

        self.assertEqual(payload.returned, 1)
        self.assertEqual(payload.rows[0]["source"], "db")
        self.assertEqual(payload.rows[0]["home"], "Inter")

    def test_get_day_matches_falls_back_to_api_when_db_empty_for_that_date(self):
        """Data NON ancora sincronizzata (DB vuoto per quella finestra, es.
        troppo lontana nel futuro o job non ancora eseguito): l'API esterna
        resta un fallback, cosi' la Dashboard non mostra mai "vuoto" per un
        semplice ritardo di sync."""
        service = DashboardService()
        service.registry.list_markets = lambda: ["h2h"]
        call_count = {"n": 0}

        def _counting_fetch(target_date, force_refresh=False):
            call_count["n"] += 1
            return [self._fixture(1001, "2026-09-01", "NS", "Inter", "Milan")]

        service._fetch_api_day_fixtures = _counting_fetch
        service._fetch_matches = lambda target_date=None, day_margin=1: []
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None: {}

        payload = service.get_day_matches(target_date=date(2026, 9, 1), limit=50, with_predictions=True)

        self.assertEqual(call_count["n"], 1)
        self.assertEqual(payload.returned, 1)
        self.assertEqual(payload.rows[0]["source"], "api_sports")

    def test_get_day_matches_falls_back_to_api_when_db_has_only_other_dates(self):
        """Il DB ha righe nella finestra +-1gg (margine di `_fetch_matches`)
        ma NESSUNA esattamente in `target_date`: deve comunque scattare il
        fallback API (`db_has_target_date` valuta la data ESATTA, non la
        finestra allargata)."""
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
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None: {}

        service.get_day_matches(target_date=date(2026, 9, 1), limit=50, with_predictions=True)

        self.assertEqual(call_count["n"], 1)

    def test_get_day_matches_calls_api_for_future_date_even_if_db_has_data(self):
        """Le partite FUTURE restano diverse dalle storiche: anche se il DB
        ha gia' la fixture (sync quotidiano, quindi potenzialmente non piu'
        fresco di 24h), le quote possono ancora muoversi e la data/orario
        puo' essere spostato - l'API resta la fonte primaria (protetta
        comunque da cache TTL 60s e dal guard quota-esaurita, non introduce
        uno spreco ulteriore), a differenza delle date STORICHE (concluse,
        mai piu' soggette a cambiamento) dove il DB e' definitivo."""
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
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None: {}

        service.get_day_matches(target_date=date(2099, 9, 1), limit=50, with_predictions=True)

        self.assertEqual(call_count["n"], 1)

    def test_get_day_matches_calls_api_for_today_even_if_db_has_data(self):
        """Oggi (partite potenzialmente live/in corso o non ancora
        iniziate) NON e' trattato come storico: l'API resta la fonte
        primaria anche se il DB ha gia' la fixture per la data odierna."""
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
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None: {}

        service.get_day_matches(target_date=today, limit=50, with_predictions=True)

        self.assertEqual(call_count["n"], 1)

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
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None: {}

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
        service._predict_fixture = lambda fixture_id, markets, db_match=None, status=None: {}
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


if __name__ == "__main__":
    unittest.main()








