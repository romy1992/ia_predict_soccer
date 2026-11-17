"""Campi di trasparenza aggiuntivi sul serving (2026-11).

- `markets_missing_reasons`: perche' un mercato registrato non ha previsione;
- `data_quality`: quali input pre-partita esistono davvero (quote, medie
  statistiche, xG);
- `computed_at`: istante reale di calcolo della previsione (snapshot), non
  l'ora della richiesta.

Sono tutti campi in piu': `markets_unavailable` e le chiavi esistenti di ogni
previsione non cambiano."""

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.dashboard_service import DashboardService
from src.ml.serving.prediction_snapshot_service import PredictionSnapshotService
from src.service_ia.model.match import Base, MatchPredictionSnapshot
from src.storage import bucket_store


def _service() -> DashboardService:
    return DashboardService.__new__(DashboardService)


def _match(odds=None, mean_statistics=None, id_team_home=1, id_team_away=2):
    odds_rows = [SimpleNamespace(to_dict=lambda: odds)] if odds is not None else []
    return SimpleNamespace(
        odds=odds_rows, mean_statistics=mean_statistics, id_team_home=id_team_home, id_team_away=id_team_away
    )


class TestMarketsMissingReasons(unittest.TestCase):
    def test_nessun_mercato_mancante(self):
        reasons = _service()._markets_missing_reasons(_match(odds={"h2h": {"a": 1}}), ["h2h"], {"h2h": {}})
        self.assertEqual(reasons, {})

    def test_partita_non_nel_db(self):
        reasons = _service()._markets_missing_reasons(None, ["h2h", "cards"], {})
        self.assertEqual(reasons, {"h2h": "match_not_in_db", "cards": "match_not_in_db"})

    def test_partita_senza_alcuna_riga_quote(self):
        reasons = _service()._markets_missing_reasons(_match(odds=None), ["h2h"], {})
        self.assertEqual(reasons, {"h2h": "no_odds_row"})

    def test_distingue_quote_assenti_per_mercato_da_previsione_non_calcolata(self):
        match = _match(odds={"h2h": {"home_Bet365": "1.8"}, "cards": None, "corners": {}})
        reasons = _service()._markets_missing_reasons(
            match, ["h2h", "cards_line_4_5", "corners_line_9_5", "goal_no_goal"], {}
        )
        self.assertEqual(
            reasons,
            {
                "h2h": "not_computed",
                "cards_line_4_5": "no_odds_for_market",
                "corners_line_9_5": "no_odds_for_market",
                "goal_no_goal": "no_odds_for_market",
            },
        )

    def test_coerente_con_markets_unavailable(self):
        """Ogni mercato in `markets_unavailable` ha motivo `no_odds_for_market`."""
        service = _service()
        match = _match(odds={"h2h": {"home_Bet365": "1.8"}, "cards": None})
        markets = ["h2h", "cards_line_4_5"]
        unavailable = service._markets_without_odds(match, markets, {})
        reasons = service._markets_missing_reasons(match, markets, {})
        self.assertEqual(unavailable, ["cards_line_4_5"])
        self.assertEqual({m for m, r in reasons.items() if r == "no_odds_for_market"}, set(unavailable))


class TestRowDataQuality(unittest.TestCase):
    def test_partita_non_nel_db_non_dichiara_nulla(self):
        self.assertIsNone(DashboardService._row_data_quality(None))

    def test_tutto_disponibile(self):
        match = _match(
            odds={"h2h": {}},
            mean_statistics=[
                {"id_team": 1, "mean_expected_goals": 1.4},
                {"id_team": 2, "mean_expected_goals": 0.9},
            ],
        )
        self.assertEqual(
            DashboardService._row_data_quality(match),
            {"odds_available": True, "team_statistics_available": True, "xg_available": True},
        )

    def test_xg_mancante_per_una_squadra(self):
        match = _match(
            odds={"h2h": {}},
            mean_statistics=[
                {"id_team": 1, "mean_expected_goals": 1.4},
                {"id_team": 2, "mean_expected_goals": None},
            ],
        )
        quality = DashboardService._row_data_quality(match)
        self.assertTrue(quality["team_statistics_available"])
        self.assertFalse(quality["xg_available"])

    def test_xg_non_numerico_o_nan_conta_come_mancante(self):
        for value in ("", "n/a", float("nan")):
            match = _match(
                odds=None,
                mean_statistics=[
                    {"id_team": 1, "mean_expected_goals": 1.0},
                    {"id_team": 2, "mean_expected_goals": value},
                ],
            )
            quality = DashboardService._row_data_quality(match)
            self.assertFalse(quality["xg_available"], value)
            self.assertFalse(quality["odds_available"])

    def test_xg_a_zero_conta_come_mancante(self):
        """`calculate_mean` azzera l'xG assente prima di fare la media: "mai
        rilevato" arriva come 0.0, non come None. Dichiararlo disponibile
        sarebbe falso proprio nel caso che il campo deve segnalare."""
        for value in (0, 0.0, "0", "0.0"):
            match = _match(
                odds={"h2h": {}},
                mean_statistics=[
                    {"id_team": 1, "mean_expected_goals": 1.3},
                    {"id_team": 2, "mean_expected_goals": value},
                ],
            )
            quality = DashboardService._row_data_quality(match)
            self.assertTrue(quality["team_statistics_available"], value)
            self.assertFalse(quality["xg_available"], value)

    def test_medie_di_squadre_diverse_da_quelle_in_campo_non_contano(self):
        """Stessa risoluzione del serving (`_resolve_mean_stats`, per id
        squadra): se le medie non sono di casa/trasferta il modello non riceve
        alcuna feature `mean_*`."""
        from src.service_ia.training.market_service.filter_market_service import FilterMarketService

        means = [{"id_team": 7, "mean_expected_goals": 1.4}, {"id_team": 8, "mean_expected_goals": 0.9}]
        match = _match(odds={"h2h": {}}, mean_statistics=means, id_team_home=1, id_team_away=2)
        self.assertEqual(
            DashboardService._row_data_quality(match),
            {"odds_available": True, "team_statistics_available": False, "xg_available": False},
        )
        self.assertEqual(
            FilterMarketService._extract_mean_features(
                {"mean_statistics": means, "id_team_home": 1, "id_team_away": 2}
            ),
            {},
        )

    def test_coerente_con_le_feature_che_il_serving_costruisce(self):
        """`team_statistics_available` vero <=> `_extract_mean_features` produce
        le feature `mean_*`; xG a zero <=> feature xG a 0.0."""
        from src.service_ia.training.market_service.filter_market_service import FilterMarketService

        means = [
            {"id_team": 1, "mean_expected_goals": 0.0, "mean_Fouls": 11.0},
            {"id_team": 2, "mean_expected_goals": 1.1, "mean_Fouls": 9.0},
        ]
        match = _match(odds={"h2h": {}}, mean_statistics=means)
        quality = DashboardService._row_data_quality(match)
        features = FilterMarketService._extract_mean_features(
            {"mean_statistics": means, "id_team_home": 1, "id_team_away": 2}
        )
        self.assertTrue(quality["team_statistics_available"])
        self.assertFalse(quality["xg_available"])
        self.assertEqual(features["mean_expected_goals_home_stat"], 0.0)
        self.assertEqual(features["mean_fouls_home_stat"], 11.0)

    def test_json_delle_medie_malformato_non_solleva(self):
        match = _match(odds={"h2h": {}}, mean_statistics=["non-un-dict", 3])
        self.assertEqual(
            DashboardService._row_data_quality(match),
            {"odds_available": True, "team_statistics_available": False, "xg_available": False},
        )

    def test_medie_statistiche_assenti(self):
        for means in (None, [], [{"id_team": 1, "mean_expected_goals": 1.0}], "non-una-lista"):
            quality = DashboardService._row_data_quality(_match(odds={"h2h": {}}, mean_statistics=means))
            self.assertEqual(
                quality,
                {"odds_available": True, "team_statistics_available": False, "xg_available": False},
                means,
            )


class _FakeFilterService:
    def __init__(self, frames):
        self._frames = frames

    def build_prediction_frames(self, fixture_id, markets):
        return {m: self._frames[m] for m in markets if m in self._frames}


class _FakeModel:
    def predict_proba(self, X):
        return [[0.3, 0.7]]


class TestComputedAt(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        patcher = mock.patch(
            "src.repository.match_prediction_snapshot_repository.SessionLocal",
            new=sessionmaker(bind=engine, expire_on_commit=False),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        PredictionSnapshotService.clear_model_cache()
        self.addCleanup(PredictionSnapshotService.clear_model_cache)

    def test_snapshot_esistente_riporta_il_suo_istante_di_calcolo(self):
        snapshot = MatchPredictionSnapshot(
            fixture_id=1,
            market="h2h",
            prediction=1,
            probability=0.75,
            model_name="logistic",
            model_run_id="run_old",
            feature_fingerprint="x",
            computed_at=datetime(2026, 9, 9, 7, 30, tzinfo=timezone.utc),
        )
        entry = PredictionSnapshotService._entry_from_snapshot(snapshot)
        self.assertEqual(entry["computed_at"], "2026-09-09T07:30:00+00:00")
        # Le chiavi storiche restano identiche.
        self.assertEqual(
            {k: entry[k] for k in ("prediction", "probability", "model_name", "run_id")},
            {"prediction": 1, "probability": 0.75, "model_name": "logistic", "run_id": "run_old"},
        )

    def test_timestamp_naive_del_db_viene_dichiarato_utc(self):
        snapshot = SimpleNamespace(
            prediction=0, probability=0.4, model_name="rf", model_run_id="r", computed_at=datetime(2026, 9, 9, 7, 30)
        )
        self.assertEqual(
            PredictionSnapshotService._entry_from_snapshot(snapshot)["computed_at"], "2026-09-09T07:30:00+00:00"
        )

    def test_senza_istante_non_se_ne_inventa_uno(self):
        snapshot = SimpleNamespace(prediction=0, probability=0.4, model_name="rf", model_run_id="r", computed_at=None)
        self.assertIsNone(PredictionSnapshotService._entry_from_snapshot(snapshot)["computed_at"])

    def test_calcolo_nuovo_espone_lo_stesso_istante_salvato_sullo_snapshot(self):
        model_path = "best_models/h2h/h2h_champion_test.pkl"
        bucket_store.put_json(model_path, {"placeholder": True})
        frame = pd.DataFrame(
            [{"id_fixture": 7, "season": 2026, "league": 39, "market": "h2h", "prediction_at": "2026-09-09", "odds_avg": 1.8}]
        )
        service = PredictionSnapshotService()
        service.filter_service = _FakeFilterService({"h2h": frame})
        service.registry = mock.Mock()
        service.registry.get_production = lambda market: {
            "model_path": model_path, "model_name": "logistic", "run_id": "run1", "feature_names": []
        }
        service.registry.get_latest = lambda market: None
        with mock.patch.object(PredictionSnapshotService, "_load_model", classmethod(lambda cls, path: _FakeModel())):
            before = datetime.now(timezone.utc)
            payload = service.resolve_predictions(fixture_id=7, markets=["h2h"], status="NS")
            after = datetime.now(timezone.utc)

        computed_at = datetime.fromisoformat(payload["h2h"]["computed_at"])
        self.assertTrue(before <= computed_at <= after)
        saved = service.repo.get_latest(fixture_id=7, market="h2h")
        self.assertEqual(PredictionSnapshotService._iso_utc(saved.computed_at), payload["h2h"]["computed_at"])


class TestMotivoMancanzaModello(unittest.TestCase):
    def setUp(self):
        DashboardService._model_reason_cache.clear()
        self.addCleanup(DashboardService._model_reason_cache.clear)

    def _snapshot_service(self, production=None, latest=None):
        svc = PredictionSnapshotService.__new__(PredictionSnapshotService)
        svc.registry = mock.Mock()
        svc.registry.get_production = lambda market: production
        svc.registry.get_latest = lambda market: latest
        return svc

    def test_senza_modello_nel_registry(self):
        self.assertEqual(self._snapshot_service().explain_missing_market("h2h"), "no_model_registered")

    def test_modello_ritirato(self):
        svc = self._snapshot_service(latest={"current_stage": "retired", "model_path": "x"})
        self.assertEqual(svc.explain_missing_market("corners"), "model_retired")

    def test_file_modello_mancante(self):
        svc = self._snapshot_service(production={"model_path": "best_models/nonesiste.pkl"})
        self.assertEqual(svc.explain_missing_market("h2h"), "model_file_missing")

    def test_modello_presente_nessun_motivo(self):
        path = "best_models/h2h/trasparenza_test.pkl"
        bucket_store.put_json(path, {"placeholder": True})
        svc = self._snapshot_service(production={"model_path": path})
        self.assertIsNone(svc.explain_missing_market("h2h"))

    def test_dashboard_raffina_not_computed_con_il_motivo_del_registry(self):
        service = _service()
        service._snapshot_service = mock.Mock()
        service._snapshot_service.explain_missing_market = lambda m: "no_model_registered" if m == "goal_no_goal" else None
        match = _match(odds={"h2h": {"home_Bet365": "1.8"}, "goal_no_goal": {"yes": "1.7"}})
        reasons = service._markets_missing_reasons(match, ["h2h", "goal_no_goal"], {})
        self.assertEqual(reasons, {"h2h": "not_computed", "goal_no_goal": "no_model_registered"})

    def test_il_motivo_del_modello_viene_prima_di_quote_e_partita(self):
        """Senza un modello utilizzabile la previsione non puo' comparire,
        qualunque cosa succeda a quote e dati: e' quello il motivo da dire,
        anche quando mancano pure le quote o la partita non e' nel DB."""
        service = _service()
        service._snapshot_service = mock.Mock()
        motivi = {"corners": "model_retired", "cards": "no_model_registered", "dc": "model_file_missing"}
        service._snapshot_service.explain_missing_market = lambda m: motivi.get(m)
        mercati = ["h2h", "corners", "cards", "dc"]
        attesi_modello = {"corners": "model_retired", "cards": "no_model_registered", "dc": "model_file_missing"}

        senza_quote_per_mercato = _match(odds={"h2h": {"home_Bet365": "1.8"}, "corners": None, "cards": {}, "dc": None})
        self.assertEqual(
            service._markets_missing_reasons(senza_quote_per_mercato, mercati, {}),
            {"h2h": "not_computed", **attesi_modello},
        )
        self.assertEqual(
            service._markets_missing_reasons(_match(odds=None), mercati, {}),
            {"h2h": "no_odds_row", **attesi_modello},
        )
        self.assertEqual(
            service._markets_missing_reasons(None, mercati, {}),
            {"h2h": "match_not_in_db", **attesi_modello},
        )

    def test_markets_unavailable_resta_invariato_qualunque_sia_il_motivo(self):
        """Il campo storico (badge "In coda") dipende solo dalle quote."""
        service = _service()
        service._snapshot_service = mock.Mock()
        service._snapshot_service.explain_missing_market = lambda m: "model_retired"
        match = _match(odds={"h2h": {"home_Bet365": "1.8"}, "cards": None})
        self.assertEqual(service._markets_without_odds(match, ["h2h", "cards_line_4_5"], {}), ["cards_line_4_5"])

    def test_feature_riempite_con_zero_sono_dichiarate_e_probabilita_invariata(self):
        model_path = "best_models/h2h/trasparenza_feat_test.pkl"
        bucket_store.put_json(model_path, {"placeholder": True})
        frame = pd.DataFrame(
            [{"id_fixture": 9, "season": 2026, "league": 39, "market": "h2h", "prediction_at": "2026-09-09", "odds_avg": 1.8}]
        )
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        seen = {}

        class _Model:
            def predict_proba(self, X):
                seen["X"] = X.copy()
                return [[0.3, 0.7]]

        with mock.patch(
            "src.repository.match_prediction_snapshot_repository.SessionLocal",
            new=sessionmaker(bind=engine, expire_on_commit=False),
        ):
            service = PredictionSnapshotService()
            service.filter_service = _FakeFilterService({"h2h": frame})
            service.registry = mock.Mock()
            service.registry.get_production = lambda market: {
                "model_path": model_path,
                "model_name": "logistic",
                "run_id": "r",
                "feature_names": ["odds_avg", "mean_expected_goals_home_stat", "mean_expected_goals_away_stat"],
            }
            service.registry.get_latest = lambda market: None
            with mock.patch.object(PredictionSnapshotService, "_load_model", classmethod(lambda cls, path: _Model())):
                payload = service.resolve_predictions(fixture_id=9, markets=["h2h"], status="NS")
        entry = payload["h2h"]
        self.assertEqual(entry["features_defaulted"], ["mean_expected_goals_home_stat", "mean_expected_goals_away_stat"])
        self.assertEqual(entry["features_defaulted_count"], 2)
        self.assertEqual(entry["probability"], 0.7)
        self.assertEqual(float(seen["X"]["mean_expected_goals_home_stat"].iloc[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
