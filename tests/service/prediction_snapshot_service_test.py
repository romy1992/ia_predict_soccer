import tempfile
import unittest
from unittest import mock

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.ml.serving.prediction_snapshot_service import PredictionSnapshotService, compute_feature_fingerprint
from src.repository.match_prediction_snapshot_repository import MatchPredictionSnapshotRepository
from src.service_ia.model.match import Base, MatchPredictionSnapshot


def _make_session_factory():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


class TestComputeFeatureFingerprint(unittest.TestCase):
    def test_deterministic_for_identical_values(self):
        X = pd.DataFrame([{"a": 1.0, "b": 2.5}])
        self.assertEqual(compute_feature_fingerprint(X), compute_feature_fingerprint(X.copy()))

    def test_changes_when_a_value_changes(self):
        X1 = pd.DataFrame([{"a": 1.0, "b": 2.5}])
        X2 = pd.DataFrame([{"a": 1.0, "b": 2.6}])
        self.assertNotEqual(compute_feature_fingerprint(X1), compute_feature_fingerprint(X2))

    def test_stable_across_float_precision_noise(self):
        X1 = pd.DataFrame([{"a": 1.2345671}])
        X2 = pd.DataFrame([{"a": 1.2345669}])
        self.assertEqual(compute_feature_fingerprint(X1), compute_feature_fingerprint(X2))

    def test_independent_of_column_order(self):
        X1 = pd.DataFrame([{"a": 1.0, "b": 2.0, "c": 3.0}])
        X2 = pd.DataFrame([{"c": 3.0, "a": 1.0, "b": 2.0}])
        self.assertEqual(compute_feature_fingerprint(X1), compute_feature_fingerprint(X2))

    def test_empty_frame_returns_stable_marker(self):
        self.assertEqual(compute_feature_fingerprint(pd.DataFrame()), "empty")
        self.assertEqual(compute_feature_fingerprint(None), "empty")


class _FakeFilterService:
    def __init__(self, frames_by_market):
        self._frames = frames_by_market
        self.from_match_calls = 0
        self.query_calls = 0

    def build_prediction_frames_from_match(self, match, markets):
        self.from_match_calls += 1
        return {m: self._frames[m] for m in markets if m in self._frames}

    def build_prediction_frames(self, fixture_id, markets):
        self.query_calls += 1
        return {m: self._frames[m] for m in markets if m in self._frames}


class _ExplodingFilterService:
    """Usata per assicurarsi che il frame NON venga MAI costruito quando
    una riga persistita e' gia' sufficiente (fast path partita conclusa)."""

    def build_prediction_frames_from_match(self, match, markets):
        raise AssertionError("build_prediction_frames_from_match non doveva essere chiamato")

    def build_prediction_frames(self, fixture_id, markets):
        raise AssertionError("build_prediction_frames non doveva essere chiamato")


class _FakeModel:
    def __init__(self, probability: float):
        self.probability = probability
        self.calls = 0

    def predict_proba(self, X):
        self.calls += 1
        return [[1.0 - self.probability, self.probability]]


def _frame_for(fixture_id, market, **features):
    row = {"id_fixture": fixture_id, "season": 2026, "league": 39, "market": market, "prediction_at": "2026-09-09"}
    row.update(features)
    return pd.DataFrame([row])


class TestPredictionSnapshotServiceResolvePredictions(unittest.TestCase):
    def setUp(self):
        self.session_factory = _make_session_factory()
        self._patch = mock.patch(
            "src.repository.match_prediction_snapshot_repository.SessionLocal", new=self.session_factory
        )
        self._patch.start()
        self.addCleanup(self._patch.stop)
        PredictionSnapshotService.clear_model_cache()
        self.addCleanup(PredictionSnapshotService.clear_model_cache)

        self._tmp_model_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        self._tmp_model_file.close()
        self.model_path = self._tmp_model_file.name

    def _service(self, frames_by_market, model_meta_by_market, fake_model):
        service = PredictionSnapshotService()
        service.filter_service = _FakeFilterService(frames_by_market)
        service.registry = mock.Mock()
        service.registry.get_production = lambda market: model_meta_by_market.get(market)
        service.registry.get_latest = lambda market: None
        load_patch = mock.patch.object(
            PredictionSnapshotService, "_load_model", classmethod(lambda cls, path: fake_model)
        )
        load_patch.start()
        self.addCleanup(load_patch.stop)
        return service

    def _model_meta(self, run_id="run1", model_name="logistic"):
        return {"model_path": self.model_path, "model_name": model_name, "run_id": run_id, "feature_names": []}

    # --- Partita CONCLUSA ---

    def test_final_match_no_snapshot_computes_and_persists(self):
        frame = _frame_for(1, "h2h", odds_avg=1.8)
        fake_model = _FakeModel(0.7)
        service = self._service({"h2h": frame}, {"h2h": self._model_meta()}, fake_model)

        payload = service.resolve_predictions(fixture_id=1, markets=["h2h"], status="FT")

        self.assertEqual(payload["h2h"]["prediction"], 1)
        self.assertAlmostEqual(payload["h2h"]["probability"], 0.7)
        rows = service.repo.list_for_fixture(fixture_id=1, market="h2h")
        self.assertEqual(len(rows), 1)

    def test_final_match_with_existing_snapshot_skips_frame_and_model(self):
        service = PredictionSnapshotService()
        service.repo.save(
            MatchPredictionSnapshot(
                fixture_id=1,
                market="h2h",
                prediction=1,
                probability=0.75,
                model_name="logistic",
                model_run_id="run_old",
                feature_fingerprint="whatever",
            )
        )
        service.filter_service = _ExplodingFilterService()
        service.registry = mock.Mock()
        service.registry.get_production = lambda market: (_ for _ in ()).throw(
            AssertionError("registry non doveva essere consultato")
        )

        payload = service.resolve_predictions(fixture_id=1, markets=["h2h"], status="FT")

        self.assertEqual(payload["h2h"]["probability"], 0.75)
        self.assertEqual(payload["h2h"]["run_id"], "run_old")

    def test_final_match_stays_frozen_even_if_production_model_changed(self):
        service = PredictionSnapshotService()
        from src.service_ia.model.match import MatchPredictionSnapshot

        service.repo.save(
            MatchPredictionSnapshot(
                fixture_id=1,
                market="h2h",
                prediction=0,
                probability=0.3,
                model_name="logistic",
                model_run_id="run_old",
                feature_fingerprint="fp_old",
            )
        )
        fake_model = _FakeModel(0.9)
        service.filter_service = _ExplodingFilterService()
        service.registry = mock.Mock()
        service.registry.get_production = lambda market: self._model_meta(run_id="run_new")
        load_patch = mock.patch.object(
            PredictionSnapshotService, "_load_model", classmethod(lambda cls, path: fake_model)
        )
        load_patch.start()
        self.addCleanup(load_patch.stop)

        payload = service.resolve_predictions(fixture_id=1, markets=["h2h"], status="FT")

        self.assertAlmostEqual(payload["h2h"]["probability"], 0.3)
        self.assertEqual(payload["h2h"]["run_id"], "run_old")
        rows = service.repo.list_for_fixture(fixture_id=1, market="h2h")
        self.assertEqual(len(rows), 1)

    # --- Partita NON conclusa (NS/live) ---

    def test_ns_match_no_snapshot_computes_and_persists(self):
        frame = _frame_for(1, "h2h", odds_avg=1.8)
        fake_model = _FakeModel(0.6)
        service = self._service({"h2h": frame}, {"h2h": self._model_meta()}, fake_model)

        payload = service.resolve_predictions(fixture_id=1, markets=["h2h"], status="NS")

        self.assertAlmostEqual(payload["h2h"]["probability"], 0.6)
        self.assertEqual(fake_model.calls, 1)
        rows = service.repo.list_for_fixture(fixture_id=1, market="h2h")
        self.assertEqual(len(rows), 1)

    def test_ns_match_reuses_snapshot_when_fingerprint_and_model_unchanged(self):
        frame = _frame_for(1, "h2h", odds_avg=1.8)
        X = frame.drop(columns=["market", "id_fixture", "season", "league", "prediction_at"])
        fingerprint = compute_feature_fingerprint(X)

        service = PredictionSnapshotService()
        from src.service_ia.model.match import MatchPredictionSnapshot

        service.repo.save(
            MatchPredictionSnapshot(
                fixture_id=1,
                market="h2h",
                prediction=1,
                probability=0.65,
                model_name="logistic",
                model_run_id="run1",
                feature_fingerprint=fingerprint,
            )
        )
        fake_model = _FakeModel(0.99)  # se venisse chiamato, il risultato sarebbe diverso -> test lo rileverebbe
        service.filter_service = _FakeFilterService({"h2h": frame})
        service.registry = mock.Mock()
        service.registry.get_production = lambda market: self._model_meta(run_id="run1")
        load_patch = mock.patch.object(
            PredictionSnapshotService, "_load_model", classmethod(lambda cls, path: fake_model)
        )
        load_patch.start()
        self.addCleanup(load_patch.stop)

        payload = service.resolve_predictions(fixture_id=1, markets=["h2h"], status="NS")

        self.assertAlmostEqual(payload["h2h"]["probability"], 0.65)
        self.assertEqual(fake_model.calls, 0)
        rows = service.repo.list_for_fixture(fixture_id=1, market="h2h")
        self.assertEqual(len(rows), 1)

    def test_ns_match_recomputes_and_appends_when_feature_changes(self):
        frame = _frame_for(1, "h2h", odds_avg=1.8)
        fake_model = _FakeModel(0.55)
        service = self._service({"h2h": frame}, {"h2h": self._model_meta()}, fake_model)
        from src.service_ia.model.match import MatchPredictionSnapshot

        service.repo.save(
            MatchPredictionSnapshot(
                fixture_id=1,
                market="h2h",
                prediction=1,
                probability=0.6,
                model_name="logistic",
                model_run_id="run1",
                feature_fingerprint="stale_fingerprint",
            )
        )

        payload = service.resolve_predictions(fixture_id=1, markets=["h2h"], status="NS")

        self.assertAlmostEqual(payload["h2h"]["probability"], 0.55)
        self.assertEqual(fake_model.calls, 1)
        rows = service.repo.list_for_fixture(fixture_id=1, market="h2h")
        self.assertEqual(len(rows), 2)

    def test_ns_match_recomputes_when_production_model_run_id_changes(self):
        frame = _frame_for(1, "h2h", odds_avg=1.8)
        X = frame.drop(columns=["market", "id_fixture", "season", "league", "prediction_at"])
        fingerprint = compute_feature_fingerprint(X)

        fake_model = _FakeModel(0.8)
        service = self._service({"h2h": frame}, {"h2h": self._model_meta(run_id="run_new")}, fake_model)
        from src.service_ia.model.match import MatchPredictionSnapshot

        service.repo.save(
            MatchPredictionSnapshot(
                fixture_id=1,
                market="h2h",
                prediction=1,
                probability=0.6,
                model_name="logistic",
                model_run_id="run_old",
                feature_fingerprint=fingerprint,
            )
        )

        payload = service.resolve_predictions(fixture_id=1, markets=["h2h"], status="NS")

        self.assertAlmostEqual(payload["h2h"]["probability"], 0.8)
        self.assertEqual(fake_model.calls, 1)
        rows = service.repo.list_for_fixture(fixture_id=1, market="h2h")
        self.assertEqual(len(rows), 2)

    # --- Casi limite ---

    def test_no_markets_returns_empty_without_any_side_effects(self):
        service = PredictionSnapshotService()
        service.filter_service = _ExplodingFilterService()
        payload = service.resolve_predictions(fixture_id=1, markets=[], status="NS")
        self.assertEqual(payload, {})

    def test_market_without_production_model_is_skipped(self):
        frame = _frame_for(1, "h2h", odds_avg=1.8)
        service = self._service({"h2h": frame}, {}, _FakeModel(0.5))
        payload = service.resolve_predictions(fixture_id=1, markets=["h2h"], status="NS")
        self.assertEqual(payload, {})

    # --- allow_compute=False (2026-09-10, vista storica sempre veloce) ---

    def test_final_match_allow_compute_false_serves_only_existing_snapshot(self):
        service = PredictionSnapshotService()
        service.repo.save(
            MatchPredictionSnapshot(
                fixture_id=1,
                market="h2h",
                prediction=1,
                probability=0.75,
                model_name="logistic",
                model_run_id="run_old",
                feature_fingerprint="whatever",
            )
        )
        service.filter_service = _ExplodingFilterService()
        service.registry = mock.Mock()
        service.registry.get_production = lambda market: (_ for _ in ()).throw(
            AssertionError("registry non doveva essere consultato")
        )

        payload = service.resolve_predictions(
            fixture_id=1, markets=["h2h", "goal_no_goal"], status="FT", allow_compute=False
        )

        self.assertEqual(payload["h2h"]["probability"], 0.75)
        self.assertNotIn("goal_no_goal", payload)

    def test_final_match_allow_compute_false_without_snapshot_skips_market_entirely(self):
        service = PredictionSnapshotService()
        service.filter_service = _ExplodingFilterService()
        service.registry = mock.Mock()
        service.registry.get_production = lambda market: (_ for _ in ()).throw(
            AssertionError("registry non doveva essere consultato")
        )

        payload = service.resolve_predictions(fixture_id=1, markets=["h2h"], status="FT", allow_compute=False)

        self.assertEqual(payload, {})
        rows = service.repo.list_for_fixture(fixture_id=1, market="h2h")
        self.assertEqual(len(rows), 0)

    def test_ns_match_allow_compute_false_never_computes(self):
        """Caso raro/difensivo: una data storica non dovrebbe mai avere
        fixture NS/live, ma se capita non deve comunque mai calcolare."""
        service = PredictionSnapshotService()
        service.filter_service = _ExplodingFilterService()
        service.registry = mock.Mock()
        service.registry.get_production = lambda market: (_ for _ in ()).throw(
            AssertionError("registry non doveva essere consultato")
        )

        payload = service.resolve_predictions(fixture_id=1, markets=["h2h"], status="NS", allow_compute=False)

        self.assertEqual(payload, {})

    # --- force=True (2026-09-10, bottone "Ricalcola previsione") ---

    def test_force_on_frozen_final_match_produces_a_new_row_not_the_old_one(self):
        frame = _frame_for(1, "h2h", odds_avg=1.8)
        fake_model = _FakeModel(0.9)
        service = self._service({"h2h": frame}, {"h2h": self._model_meta(run_id="run_new")}, fake_model)
        service.repo.save(
            MatchPredictionSnapshot(
                fixture_id=1,
                market="h2h",
                prediction=0,
                probability=0.3,
                model_name="logistic",
                model_run_id="run_old",
                feature_fingerprint="fp_old",
            )
        )

        payload = service.resolve_predictions(fixture_id=1, markets=["h2h"], status="FT", force=True)

        self.assertAlmostEqual(payload["h2h"]["probability"], 0.9)
        self.assertEqual(payload["h2h"]["run_id"], "run_new")
        rows = service.repo.list_for_fixture(fixture_id=1, market="h2h")
        self.assertEqual(len(rows), 2)

    def test_force_on_ns_match_bypasses_fingerprint_reuse(self):
        frame = _frame_for(1, "h2h", odds_avg=1.8)
        X = frame.drop(columns=["market", "id_fixture", "season", "league", "prediction_at"])
        fingerprint = compute_feature_fingerprint(X)

        fake_model = _FakeModel(0.77)
        service = self._service({"h2h": frame}, {"h2h": self._model_meta(run_id="run1")}, fake_model)
        service.repo.save(
            MatchPredictionSnapshot(
                fixture_id=1,
                market="h2h",
                prediction=1,
                probability=0.65,
                model_name="logistic",
                model_run_id="run1",
                feature_fingerprint=fingerprint,  # fingerprint E modello IDENTICI: normalmente riuserebbe
            )
        )

        payload = service.resolve_predictions(fixture_id=1, markets=["h2h"], status="NS", force=True)

        self.assertAlmostEqual(payload["h2h"]["probability"], 0.77)
        self.assertEqual(fake_model.calls, 1)
        rows = service.repo.list_for_fixture(fixture_id=1, market="h2h")
        self.assertEqual(len(rows), 2)

    def test_force_ignores_allow_compute_false(self):
        """`force=True` ha priorita' su `allow_compute` (docstring): anche
        se il chiamante passa `allow_compute=False` per errore/legacy, un
        `force=True` esplicito deve comunque calcolare."""
        frame = _frame_for(1, "h2h", odds_avg=1.8)
        fake_model = _FakeModel(0.55)
        service = self._service({"h2h": frame}, {"h2h": self._model_meta()}, fake_model)

        payload = service.resolve_predictions(
            fixture_id=1, markets=["h2h"], status="FT", allow_compute=False, force=True
        )

        self.assertAlmostEqual(payload["h2h"]["probability"], 0.55)
        self.assertEqual(fake_model.calls, 1)

    def test_model_cache_is_shared_across_instances(self):
        """`_model_cache` a livello di CLASSE (2026-09-09, fix "sempre
        lentissimo"): un secondo `PredictionSnapshotService()` NON deve
        ricaricare lo stesso `model_path` da disco. A differenza degli
        altri test, qui NON sostituiamo `_load_model` (bypasserebbe la
        cache che vogliamo verificare) - mockiamo solo `joblib.load`, il
        vero I/O su disco che `_load_model` invoca SOLO al cache-miss."""
        real_model = _FakeModel(0.5)
        joblib_patch = mock.patch("joblib.load", return_value=real_model)
        mock_joblib_load = joblib_patch.start()
        self.addCleanup(joblib_patch.stop)

        frame1 = _frame_for(1, "h2h", odds_avg=1.8)
        frame2 = _frame_for(2, "h2h", odds_avg=2.1)
        meta = {"h2h": self._model_meta()}

        service1 = PredictionSnapshotService()
        service1.filter_service = _FakeFilterService({"h2h": frame1})
        service1.registry = mock.Mock()
        service1.registry.get_production = lambda market: meta.get(market)
        service1.resolve_predictions(fixture_id=1, markets=["h2h"], status="NS")

        service2 = PredictionSnapshotService()
        service2.filter_service = _FakeFilterService({"h2h": frame2})
        service2.registry = mock.Mock()
        service2.registry.get_production = lambda market: meta.get(market)
        service2.resolve_predictions(fixture_id=2, markets=["h2h"], status="NS")

        self.assertEqual(mock_joblib_load.call_count, 1)


if __name__ == "__main__":
    unittest.main()
