import os
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ml.experts.direct.direct_market_expert import DIRECT_MARKET_SPECS, DirectMarketExpert


class _FakeRegistry:
    def __init__(self, production_run=None, latest_run=None):
        self._production_run = production_run
        self._latest_run = latest_run

    def get_production(self, market):
        return self._production_run

    def get_latest(self, market):
        return self._latest_run


class TestDirectMarketExpert(unittest.TestCase):
    def _fitted_estimator(self):
        rng = np.random.RandomState(0)
        X = pd.DataFrame({"f1": rng.normal(size=100), "f2": rng.normal(size=100)})
        y = (X["f1"] > 0).astype(int)
        model = LogisticRegression().fit(X, y)
        return model, X

    def test_unknown_market_raises(self):
        model, _ = self._fitted_estimator()
        with self.assertRaises(ValueError):
            DirectMarketExpert.from_estimator(market="not_a_market", estimator=model)

    def test_estimator_without_predict_proba_raises(self):
        class Dummy:
            pass

        with self.assertRaises(TypeError):
            DirectMarketExpert.from_estimator(market="h2h", estimator=Dummy())

    def test_predict_proba_matches_manual_positive_class_extraction(self):
        model, X = self._fitted_estimator()
        expert = DirectMarketExpert.from_estimator(market="h2h", estimator=model, feature_names=["f1", "f2"])

        proba = expert.predict_proba(X)
        expected = model.predict_proba(X)[:, -1]

        np.testing.assert_allclose(proba, expected)
        self.assertTrue(np.all(proba >= 0.0) and np.all(proba <= 1.0))

    def test_predict_respects_threshold(self):
        model, X = self._fitted_estimator()
        expert = DirectMarketExpert.from_estimator(market="h2h", estimator=model, feature_names=["f1", "f2"])

        predictions_low_threshold = expert.predict(X, threshold=0.01)
        predictions_high_threshold = expert.predict(X, threshold=0.99)

        self.assertGreaterEqual(predictions_low_threshold.sum(), predictions_high_threshold.sum())

    def test_h2h_semantics_explicitly_marks_binary_not_1x2(self):
        spec = DIRECT_MARKET_SPECS["h2h"]
        self.assertEqual(spec["classification_type"], "binary")
        self.assertIn("1X2", spec["outcome_semantics"])
        self.assertIn("MARKET-01", spec["outcome_semantics"])

        model, _ = self._fitted_estimator()
        expert = DirectMarketExpert.from_estimator(market="h2h", estimator=model)
        self.assertEqual(expert.classification_type, "binary")
        self.assertIn("1X2", expert.outcome_semantics)

    def test_all_supported_markets_are_binary_today(self):
        # Acceptance: nessun mercato "direct" attuale deve auto-dichiararsi multiclasse.
        for market, spec in DIRECT_MARKET_SPECS.items():
            self.assertEqual(spec["classification_type"], "binary", msg=f"{market} non e' binario")

    def test_load_production_uses_registry_and_reads_model_path(self):
        model, _ = self._fitted_estimator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "h2h_champion.pkl")
            joblib.dump(model, model_path)

            fake_run = {
                "market": "h2h",
                "model_path": model_path,
                "feature_names": ["f1", "f2"],
                "run_id": "h2h_20260101T000000000000Z",
                "current_stage": "production",
            }
            registry = _FakeRegistry(production_run=fake_run)

            expert = DirectMarketExpert.load_production(market="h2h", registry=registry)
            self.assertEqual(expert.market, "h2h")
            self.assertEqual(expert.stage, "production")
            self.assertEqual(expert.run_id, fake_run["run_id"])

    def test_load_production_raises_lookup_error_when_missing(self):
        registry = _FakeRegistry(production_run=None)
        with self.assertRaises(LookupError):
            DirectMarketExpert.load_production(market="h2h", registry=registry)

    def test_from_run_raises_file_not_found_for_missing_model_path(self):
        fake_run = {
            "market": "h2h",
            "model_path": os.path.join(tempfile.gettempdir(), "does_not_exist_12345.pkl"),
            "feature_names": [],
            "run_id": "missing",
            "current_stage": "production",
        }
        registry = _FakeRegistry(production_run=fake_run)
        with self.assertRaises(FileNotFoundError):
            DirectMarketExpert.load_production(market="h2h", registry=registry)


if __name__ == "__main__":
    unittest.main()
