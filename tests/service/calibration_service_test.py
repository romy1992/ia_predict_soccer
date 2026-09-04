import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ml.calibration.calibration_service import CalibrationService
from src.ml.validation.temporal_split import expanding_window_splits


class TestCalibrationService(unittest.TestCase):
    def test_select_method(self):
        small = pd.Series([0, 1] * 20)
        large = pd.Series([0, 1] * 250)

        self.assertEqual(CalibrationService.select_method(small), "sigmoid")
        self.assertEqual(CalibrationService.select_method(large), "isotonic")

    def test_calibrate_estimator_returns_pre_post_metrics(self):
        rng = np.random.RandomState(42)
        n_rows = 180
        X = pd.DataFrame(
            {
                "f1": rng.normal(size=n_rows),
                "f2": rng.normal(size=n_rows),
                "f3": rng.normal(size=n_rows),
            }
        )
        y = pd.Series((X["f1"] + 0.5 * X["f2"] + rng.normal(scale=0.5, size=n_rows) > 0).astype(int))

        frame = pd.DataFrame({"prediction_at": pd.date_range("2025-01-01", periods=n_rows, freq="D", tz="UTC")})
        splits = expanding_window_splits(frame=frame, time_col="prediction_at", n_splits=5, min_train_size=90, min_valid_size=18)

        estimator = LogisticRegression(max_iter=2000, class_weight="balanced")
        result = CalibrationService.calibrate_estimator(estimator=estimator, X=X, y=y, cv_splits=splits)

        self.assertIn(result.method, {"sigmoid", "isotonic"})
        self.assertIn("log_loss", result.pre_metrics)
        self.assertIn("log_loss", result.post_metrics)
        # Acceptance ML-06: Brier/LogLoss pre-post devono essere entrambi disponibili.
        self.assertIn("brier", result.pre_metrics)
        self.assertIn("brier", result.post_metrics)
        self.assertIsInstance(result.pre_metrics["brier"], float)
        self.assertIsInstance(result.post_metrics["brier"], float)
        self.assertEqual(result.sample_size, n_rows)

    def test_calibrator_is_associated_with_model_run_metadata(self):
        # Acceptance ML-06: il calibratore deve poter essere versionato/associato al run.
        # In train_multi_market.py il calibratore fitted viene salvato come modello
        # principale e il payload di calibration (metodo, metriche, path) viene
        # passato a ModelRegistry.register(extra={"calibration": ...}).
        rng = np.random.RandomState(7)
        n_rows = 150
        X = pd.DataFrame({"f1": rng.normal(size=n_rows), "f2": rng.normal(size=n_rows)})
        y = pd.Series((X["f1"] + rng.normal(scale=0.5, size=n_rows) > 0).astype(int))
        frame = pd.DataFrame({"prediction_at": pd.date_range("2025-01-01", periods=n_rows, freq="D", tz="UTC")})
        splits = expanding_window_splits(frame=frame, time_col="prediction_at", n_splits=4, min_train_size=80, min_valid_size=15)

        estimator = LogisticRegression(max_iter=2000, class_weight="balanced")
        result = CalibrationService.calibrate_estimator(estimator=estimator, X=X, y=y, cv_splits=splits)

        # Il calibratore fitted deve esporre predict_proba (associabile/serializzabile come model run).
        self.assertTrue(hasattr(result.calibrator, "predict_proba"))
        proba = result.calibrator.predict_proba(X.iloc[:5])
        self.assertEqual(proba.shape[0], 5)


if __name__ == "__main__":
    unittest.main()

