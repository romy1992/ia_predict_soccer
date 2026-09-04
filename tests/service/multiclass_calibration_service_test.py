import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ml.calibration.multiclass_calibration_service import MulticlassCalibrationService
from src.ml.validation.temporal_split import expanding_window_splits

CLASS_LABELS = ("HOME", "DRAW", "AWAY")


class TestMulticlassCalibrationService(unittest.TestCase):
    def _synthetic_frame(self, n=260, seed=7):
        rng = np.random.RandomState(seed)
        strength_diff = rng.normal(size=n)
        noise = rng.normal(scale=0.6, size=n)
        signal = strength_diff + noise

        y = np.where(signal > 0.6, "HOME", np.where(signal < -0.6, "AWAY", "DRAW"))
        X = pd.DataFrame({"strength_diff": strength_diff, "noise_feature": rng.normal(size=n)})
        time_frame = pd.DataFrame({"prediction_at": pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")})
        return X, pd.Series(y), time_frame

    def test_select_method_returns_sigmoid_for_small_samples(self):
        _, y, _ = self._synthetic_frame(n=60)
        method = MulticlassCalibrationService.select_method(y=y, class_labels=CLASS_LABELS)
        self.assertEqual(method, "sigmoid")

    def test_calibrate_estimator_produces_pre_post_metrics_and_working_calibrator(self):
        X, y, time_frame = self._synthetic_frame(n=260)
        splits = expanding_window_splits(
            frame=time_frame, time_col="prediction_at", n_splits=4, min_train_size=140, min_valid_size=25
        )
        self.assertGreaterEqual(len(splits), 2)

        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        result = MulticlassCalibrationService.calibrate_estimator(
            estimator=model, X=X, y=y, cv_splits=splits, class_labels=CLASS_LABELS
        )

        self.assertIn(result.method, {"sigmoid", "isotonic"})
        self.assertIn("log_loss", result.pre_metrics)
        self.assertIn("log_loss", result.post_metrics)
        self.assertEqual(sum(result.class_counts.values()), len(y))
        self.assertTrue(hasattr(result.calibrator, "predict_proba"))

        proba = result.calibrator.predict_proba(X.iloc[:5])
        self.assertEqual(proba.shape[1], 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_calibrate_estimator_raises_on_empty_dataset(self):
        with self.assertRaises(ValueError):
            MulticlassCalibrationService.calibrate_estimator(
                estimator=LogisticRegression(),
                X=pd.DataFrame(),
                y=pd.Series(dtype=object),
                cv_splits=[],
                class_labels=CLASS_LABELS,
            )


if __name__ == "__main__":
    unittest.main()
