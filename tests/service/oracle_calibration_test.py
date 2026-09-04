import unittest
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.ml.ensemble.oracle_calibration import (
    DEFAULT_MIN_CALIBRATION_SAMPLES,
    OracleEnsembleCalibrationReport,
    calibrate_oracle_ensemble,
    run_oracle_ensemble_calibration,
)
from src.ml.ensemble.stacking import LEARNED_STACKER, WEIGHTED_BLEND
from src.ml.validation.temporal_split import expanding_window_splits


def _synthetic_meta_features(n: int = 300, seed: int = 11):
    """Stesso generatore di `ensemble_stacking_test.py`: un esperto
    informativo e uno rumoroso, cosi' da avere un meta-model vincitore
    ben definito su cui applicare la calibrazione."""
    rng = np.random.RandomState(seed)
    y = rng.binomial(1, 0.5, size=n)
    strong = np.clip(np.where(y == 1, 0.85, 0.15) + rng.normal(scale=0.08, size=n), 0.01, 0.99)
    noise = np.clip(rng.uniform(0.3, 0.7, size=n), 0.01, 0.99)
    meta_features = pd.DataFrame({"expert_strong__yes": strong, "expert_noise__yes": noise})
    prediction_at = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    return meta_features, y, prediction_at


def _temporal_cv_splits(n: int, n_splits: int = 4, min_train_size: int = 120, min_valid_size: int = 30):
    frame = pd.DataFrame({"prediction_at": pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")})
    return expanding_window_splits(
        frame=frame,
        time_col="prediction_at",
        n_splits=n_splits,
        min_train_size=min_train_size,
        min_valid_size=min_valid_size,
    )


class TestCalibrateOracleEnsemble(unittest.TestCase):
    def test_calibration_applied_with_sufficient_sample(self):
        meta_features, y, _ = _synthetic_meta_features()
        splits = _temporal_cv_splits(len(meta_features))
        report = calibrate_oracle_ensemble(
            meta_features=meta_features, y_true=y, cv_splits=splits, market="btts"
        )

        self.assertIsInstance(report, OracleEnsembleCalibrationReport)
        self.assertTrue(report.calibration_applied)
        self.assertIsNone(report.fallback_reason)
        self.assertIn(report.method, {"sigmoid", "isotonic"})
        self.assertIsNotNone(report.pre_metrics)
        self.assertIsNotNone(report.post_metrics)
        self.assertIn("log_loss", report.pre_metrics)
        self.assertIn("log_loss", report.post_metrics)
        self.assertIsInstance(report.final_model, CalibratedClassifierCV)
        self.assertIn(report.stacking_report.best_approach, {WEIGHTED_BLEND, LEARNED_STACKER})

    def test_final_model_produces_valid_probabilities(self):
        meta_features, y, _ = _synthetic_meta_features()
        splits = _temporal_cv_splits(len(meta_features))
        report = calibrate_oracle_ensemble(
            meta_features=meta_features, y_true=y, cv_splits=splits, market="btts"
        )

        proba = report.final_model.predict_proba(meta_features)
        self.assertEqual(proba.shape, (len(meta_features), 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue(np.all(proba >= 0.0) and np.all(proba <= 1.0))

    def test_fallback_when_sample_size_below_threshold(self):
        meta_features, y, _ = _synthetic_meta_features()
        splits = _temporal_cv_splits(len(meta_features))
        report = calibrate_oracle_ensemble(
            meta_features=meta_features,
            y_true=y,
            cv_splits=splits,
            market="btts",
            min_calibration_samples=10_000,
        )

        self.assertFalse(report.calibration_applied)
        self.assertIsNotNone(report.fallback_reason)
        self.assertIn("sample_size", report.fallback_reason)
        self.assertIsNone(report.pre_metrics)
        self.assertIsNone(report.post_metrics)
        # Fallback: il modello finale e' il meta-model raw di ORACLE-02, non calibrato.
        self.assertIs(report.final_model, report.stacking_report.meta_model)

    def test_fallback_when_calibration_service_raises(self):
        meta_features, y, _ = _synthetic_meta_features()
        splits = _temporal_cv_splits(len(meta_features))

        with mock.patch(
            "src.ml.ensemble.oracle_calibration.CalibrationService.calibrate_estimator",
            side_effect=ValueError("boom"),
        ):
            report = calibrate_oracle_ensemble(
                meta_features=meta_features, y_true=y, cv_splits=splits, market="btts"
            )

        self.assertFalse(report.calibration_applied)
        self.assertIn("calibration_failed", report.fallback_reason)
        self.assertIs(report.final_model, report.stacking_report.meta_model)

    def test_default_min_calibration_samples_is_used(self):
        meta_features, y, _ = _synthetic_meta_features(n=DEFAULT_MIN_CALIBRATION_SAMPLES - 5)
        splits = _temporal_cv_splits(
            len(meta_features), n_splits=2, min_train_size=30, min_valid_size=10
        )
        report = calibrate_oracle_ensemble(
            meta_features=meta_features, y_true=y, cv_splits=splits, market="btts"
        )
        self.assertFalse(report.calibration_applied)
        self.assertIn("min_calibration_samples", report.fallback_reason)

    def test_raises_on_empty_meta_features(self):
        with self.assertRaises(ValueError):
            calibrate_oracle_ensemble(meta_features=pd.DataFrame(), y_true=[], cv_splits=[], market="btts")


class TestRunOracleEnsembleCalibration(unittest.TestCase):
    def test_end_to_end_without_saving(self):
        meta_features, y, prediction_at = _synthetic_meta_features()
        result = run_oracle_ensemble_calibration(
            market="btts",
            meta_features=meta_features,
            y_true=y,
            prediction_at=prediction_at,
            save_model=False,
        )
        self.assertEqual(result.market, "btts")
        self.assertEqual(result.status, "calibrated")
        self.assertTrue(result.calibration_applied)
        self.assertIn(result.best_approach, {WEIGHTED_BLEND, LEARNED_STACKER})
        self.assertIsNotNone(result.details["pre_metrics"])
        self.assertIsNotNone(result.details["post_metrics"])
        self.assertIsNone(result.details["run"])

    def test_skipped_when_no_data(self):
        result = run_oracle_ensemble_calibration(
            market="btts", meta_features=pd.DataFrame(), y_true=[], prediction_at=[], save_model=False
        )
        self.assertEqual(result.status, "skipped_no_data")
        self.assertEqual(result.rows, 0)
        self.assertIsNone(result.calibration_applied)

    def test_skipped_when_insufficient_rows_for_temporal_cv(self):
        meta_features, y, prediction_at = _synthetic_meta_features(n=20)
        result = run_oracle_ensemble_calibration(
            market="btts",
            meta_features=meta_features,
            y_true=y,
            prediction_at=prediction_at,
            save_model=False,
        )
        self.assertEqual(result.status, "skipped_insufficient_rows_for_temporal_cv")
        self.assertIsNone(result.calibration_applied)

    def test_registers_candidate_never_production(self):
        meta_features, y, prediction_at = _synthetic_meta_features()

        with mock.patch("src.ml.ensemble.oracle_calibration.ModelRegistry") as registry_cls, mock.patch(
            "src.ml.ensemble.oracle_calibration.joblib.dump"
        ) as dump_mock, mock.patch("src.ml.ensemble.oracle_calibration.os.makedirs"):
            registry_instance = registry_cls.return_value
            registry_instance.register.return_value = {"run_id": "btts_oracle_ensemble_test", "stage": "candidate"}

            result = run_oracle_ensemble_calibration(
                market="btts",
                meta_features=meta_features,
                y_true=y,
                prediction_at=prediction_at,
                save_model=True,
            )

        self.assertEqual(result.details["run"]["stage"], "candidate")
        _, register_kwargs = registry_instance.register.call_args
        self.assertEqual(register_kwargs["stage"], "candidate")
        dump_mock.assert_called_once()

    def test_saved_model_is_the_calibrated_one_when_applied(self):
        meta_features, y, prediction_at = _synthetic_meta_features()

        with mock.patch("src.ml.ensemble.oracle_calibration.ModelRegistry") as registry_cls, mock.patch(
            "src.ml.ensemble.oracle_calibration.joblib.dump"
        ) as dump_mock, mock.patch("src.ml.ensemble.oracle_calibration.os.makedirs"):
            registry_instance = registry_cls.return_value
            registry_instance.register.return_value = {"run_id": "btts_oracle_ensemble_test", "stage": "candidate"}

            result = run_oracle_ensemble_calibration(
                market="btts",
                meta_features=meta_features,
                y_true=y,
                prediction_at=prediction_at,
                save_model=True,
            )

        self.assertTrue(result.calibration_applied)
        saved_model = dump_mock.call_args[0][0]
        self.assertIsInstance(saved_model, CalibratedClassifierCV)

    def test_fallback_run_registers_raw_meta_model(self):
        meta_features, y, prediction_at = _synthetic_meta_features()

        with mock.patch(
            "src.ml.ensemble.oracle_calibration.CalibrationService.calibrate_estimator",
            side_effect=ValueError("boom"),
        ), mock.patch("src.ml.ensemble.oracle_calibration.ModelRegistry") as registry_cls, mock.patch(
            "src.ml.ensemble.oracle_calibration.joblib.dump"
        ) as dump_mock, mock.patch("src.ml.ensemble.oracle_calibration.os.makedirs"):
            registry_instance = registry_cls.return_value
            registry_instance.register.return_value = {"run_id": "btts_oracle_ensemble_fallback", "stage": "candidate"}

            result = run_oracle_ensemble_calibration(
                market="btts",
                meta_features=meta_features,
                y_true=y,
                prediction_at=prediction_at,
                save_model=True,
            )

        self.assertEqual(result.status, "calibration_fallback")
        self.assertFalse(result.calibration_applied)
        _, register_kwargs = registry_instance.register.call_args
        self.assertEqual(register_kwargs["stage"], "candidate")
        self.assertIn("raw_fallback", register_kwargs["model_name"])


if __name__ == "__main__":
    unittest.main()
