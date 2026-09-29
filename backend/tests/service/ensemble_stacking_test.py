import unittest
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from src.ml.ensemble.expert_output import ExpertOutput
from src.ml.ensemble.stacking import (
    LEARNED_STACKER,
    WEIGHTED_BLEND,
    StackingBenchmarkReport,
    WeightedBlendClassifier,
    benchmark_stacking_approaches,
    build_meta_features_from_expert_outputs,
    default_stacker_estimator,
    generate_stacking_oof,
    run_stacking_benchmark,
)
from src.ml.validation.temporal_split import expanding_window_splits


def _synthetic_meta_features(n: int = 300, seed: int = 11):
    """Due 'esperti': uno informativo (`expert_strong`) e uno rumore puro
    scorrelato dal target (`expert_noise`), per verificare che sia il blend
    sia lo stacker imparino a pesare correttamente le due colonne."""
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


class TestWeightedBlendClassifier(unittest.TestCase):
    def test_weights_sum_to_one_and_are_non_negative(self):
        meta_features, y, _ = _synthetic_meta_features()
        model = WeightedBlendClassifier().fit(meta_features, y)
        self.assertAlmostEqual(float(np.sum(model.weights_)), 1.0, places=6)
        self.assertTrue(np.all(model.weights_ >= 0.0))

    def test_predict_proba_shape_and_range(self):
        meta_features, y, _ = _synthetic_meta_features()
        model = WeightedBlendClassifier().fit(meta_features, y)
        proba = model.predict_proba(meta_features)
        self.assertEqual(proba.shape, (len(meta_features), 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
        self.assertTrue(np.all(proba >= 0.0) and np.all(proba <= 1.0))

    def test_favors_more_informative_expert(self):
        meta_features, y, _ = _synthetic_meta_features()
        model = WeightedBlendClassifier().fit(meta_features, y)
        weight_strong, weight_noise = model.weights_
        self.assertGreater(weight_strong, weight_noise)

    def test_is_sklearn_cloneable(self):
        model = WeightedBlendClassifier(l2=0.1)
        cloned = clone(model)
        self.assertEqual(cloned.l2, 0.1)
        self.assertFalse(hasattr(cloned, "weights_"))

    def test_single_class_training_fold_falls_back_to_uniform_weights(self):
        X = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.9, 0.8, 0.7]})
        y = np.array([1, 1, 1])
        model = WeightedBlendClassifier().fit(X, y)
        np.testing.assert_allclose(model.weights_, [0.5, 0.5])


class TestDefaultStackerEstimator(unittest.TestCase):
    def test_returns_logistic_regression(self):
        self.assertIsInstance(default_stacker_estimator(), LogisticRegression)


class TestGenerateStackingOof(unittest.TestCase):
    def test_oof_has_expected_columns_and_length(self):
        meta_features, y, _ = _synthetic_meta_features()
        splits = _temporal_cv_splits(len(meta_features))
        oof = generate_stacking_oof(WeightedBlendClassifier(), meta_features, y, splits)
        self.assertEqual(set(oof.columns), {"index", "probability", "y_true"})
        self.assertGreater(len(oof), 0)

    def test_raises_on_empty_cv_splits(self):
        meta_features, y, _ = _synthetic_meta_features()
        with self.assertRaises(ValueError):
            generate_stacking_oof(WeightedBlendClassifier(), meta_features, y, [])

    def test_raises_on_empty_meta_features(self):
        with self.assertRaises(ValueError):
            generate_stacking_oof(WeightedBlendClassifier(), pd.DataFrame(), [], [([0], [1])])

    def test_oof_predictions_are_unaffected_by_future_rows(self):
        """Acceptance criteria 'OOF temporalmente corrette': niente leakage
        dal futuro. Modificando SOLO l'ultimo fold (training+validation),
        le OOF prodotte per i fold precedenti non devono cambiare."""
        meta_features, y, _ = _synthetic_meta_features(n=300, seed=21)
        splits = _temporal_cv_splits(len(meta_features), n_splits=4, min_train_size=120, min_valid_size=30)

        for estimator in (WeightedBlendClassifier(), LogisticRegression(max_iter=1000)):
            oof_before = generate_stacking_oof(estimator, meta_features, y, splits)

            last_valid_start = splits[-1][1][0]
            mutated = meta_features.copy()
            rng = np.random.RandomState(99)
            mutated.iloc[last_valid_start:] = rng.uniform(0.0, 1.0, size=mutated.iloc[last_valid_start:].shape)
            oof_after = generate_stacking_oof(estimator, mutated, y, splits)

            unaffected_before = oof_before[oof_before["index"] < last_valid_start].reset_index(drop=True)
            unaffected_after = oof_after[oof_after["index"] < last_valid_start].reset_index(drop=True)
            self.assertGreater(len(unaffected_before), 0)
            pd.testing.assert_frame_equal(unaffected_before, unaffected_after)


class TestBenchmarkStackingApproaches(unittest.TestCase):
    def test_report_contains_both_approaches(self):
        meta_features, y, _ = _synthetic_meta_features()
        splits = _temporal_cv_splits(len(meta_features))
        report = benchmark_stacking_approaches(meta_features=meta_features, y_true=y, cv_splits=splits)

        self.assertIsInstance(report, StackingBenchmarkReport)
        self.assertEqual(set(report.approach_metrics.keys()), {WEIGHTED_BLEND, LEARNED_STACKER})
        self.assertEqual(set(report.single_expert_metrics.keys()), set(meta_features.columns))
        self.assertIn(report.best_approach, {WEIGHTED_BLEND, LEARNED_STACKER})

    def test_meta_model_is_fitted_on_full_dataset(self):
        meta_features, y, _ = _synthetic_meta_features()
        splits = _temporal_cv_splits(len(meta_features))
        report = benchmark_stacking_approaches(meta_features=meta_features, y_true=y, cv_splits=splits)

        proba = report.meta_model.predict_proba(meta_features)
        self.assertEqual(len(proba), len(meta_features))

    def test_ensemble_outperforms_pure_noise_single_expert(self):
        meta_features, y, _ = _synthetic_meta_features()
        splits = _temporal_cv_splits(len(meta_features))
        report = benchmark_stacking_approaches(meta_features=meta_features, y_true=y, cv_splits=splits)

        best_log_loss = report.approach_metrics[report.best_approach]["log_loss"]
        noise_log_loss = report.single_expert_metrics["expert_noise__yes"]["log_loss"]
        self.assertLess(best_log_loss, noise_log_loss)

    def test_raises_on_empty_meta_features(self):
        with self.assertRaises(ValueError):
            benchmark_stacking_approaches(meta_features=pd.DataFrame(), y_true=[], cv_splits=[])


class TestBuildMetaFeaturesFromExpertOutputs(unittest.TestCase):
    def test_combines_multiple_experts_per_row(self):
        rows = [
            [
                ExpertOutput(expert_name="statistics", probability_vector={"home_win": 0.6, "not_home_win": 0.4}),
                ExpertOutput(expert_name="market_odds", probability_vector={"Home": 0.55}),
            ],
            [
                ExpertOutput(expert_name="statistics", probability_vector={"home_win": 0.3, "not_home_win": 0.7}),
                ExpertOutput(expert_name="market_odds", probability_vector={"Home": 0.4}),
            ],
        ]
        frame = build_meta_features_from_expert_outputs(rows)
        self.assertIn("statistics__home_win", frame.columns)
        self.assertIn("market_odds__Home", frame.columns)
        self.assertEqual(len(frame), 2)
        self.assertAlmostEqual(frame.loc[0, "statistics__home_win"], 0.6)

    def test_missing_expert_output_filled_with_default(self):
        rows = [
            [ExpertOutput(expert_name="statistics", probability_vector={"home_win": 0.6, "not_home_win": 0.4})],
            [
                ExpertOutput(expert_name="statistics", probability_vector={"home_win": 0.3, "not_home_win": 0.7}),
                ExpertOutput(expert_name="market_odds", probability_vector={"Home": 0.4}),
            ],
        ]
        frame = build_meta_features_from_expert_outputs(rows, fill_value=0.0)
        self.assertEqual(frame.loc[0, "market_odds__Home"], 0.0)
        self.assertFalse(frame.isna().any().any())

    def test_empty_rows_returns_empty_frame(self):
        frame = build_meta_features_from_expert_outputs([])
        self.assertTrue(frame.empty)


class TestRunStackingBenchmark(unittest.TestCase):
    def test_end_to_end_without_saving(self):
        meta_features, y, prediction_at = _synthetic_meta_features()
        result = run_stacking_benchmark(
            market="btts", meta_features=meta_features, y_true=y, prediction_at=prediction_at, save_model=False
        )
        self.assertEqual(result.market, "btts")
        self.assertEqual(result.status, "benchmarked")
        self.assertIn(result.best_approach, {WEIGHTED_BLEND, LEARNED_STACKER})
        self.assertIsNone(result.details["run"])

    def test_skipped_when_no_data(self):
        result = run_stacking_benchmark(
            market="btts", meta_features=pd.DataFrame(), y_true=[], prediction_at=[], save_model=False
        )
        self.assertEqual(result.status, "skipped_no_data")
        self.assertEqual(result.rows, 0)

    def test_skipped_when_insufficient_rows_for_temporal_cv(self):
        meta_features, y, prediction_at = _synthetic_meta_features(n=20)
        result = run_stacking_benchmark(
            market="btts", meta_features=meta_features, y_true=y, prediction_at=prediction_at, save_model=False
        )
        self.assertEqual(result.status, "skipped_insufficient_rows_for_temporal_cv")

    def test_registers_candidate_never_production(self):
        meta_features, y, prediction_at = _synthetic_meta_features()

        with mock.patch("src.ml.ensemble.stacking.ModelRegistry") as registry_cls, mock.patch(
            "src.ml.ensemble.stacking.joblib.dump"
        ) as dump_mock, mock.patch("src.ml.ensemble.stacking.os.makedirs"):
            registry_instance = registry_cls.return_value
            registry_instance.register.return_value = {"run_id": "btts_meta_test", "stage": "candidate"}

            result = run_stacking_benchmark(
                market="btts", meta_features=meta_features, y_true=y, prediction_at=prediction_at, save_model=True
            )

        self.assertEqual(result.details["run"]["stage"], "candidate")
        _, register_kwargs = registry_instance.register.call_args
        self.assertEqual(register_kwargs["stage"], "candidate")
        dump_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
