import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ml.evaluation.multiclass_probability_metrics import (
    compute_multiclass_probability_metrics,
    confidence_expected_calibration_error,
    multiclass_brier_score,
    multiclass_champion_score,
    reorder_probabilities_to_labels,
    temporal_oof_multiclass_probabilities,
)

CLASS_LABELS = ("HOME", "DRAW", "AWAY")


class TestMulticlassProbabilityMetrics(unittest.TestCase):
    def test_reorder_probabilities_to_labels_handles_non_canonical_estimator_order(self):
        # sklearn ordinerebbe le classi alfabeticamente: AWAY, DRAW, HOME.
        estimator_classes = ["AWAY", "DRAW", "HOME"]
        raw = np.array([[0.1, 0.2, 0.7]])  # P(AWAY)=0.1, P(DRAW)=0.2, P(HOME)=0.7

        ordered = reorder_probabilities_to_labels(estimator_classes, raw, CLASS_LABELS)

        # target order: HOME, DRAW, AWAY
        np.testing.assert_allclose(ordered[0], [0.7, 0.2, 0.1])

    def test_multiclass_brier_score_is_zero_for_perfect_prediction(self):
        y_idx = np.array([0, 1, 2])
        proba = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertAlmostEqual(multiclass_brier_score(y_idx, proba), 0.0, places=6)

    def test_multiclass_brier_score_is_two_for_completely_wrong_prediction(self):
        y_idx = np.array([0])
        proba = np.array([[0.0, 0.0, 1.0]])
        self.assertAlmostEqual(multiclass_brier_score(y_idx, proba), 2.0, places=6)

    def test_confidence_ece_is_low_when_confidence_matches_accuracy(self):
        y_idx = np.array([0, 0, 0, 0])
        proba = np.array([[0.9, 0.05, 0.05]] * 4)
        ece = confidence_expected_calibration_error(y_idx, proba, n_bins=10)
        self.assertLess(ece, 0.2)

    def test_compute_multiclass_probability_metrics_returns_expected_keys(self):
        y_true = ["HOME", "DRAW", "AWAY", "HOME", "AWAY", "DRAW"]
        proba = np.array(
            [
                [0.6, 0.25, 0.15],
                [0.2, 0.6, 0.2],
                [0.1, 0.2, 0.7],
                [0.55, 0.25, 0.2],
                [0.15, 0.15, 0.7],
                [0.3, 0.5, 0.2],
            ]
        )
        metrics = compute_multiclass_probability_metrics(y_true=y_true, probabilities=proba, class_labels=CLASS_LABELS)

        for key in ("log_loss", "brier", "ece", "auc_ovr_macro", "sample_size", "classes"):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["sample_size"], 6)
        self.assertIsNotNone(metrics["auc_ovr_macro"])  # tutte e 3 le classi presenti nel campione

    def test_auc_is_none_when_a_class_is_missing_from_the_sample(self):
        y_true = ["HOME", "HOME", "AWAY"]
        proba = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
        metrics = compute_multiclass_probability_metrics(y_true=y_true, probabilities=proba, class_labels=CLASS_LABELS)
        self.assertIsNone(metrics["auc_ovr_macro"])

    def test_champion_score_prefers_lower_loss_and_error(self):
        score_a = multiclass_champion_score({"log_loss": 0.7, "brier": 0.4, "ece": 0.05}, f1_weighted=0.55)
        score_b = multiclass_champion_score({"log_loss": 1.1, "brier": 0.9, "ece": 0.20}, f1_weighted=0.55)
        self.assertGreater(score_a, score_b)

    def test_temporal_oof_multiclass_probabilities_sum_to_one(self):
        rng = np.random.RandomState(0)
        n = 120
        X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
        y = pd.Series(rng.choice(CLASS_LABELS, size=n))

        cv_splits = [(list(range(0, 60)), list(range(60, 80))), (list(range(0, 80)), list(range(80, 100)))]
        model = LogisticRegression(max_iter=1000)

        oof = temporal_oof_multiclass_probabilities(estimator=model, X=X, y=y, cv_splits=cv_splits, class_labels=CLASS_LABELS)

        self.assertFalse(oof.empty)
        prob_cols = [f"prob_{label}" for label in CLASS_LABELS]
        row_sums = oof[prob_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
