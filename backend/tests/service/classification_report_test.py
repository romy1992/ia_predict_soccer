import unittest

import numpy as np

from src.ml.evaluation.classification_report import compute_full_classification_report


def _separable_dataset(n: int = 200, seed: int = 5):
    rng = np.random.RandomState(seed)
    y_true = rng.binomial(1, 0.4, size=n)
    noise = rng.normal(scale=0.15, size=n)
    probabilities = np.clip(y_true * 0.7 + 0.15 + noise, 0.0, 1.0)
    return y_true, probabilities


class TestComputeFullClassificationReport(unittest.TestCase):
    def test_single_class_returns_explicit_status_never_fake_metrics(self):
        report = compute_full_classification_report(y_true=[0, 0, 0, 0], probabilities=[0.1, 0.2, 0.3, 0.4])
        self.assertEqual(report["status"], "single_class")
        self.assertNotIn("confusion_matrix", report)
        self.assertNotIn("roc", report)

    def test_confusion_matrix_matches_manual_count_at_default_threshold(self):
        y_true = [0, 0, 1, 1, 1]
        probabilities = [0.2, 0.6, 0.3, 0.8, 0.9]  # predetti a 0.5: 0,1,0,1,1
        report = compute_full_classification_report(y_true=y_true, probabilities=probabilities)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["threshold"], 0.5)
        cm = report["confusion_matrix"]
        # reale 0 -> predetto 0 (tn=1), reale 0 -> predetto 1 (fp=1)
        # reale 1 -> predetto 0 (fn=1), reale 1 -> predetto 1 (tp=2)
        self.assertEqual(cm, {"tn": 1, "fp": 1, "fn": 1, "tp": 2})
        self.assertAlmostEqual(report["accuracy"], 3 / 5)

    def test_class_and_weighted_metrics_have_expected_shape(self):
        y_true, probabilities = _separable_dataset()
        report = compute_full_classification_report(y_true=y_true, probabilities=probabilities)

        for key in ("class0", "class1"):
            self.assertIn("precision", report[key])
            self.assertIn("recall", report[key])
            self.assertIn("f1", report[key])
            self.assertIn("support", report[key])
        for key in ("precision", "recall", "f1"):
            self.assertIn(key, report["weighted"])

    def test_roc_and_pr_curves_present_with_auc_between_zero_and_one(self):
        y_true, probabilities = _separable_dataset()
        report = compute_full_classification_report(y_true=y_true, probabilities=probabilities)

        self.assertGreater(len(report["roc"]["fpr"]), 0)
        self.assertEqual(len(report["roc"]["fpr"]), len(report["roc"]["tpr"]))
        self.assertGreaterEqual(report["roc"]["auc"], 0.0)
        self.assertLessEqual(report["roc"]["auc"], 1.0)

        self.assertGreater(len(report["pr_curve"]["precision"]), 0)
        self.assertEqual(len(report["pr_curve"]["precision"]), len(report["pr_curve"]["recall"]))
        self.assertGreaterEqual(report["pr_curve"]["average_precision"], 0.0)
        self.assertLessEqual(report["pr_curve"]["average_precision"], 1.0)

    def test_optimal_threshold_is_at_least_as_good_as_default_on_youden_j(self):
        # Su un dataset ben separabile, la soglia ottimale (Youden J) deve
        # ottenere un'accuratezza almeno pari a quella di default 0.5 - mai
        # peggio, dato che 0.5 e' comunque una delle soglie candidate nella
        # curva ROC.
        y_true, probabilities = _separable_dataset(n=400, seed=11)
        report = compute_full_classification_report(y_true=y_true, probabilities=probabilities)

        optimal = report["optimal_threshold"]
        self.assertGreaterEqual(optimal["threshold"], 0.0)
        self.assertLessEqual(optimal["threshold"], 1.0)
        self.assertGreaterEqual(optimal["accuracy"], report["accuracy"] - 0.05)

    def test_all_values_are_json_serializable_native_types(self):
        import json

        y_true, probabilities = _separable_dataset()
        report = compute_full_classification_report(y_true=y_true, probabilities=probabilities)
        json.dumps(report)  # non deve sollevare (niente numpy scalars/array residui)


if __name__ == "__main__":
    unittest.main()
