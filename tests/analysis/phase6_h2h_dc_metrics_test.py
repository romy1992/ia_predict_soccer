"""Metriche della fase 6 (h2h/dc): majority baseline e flag di degenerazione
senza toccare il DB."""
import unittest

import numpy as np

from scripts.analysis.phase6_h2h_dc_retrain_and_cascade import score_oof


class TestPhase6ScoreOof(unittest.TestCase):
    def test_majority_classifier_is_flagged_degenerate(self):
        y = np.array([1] * 90 + [0] * 10)
        p = np.full(100, 0.9)
        metrics = score_oof(y, p)
        self.assertGreater(metrics["accuracy"], 0.85)
        self.assertLess(abs(metrics["accuracy_minus_majority_pp"]), 0.01)
        self.assertTrue(metrics["degenerate"])
        self.assertLess(metrics["minority_recall"], 0.25)
        self.assertEqual(metrics["confusion_matrix"]["tn"] + metrics["confusion_matrix"]["fp"], 10)
        self.assertEqual(metrics["confusion_matrix"]["fn"] + metrics["confusion_matrix"]["tp"], 90)

    def test_balanced_predictions_beat_majority_and_are_not_degenerate(self):
        y = np.array([0, 1] * 50)
        p = np.where(y == 1, 0.8, 0.2).astype(float)
        metrics = score_oof(y, p)
        self.assertGreater(metrics["accuracy_minus_majority_pp"], 40.0)
        self.assertFalse(metrics["degenerate"])
        self.assertGreaterEqual(metrics["macro_f1"], 0.9)
        self.assertGreaterEqual(metrics["minority_recall"], 0.9)
        self.assertGreater(metrics["auc"], 0.9)

    def test_bookmaker_baseline_is_attached_when_provided(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.2, 0.8, 0.3, 0.7])
        market = np.array([0.4, 0.6, 0.45, 0.55])
        metrics = score_oof(y, p, p_market=market)
        self.assertIsNotNone(metrics["bookmaker_implied_baseline"])
        self.assertIn("auc", metrics["bookmaker_implied_baseline"])
        self.assertIn("classification_report", metrics)


if __name__ == "__main__":
    unittest.main()
