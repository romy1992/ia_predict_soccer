import unittest

import pandas as pd

from src.ml.evaluation.probability_metrics import (
    champion_probability_score,
    compute_probability_metrics,
    grouped_probability_report,
)


class TestProbabilityMetrics(unittest.TestCase):
    def test_compute_metrics(self):
        y_true = [0, 1, 1, 0, 1, 0]
        probs = [0.2, 0.8, 0.7, 0.3, 0.6, 0.4]

        metrics = compute_probability_metrics(y_true=y_true, probabilities=probs, n_bins=5)

        self.assertIn("log_loss", metrics)
        self.assertIn("brier", metrics)
        self.assertIn("ece", metrics)
        self.assertIn("auc", metrics)
        self.assertEqual(metrics["sample_size"], 6)

    def test_grouped_report(self):
        frame = pd.DataFrame(
            {
                "market": ["h2h", "h2h", "h2h", "h2h"],
                "season": [2025, 2025, 2026, 2026],
                "league": [135, 135, 39, 39],
                "y": [0, 1, 1, 0],
                "probability": [0.3, 0.7, 0.8, 0.2],
            }
        )

        rows = grouped_probability_report(
            frame=frame,
            probability_col="probability",
            target_col="y",
            group_cols=["market", "season", "league"],
            n_bins=4,
        )

        self.assertGreaterEqual(len(rows), 2)
        self.assertIn("log_loss", rows[0])
        self.assertIn("brier", rows[0])

    def test_champion_probability_score(self):
        score_a = champion_probability_score({"log_loss": 0.45, "brier": 0.20, "ece": 0.05}, f1_weighted=0.60)
        score_b = champion_probability_score({"log_loss": 0.70, "brier": 0.28, "ece": 0.12}, f1_weighted=0.60)
        self.assertGreater(score_a, score_b)


if __name__ == "__main__":
    unittest.main()

