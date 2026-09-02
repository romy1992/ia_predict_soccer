import unittest

import numpy as np
import pandas as pd

from src.ml.experts.goal_distribution.goal_distribution_expert import (
    DEFAULT_THRESHOLDS,
    GoalDistributionExpert,
    compare_poisson_vs_negative_binomial,
    negative_binomial_over_probabilities,
    poisson_over_probabilities,
    score_distribution_matrix,
    over_under_from_score_matrix,
)


class TestGoalDistributionExpert(unittest.TestCase):
    def test_poisson_over_probabilities_monotone_for_various_lambda(self):
        for lam in [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
            frame = poisson_over_probabilities(lam=[lam], thresholds=DEFAULT_THRESHOLDS)
            values = frame.iloc[0].to_numpy()
            # Le probabilita' devono essere non-crescenti al crescere della soglia (1.5 -> 4.5).
            self.assertTrue(np.all(np.diff(values) <= 1e-12), msg=f"Non monotone per lam={lam}: {values}")
            self.assertTrue(np.all(values >= 0.0) and np.all(values <= 1.0))

    def test_score_matrix_sums_close_to_one(self):
        matrix = score_distribution_matrix(home_lambda=1.4, away_lambda=1.1, max_goals=12)
        total_probability = matrix.to_numpy().sum()
        self.assertGreater(total_probability, 0.999)
        self.assertLessEqual(total_probability, 1.0000001)

    def test_total_goals_distribution_matches_combined_poisson(self):
        home_lambda, away_lambda = 1.3, 1.2
        matrix = score_distribution_matrix(home_lambda=home_lambda, away_lambda=away_lambda, max_goals=15)

        from_matrix = over_under_from_score_matrix(matrix, thresholds=DEFAULT_THRESHOLDS)
        from_combined = poisson_over_probabilities(
            lam=[home_lambda + away_lambda], thresholds=DEFAULT_THRESHOLDS
        ).iloc[0].to_dict()

        for key in from_combined:
            self.assertAlmostEqual(from_matrix[key], from_combined[key], places=3)

    def test_estimate_lambdas_from_ratings_uses_multiplicative_formula(self):
        home_lambda, away_lambda = GoalDistributionExpert.estimate_lambdas_from_ratings(
            home_attack_rating=1.8,
            away_defense_rating=1.35,
            away_attack_rating=1.1,
            home_defense_rating=1.35,
            league_avg_goals=1.35,
        )
        self.assertAlmostEqual(home_lambda, 1.8 * (1.35 / 1.35), places=6)
        self.assertAlmostEqual(away_lambda, 1.1 * (1.35 / 1.35), places=6)

    def test_build_expert_output_contains_expected_keys_and_stable_version(self):
        expert_a = GoalDistributionExpert()
        expert_b = GoalDistributionExpert()
        self.assertEqual(expert_a.VERSION, expert_b.VERSION)
        self.assertEqual(GoalDistributionExpert.VERSION, expert_a.VERSION)

        output = expert_a.build_expert_output(home_lambda=1.5, away_lambda=1.2)
        self.assertIn("over_under", output)
        self.assertIn("over_under_from_score_matrix", output)
        self.assertIn("score_matrix", output)
        self.assertEqual(output["expert_version"], GoalDistributionExpert.VERSION)
        for label in ["over_1_5", "over_2_5", "over_3_5", "over_4_5"]:
            self.assertIn(label, output["over_under"])

    def test_fit_lambda_regressor_uses_temporal_validation_and_predicts(self):
        rng = np.random.RandomState(42)
        n_rows = 160
        strength_diff = rng.normal(loc=0.0, scale=1.0, size=n_rows)
        X = pd.DataFrame({"strength_diff": strength_diff, "home_advantage": rng.uniform(0, 1, size=n_rows)})
        lam_true = np.clip(2.6 + 0.4 * strength_diff, 0.2, None)
        y = pd.Series(rng.poisson(lam=lam_true))
        time_frame = pd.DataFrame({"prediction_at": pd.date_range("2025-01-01", periods=n_rows, freq="D", tz="UTC")})

        expert = GoalDistributionExpert()
        report = expert.fit_lambda_regressor(
            X=X,
            y_total_goals=y,
            time_col_frame=time_frame,
            n_splits=4,
            min_train_size=80,
            min_valid_size=15,
        )

        self.assertEqual(report["cv_strategy"], "expanding_window")
        self.assertGreater(len(report["folds"]), 0)
        self.assertIn("mean_mae", report)
        self.assertIn("mean_log_likelihood", report)

        predictions = expert.predict_lambda(X.iloc[:5])
        self.assertEqual(len(predictions), 5)
        self.assertTrue(np.all(predictions > 0))

    def test_compare_detects_overdispersion_and_recommends_negative_binomial(self):
        # Binomiale negativa con r=4, p=0.5 -> media=4, varianza=8 (overdispersion vs Poisson).
        overdispersed = np.random.RandomState(1).negative_binomial(n=4, p=0.5, size=3000).astype(float)
        report = compare_poisson_vs_negative_binomial(overdispersed)

        self.assertGreater(report.dispersion_index, 1.15)
        self.assertEqual(report.recommended_model, "negative_binomial")
        self.assertIsNotNone(report.nb_r)
        self.assertIsNotNone(report.nb_p)
        self.assertGreater(report.nb_r, 0.0)
        self.assertGreater(report.nb_p, 0.0)
        self.assertLess(report.nb_p, 1.0)

    def test_compare_recommends_poisson_for_equidispersed_data(self):
        poisson_like = np.random.RandomState(2).poisson(lam=2.0, size=3000).astype(float)
        report = compare_poisson_vs_negative_binomial(poisson_like)

        self.assertAlmostEqual(report.dispersion_index, 1.0, delta=0.15)
        self.assertEqual(report.recommended_model, "poisson")
        self.assertIsNone(report.nb_r)
        self.assertIsNone(report.nb_p)

    def test_negative_binomial_over_probabilities_monotone(self):
        probs = negative_binomial_over_probabilities(r=5.0, p=0.6, thresholds=DEFAULT_THRESHOLDS)
        values = [probs[f"over_{str(t).replace('.', '_')}"] for t in sorted(DEFAULT_THRESHOLDS)]
        self.assertTrue(all(values[i] >= values[i + 1] - 1e-12 for i in range(len(values) - 1)))


if __name__ == "__main__":
    unittest.main()

