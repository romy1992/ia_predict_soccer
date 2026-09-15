import unittest

import numpy as np
import pandas as pd

from src.service_ia.training.train_multi_market import _extract_selected_features, _model_space


class TestTrainMultiMarketPipeline(unittest.TestCase):
    def test_model_space_includes_selector_in_pipeline(self):
        feature_count = 12
        space = _model_space(selection_method="kbest", feature_count=feature_count)

        logistic_pipeline, logistic_grid = space["logistic"]
        rf_pipeline, rf_grid = space["random_forest"]

        self.assertIn("imputer", logistic_pipeline.named_steps)
        self.assertIn("selector", logistic_pipeline.named_steps)
        self.assertIn("scaler", logistic_pipeline.named_steps)
        self.assertIn("model", logistic_pipeline.named_steps)

        self.assertIn("imputer", rf_pipeline.named_steps)
        self.assertIn("selector", rf_pipeline.named_steps)
        self.assertNotIn("scaler", rf_pipeline.named_steps)
        self.assertIn("model", rf_pipeline.named_steps)

        self.assertTrue(any(key.startswith("selector__") for key in logistic_grid.keys()))
        self.assertTrue(any(key.startswith("selector__") for key in rf_grid.keys()))

    def test_selected_features_are_extracted_from_fitted_pipeline(self):
        feature_count = 10
        space = _model_space(selection_method="kbest", feature_count=feature_count)
        pipeline, _ = space["logistic"]

        X = pd.DataFrame(np.random.RandomState(42).randn(60, feature_count), columns=[f"f{i}" for i in range(feature_count)])
        y = pd.Series([0, 1] * 30)

        pipeline.fit(X, y)
        selected = _extract_selected_features(pipeline, X.columns.tolist())

        self.assertGreaterEqual(len(selected), 1)
        self.assertLessEqual(len(selected), feature_count)

    def test_model_space_includes_voting_candidates_and_smote(self):
        space = _model_space(selection_method="kbest", feature_count=8)
        self.assertIn("logistic", space)
        self.assertIn("random_forest", space)
        self.assertIn("random_forest_smote", space)

    def test_random_search_selects_a_champion_on_tiny_frame(self):
        from src.service_ia.training.train_multi_market import _select_champion_via_model_search

        rng = np.random.RandomState(42)
        n = 80
        X = pd.DataFrame(rng.randn(n, 6), columns=[f"f{i}" for i in range(6)])
        y = pd.Series(([0, 1] * (n // 2)))
        season = pd.Series(["2024"] * n)
        league = pd.Series(["39"] * n)
        # Due fold temporali finti ma validi (entrambe le classi in train/valid).
        splits = [
            (list(range(0, 40)), list(range(40, 60))),
            (list(range(0, 60)), list(range(60, 80))),
        ]
        result = _select_champion_via_model_search(
            X=X,
            y=y,
            cv_splits=splits,
            market="h2h",
            season_series=season,
            league_series=league,
            selection_method="kbest",
            search_strategy="random",
            random_search_iter=2,
        )
        self.assertIn(result.champion_name, {"logistic", "random_forest", "random_forest_smote", "voting", "stacking"})
        self.assertIsNotNone(result.champion_estimator)
        self.assertGreaterEqual(len(result.model_results), 3)


if __name__ == "__main__":
    unittest.main()
