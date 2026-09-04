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


if __name__ == "__main__":
    unittest.main()
