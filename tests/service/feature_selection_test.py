import unittest

import pandas as pd

from src.service_ia.pre_processing.feature_selection import FeatureSelectionService


class TestFeatureSelectionService(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4, 5, 6],
                "f2": [0, 1, 0, 1, 0, 1],
                "f3": [10, 9, 8, 7, 6, 5],
                "f4": [3, 3, 3, 3, 3, 3],
            }
        )
        self.y = pd.Series([0, 0, 0, 1, 1, 1])

    def test_select_k_best(self):
        result = FeatureSelectionService.select_k_best(self.X, self.y, k=2)
        self.assertEqual(result.X_selected.shape[1], 2)
        self.assertEqual(len(result.selected_features), 2)

    def test_select_rfe(self):
        result = FeatureSelectionService.select_rfe(self.X, self.y, n_features_to_select=2)
        self.assertEqual(result.X_selected.shape[1], 2)
        self.assertEqual(len(result.selected_features), 2)


if __name__ == "__main__":
    unittest.main()

