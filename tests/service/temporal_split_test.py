import unittest

import pandas as pd

from src.ml.validation.temporal_split import expanding_window_splits, final_holdout_split, rolling_window_splits


class TestTemporalSplit(unittest.TestCase):
    def _frame(self, rows: int = 240) -> pd.DataFrame:
        base = pd.Timestamp("2026-01-01", tz="UTC")
        return pd.DataFrame(
            {
                "prediction_at": [base + pd.Timedelta(hours=i) for i in range(rows)],
                "value": list(range(rows)),
            }
        )

    def test_expanding_window_keeps_order(self):
        frame = self._frame(240)
        splits = expanding_window_splits(frame, "prediction_at", n_splits=4, min_train_size=80, min_valid_size=20)

        self.assertGreaterEqual(len(splits), 1)
        for train_idx, valid_idx in splits:
            self.assertLess(max(train_idx), min(valid_idx))

    def test_rolling_window_size(self):
        frame = self._frame(120)
        splits = rolling_window_splits(
            frame,
            "prediction_at",
            train_window_size=40,
            valid_window_size=20,
            step_size=20,
        )

        self.assertGreaterEqual(len(splits), 1)
        for train_idx, valid_idx in splits:
            self.assertEqual(len(train_idx), 40)
            self.assertEqual(len(valid_idx), 20)

    def test_final_holdout_split(self):
        frame = self._frame(100)
        train_idx, holdout_idx = final_holdout_split(frame, "prediction_at", holdout_ratio=0.2)

        self.assertEqual(len(train_idx), 80)
        self.assertEqual(len(holdout_idx), 20)
        self.assertLess(max(train_idx), min(holdout_idx))


if __name__ == "__main__":
    unittest.main()

