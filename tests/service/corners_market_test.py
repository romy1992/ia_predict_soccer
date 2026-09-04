import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ml.markets.corners.corners_market import (
    DEFAULT_LINES,
    CornersBenchmarkReport,
    CornersExpert,
    CornersLineTrainResult,
    build_corners_frame_from_records,
    label_corners_over,
    run_corners_benchmark,
    run_corners_benchmark_from_db,
    train_corners_all_lines,
    train_corners_line,
)


def _make_match(fixture_id: int, date: datetime, home_rating: float, away_rating: float, rng: np.random.RandomState) -> dict:
    home_corner_lambda = max(1.0, 5.5 + 1.5 * home_rating)
    away_corner_lambda = max(1.0, 4.5 + 1.5 * away_rating)
    home_corners = int(rng.poisson(home_corner_lambda))
    away_corners = int(rng.poisson(away_corner_lambda))

    return {
        "id_fixture": fixture_id,
        "season": 2025,
        "status": "FT",
        "date_match": date.isoformat(),
        "current_league": 39,
        "id_team_home": 100,
        "id_team_away": 200,
        "mean_statistics": [
            {"id_team": 100, "Corner Kicks": home_corner_lambda, "Shots on Goal": 5.0},
            {"id_team": 200, "Corner Kicks": away_corner_lambda, "Shots on Goal": 4.0},
        ],
        "statistics": [
            {"statistics_team_id": 100, "score_ft": 1, "corners": home_corners},
            {"statistics_team_id": 200, "score_ft": 1, "corners": away_corners},
        ],
        "odds": [{"corners": {"over_bookA": 1.9, "under_bookA": 1.9, "over_bookB": 1.95, "under_bookB": 1.85}}],
    }


def _synthetic_matches(n: int = 260, seed: int = 21) -> list[dict]:
    rng = np.random.RandomState(seed)
    base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    matches = []
    for i in range(n):
        home_rating = float(rng.normal(scale=0.9))
        away_rating = float(rng.normal(scale=0.9))
        matches.append(_make_match(1000 + i, base_date + timedelta(days=i), home_rating, away_rating, rng))
    return matches


class TestLabelCornersOver(unittest.TestCase):
    def test_scalar_input_returns_scalar(self):
        self.assertEqual(label_corners_over(10, line=9.5), 1)
        self.assertEqual(label_corners_over(9, line=9.5), 0)

    def test_array_input_returns_array(self):
        result = label_corners_over(np.array([8, 9, 10, 11, 12]), line=9.5)
        np.testing.assert_array_equal(result, [0, 0, 1, 1, 1])

    def test_line_is_a_free_parameter_not_hardcoded(self):
        totals = np.array([8, 9, 10, 11, 12])
        low_line = label_corners_over(totals, line=8.5)
        high_line = label_corners_over(totals, line=11.5)
        # linea piu' bassa -> piu' Over (1) rispetto a linea piu' alta.
        self.assertGreaterEqual(int(low_line.sum()), int(high_line.sum()))


class TestBuildCornersFrameFromRecords(unittest.TestCase):
    def test_target_matches_real_total_corners_for_every_line(self):
        matches = _synthetic_matches(n=60)
        frame = build_corners_frame_from_records(matches)

        self.assertFalse(frame.empty)
        for _, row in frame.iterrows():
            match = next(m for m in matches if m["id_fixture"] == row["id_fixture"])
            stats = {s["statistics_team_id"]: s["corners"] for s in match["statistics"]}
            total = stats[100] + stats[200]
            self.assertEqual(int(row["total_corners"]), total)
            for line in DEFAULT_LINES:
                label = f"y_line_{str(line).replace('.', '_')}"
                self.assertEqual(int(row[label]), int(total > line))

    def test_dedicated_corner_features_present_and_consistent(self):
        matches = _synthetic_matches(n=40)
        frame = build_corners_frame_from_records(matches)

        for col in [
            "corner_mean_home_dedicated",
            "corner_mean_away_dedicated",
            "corner_mean_total_dedicated",
            "corner_mean_diff_dedicated",
        ]:
            self.assertIn(col, frame.columns)

        np.testing.assert_allclose(
            frame["corner_mean_total_dedicated"], frame["corner_mean_home_dedicated"] + frame["corner_mean_away_dedicated"]
        )
        np.testing.assert_allclose(
            frame["corner_mean_diff_dedicated"], frame["corner_mean_home_dedicated"] - frame["corner_mean_away_dedicated"]
        )

    def test_configurable_lines_produce_different_target_columns(self):
        matches = _synthetic_matches(n=40)
        custom_lines = (7.5, 13.5)
        frame = build_corners_frame_from_records(matches, lines=custom_lines)
        self.assertIn("y_line_7_5", frame.columns)
        self.assertIn("y_line_13_5", frame.columns)
        self.assertNotIn("y_line_9_5", frame.columns)

    def test_rows_ordered_by_prediction_at(self):
        matches = _synthetic_matches(n=40)
        frame = build_corners_frame_from_records(matches)
        ordered_dates = pd.to_datetime(frame["prediction_at"], utc=True)
        self.assertTrue((ordered_dates.diff().dropna() >= pd.Timedelta(0)).all())


class TestTrainCornersLine(unittest.TestCase):
    def test_produces_calibration_result_with_pre_post_metrics(self):
        matches = _synthetic_matches(n=260)
        frame = build_corners_frame_from_records(matches)
        from src.ml.validation.temporal_split import expanding_window_splits

        cv_splits = expanding_window_splits(frame=frame, time_col="prediction_at", n_splits=4, min_train_size=120, min_valid_size=30)

        result = train_corners_line(frame=frame, line=9.5, cv_splits=cv_splits)

        self.assertIsInstance(result, CornersLineTrainResult)
        self.assertEqual(result.line, 9.5)
        self.assertIn("log_loss", result.calibration.pre_metrics)
        self.assertIn("log_loss", result.calibration.post_metrics)
        self.assertTrue(hasattr(result.calibration.calibrator, "predict_proba"))

    def test_raises_for_line_not_in_dataset(self):
        matches = _synthetic_matches(n=60)
        frame = build_corners_frame_from_records(matches, lines=(9.5,))
        with self.assertRaises(ValueError):
            train_corners_line(frame=frame, line=20.5, cv_splits=[([0], [1])])


class TestTrainCornersAllLines(unittest.TestCase):
    def test_report_has_metrics_per_line(self):
        matches = _synthetic_matches(n=260)
        frame = build_corners_frame_from_records(matches)

        report = train_corners_all_lines(frame)

        self.assertIsInstance(report, CornersBenchmarkReport)
        self.assertGreater(len(report.results), 0)
        summary = report.metrics_summary()
        for label, entry in summary.items():
            self.assertIn("pre_metrics", entry)
            self.assertIn("post_metrics", entry)
            self.assertIn("line", entry)

    def test_raises_when_dataset_too_small_for_temporal_cv(self):
        matches = _synthetic_matches(n=10)
        frame = build_corners_frame_from_records(matches)
        with self.assertRaises(ValueError):
            train_corners_all_lines(frame)


class TestCornersExpert(unittest.TestCase):
    def _fitted_expert(self, line=9.5):
        matches = _synthetic_matches(n=150)
        frame = build_corners_frame_from_records(matches)
        feature_cols = [c for c in frame.columns if not c.startswith("y_line_") and c not in {"id_fixture", "season", "league", "market", "prediction_at", "total_corners"}]
        X = frame[feature_cols]
        y = frame[f"y_line_{str(line).replace('.', '_')}"].astype(int)
        model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
        expert = CornersExpert.from_estimator(line=line, estimator=model, feature_names=feature_cols)
        return expert, X

    def test_predict_proba_dict_sums_to_one(self):
        expert, X = self._fitted_expert()
        result = expert.predict_proba_dict(X)
        np.testing.assert_allclose(result["over"] + result["under"], np.ones(len(X)))

    def test_predict_matches_threshold_on_predict_proba(self):
        expert, X = self._fitted_expert()
        proba = expert.predict_proba(X)
        prediction = expert.predict(X)
        np.testing.assert_array_equal(prediction, (proba >= 0.5).astype(int))

    def test_load_production_raises_lookup_error_when_missing(self):
        registry = mock.Mock()
        registry.get_production.return_value = None
        with self.assertRaises(LookupError):
            CornersExpert.load_production(line=9.5, registry=registry)
        registry.get_production.assert_called_once_with(market="corners_line_9_5")

    def test_load_latest_raises_lookup_error_when_missing(self):
        registry = mock.Mock()
        registry.get_latest.return_value = None
        with self.assertRaises(LookupError):
            CornersExpert.load_latest(line=10.5, registry=registry)
        registry.get_latest.assert_called_once_with(market="corners_line_10_5")


class TestRunCornersBenchmark(unittest.TestCase):
    def test_end_to_end_benchmark_without_saving(self):
        matches = _synthetic_matches(n=260)
        result = run_corners_benchmark(matches=matches, save_model=False)

        self.assertEqual(result.market, "corners")
        self.assertEqual(result.status, "benchmarked")
        self.assertGreater(result.rows, 0)
        self.assertGreater(len(result.lines_trained), 0)
        self.assertEqual(result.details["runs"], {})  # save_model=False -> nessuna registrazione

    def test_empty_matches_are_skipped_without_raising(self):
        result = run_corners_benchmark(matches=[], save_model=False)
        self.assertEqual(result.status, "skipped_no_data")
        self.assertEqual(result.rows, 0)


class TestRunCornersBenchmarkFromDb(unittest.TestCase):
    def test_empty_db_result_is_skipped_without_raising(self):
        with mock.patch("src.ml.markets.corners.corners_market.MatchRepository") as repo_cls:
            repo_cls.return_value.search_filter.return_value = []
            result = run_corners_benchmark_from_db(save_model=False)

        self.assertEqual(result.status, "skipped_no_data")


if __name__ == "__main__":
    unittest.main()
