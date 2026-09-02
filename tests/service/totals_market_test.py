import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np
import pandas as pd

from src.ml.markets.totals.totals_market import (
    GOAL_BIN_LABELS,
    THRESHOLD_LABELS,
    THRESHOLDS,
    TotalsBenchmarkReport,
    benchmark_totals_approaches,
    build_totals_evaluation_frame,
    build_totals_frame_from_records,
    enforce_monotonic_over_probabilities,
    over_probabilities_from_bin_probabilities,
    run_totals_benchmark,
    run_totals_benchmark_from_db,
    score_distribution_over_probabilities,
    total_goals_to_bin,
)


def _make_match(fixture_id: int, date: datetime, home_rating: float, away_rating: float, rng: np.random.RandomState) -> dict:
    home_lambda = max(0.2, 1.6 + 0.5 * home_rating)
    away_lambda = max(0.2, 1.3 + 0.5 * away_rating)
    home_goals = int(rng.poisson(home_lambda))
    away_goals = int(rng.poisson(away_lambda))

    return {
        "id_fixture": fixture_id,
        "season": 2025,
        "status": "FT",
        "date_match": date.isoformat(),
        "current_league": 39,
        "id_team_home": 100,
        "id_team_away": 200,
        "mean_statistics": [
            {"id_team": 100, "mean_Shots on Goal": 5.0 + home_rating, "mean_Corner Kicks": 6.0},
            {"id_team": 200, "mean_Shots on Goal": 5.0 + away_rating, "mean_Corner Kicks": 4.0},
        ],
        "statistics": [
            {"statistics_team_id": 100, "score_ft": home_goals},
            {"statistics_team_id": 200, "score_ft": away_goals},
        ],
        "odds": [
            {
                "under_over_1_5": {"over_bookA": 1.35, "under_bookA": 3.1},
                "under_over_2_5": {"over_bookA": 1.9, "under_bookA": 1.9},
                "under_over_3_5": {"over_bookA": 3.2, "under_bookA": 1.3},
                "under_over_4_5": {"over_bookA": 5.5, "under_bookA": 1.12},
            }
        ],
    }


def _synthetic_matches(n: int = 300, seed: int = 11) -> list[dict]:
    rng = np.random.RandomState(seed)
    base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    matches = []
    for i in range(n):
        home_rating = float(rng.normal(scale=0.9))
        away_rating = float(rng.normal(scale=0.9))
        matches.append(_make_match(1000 + i, base_date + timedelta(days=i), home_rating, away_rating, rng))
    return matches


class TestTotalGoalsToBin(unittest.TestCase):
    def test_bin_boundaries_match_thresholds_exactly(self):
        totals = np.array([0, 1, 2, 3, 4, 5, 6, 10])
        bins = total_goals_to_bin(totals)
        np.testing.assert_array_equal(bins, [0, 0, 1, 2, 3, 4, 4, 4])

    def test_bins_are_within_expected_label_range(self):
        totals = np.arange(0, 15)
        bins = total_goals_to_bin(totals)
        self.assertTrue(set(bins.tolist()).issubset(set(GOAL_BIN_LABELS)))


class TestEnforceMonotonicOverProbabilities(unittest.TestCase):
    def test_forces_non_increasing_sequence_across_thresholds(self):
        raw = {
            "over_1_5": np.array([0.4, 0.9]),  # riga 0: viola l'ordine (< over_2_5)
            "over_2_5": np.array([0.7, 0.6]),
            "over_3_5": np.array([0.5, 0.55]),  # riga 1: viola l'ordine (> over_2_5 proiettato)
            "over_4_5": np.array([0.2, 0.1]),
        }
        result = enforce_monotonic_over_probabilities(raw)
        stacked = np.vstack([result[label] for label in THRESHOLD_LABELS])
        diffs = np.diff(stacked, axis=0)
        self.assertTrue(np.all(diffs <= 1e-9))

    def test_already_monotonic_input_is_unchanged(self):
        raw = {"over_1_5": np.array([0.9]), "over_2_5": np.array([0.6]), "over_3_5": np.array([0.3]), "over_4_5": np.array([0.1])}
        result = enforce_monotonic_over_probabilities(raw)
        for label in THRESHOLD_LABELS:
            np.testing.assert_allclose(result[label], raw[label])


class TestOverProbabilitiesFromBinProbabilities(unittest.TestCase):
    def test_matches_manual_cumulative_sum(self):
        bin_probs = np.array([[0.1, 0.2, 0.3, 0.25, 0.15]])  # ordine classi 0..4
        result = over_probabilities_from_bin_probabilities(bin_probs, bin_classes=[0, 1, 2, 3, 4])

        self.assertAlmostEqual(result["over_1_5"][0], 0.2 + 0.3 + 0.25 + 0.15, places=9)
        self.assertAlmostEqual(result["over_2_5"][0], 0.3 + 0.25 + 0.15, places=9)
        self.assertAlmostEqual(result["over_3_5"][0], 0.25 + 0.15, places=9)
        self.assertAlmostEqual(result["over_4_5"][0], 0.15, places=9)

    def test_reorders_columns_when_estimator_classes_not_sorted(self):
        # classes_ in ordine sparso: la funzione deve rimappare per indice, non per posizione.
        bin_probs = np.array([[0.15, 0.25, 0.3, 0.2, 0.1]])  # colonne nell'ordine bin_classes sotto
        result = over_probabilities_from_bin_probabilities(bin_probs, bin_classes=[4, 2, 0, 3, 1])
        # bin0=0.3, bin1=0.1, bin2=0.25, bin3=0.2, bin4=0.15
        self.assertAlmostEqual(result["over_4_5"][0], 0.15, places=9)
        self.assertAlmostEqual(result["over_1_5"][0], 0.1 + 0.25 + 0.2 + 0.15, places=9)

    def test_output_is_monotonic_by_construction(self):
        rng = np.random.RandomState(0)
        raw = rng.dirichlet(alpha=np.ones(5), size=50)
        result = over_probabilities_from_bin_probabilities(raw, bin_classes=[0, 1, 2, 3, 4])
        stacked = np.vstack([result[label] for label in THRESHOLD_LABELS])
        diffs = np.diff(stacked, axis=0)
        self.assertTrue(np.all(diffs <= 1e-9))


class TestScoreDistributionOverProbabilities(unittest.TestCase):
    def test_monotonic_across_thresholds(self):
        result = score_distribution_over_probabilities(home_lambda=1.6, away_lambda=1.3)
        values = [result[label] for label in THRESHOLD_LABELS]
        self.assertTrue(all(values[i] >= values[i + 1] for i in range(len(values) - 1)))

    def test_higher_lambda_increases_over_probability(self):
        low = score_distribution_over_probabilities(home_lambda=0.4, away_lambda=0.4)
        high = score_distribution_over_probabilities(home_lambda=2.0, away_lambda=2.0)
        self.assertGreater(high["over_2_5"], low["over_2_5"])


class TestBuildTotalsFrameFromRecords(unittest.TestCase):
    def test_target_matches_real_total_goals_not_odds(self):
        matches = _synthetic_matches(n=60)
        frame = build_totals_frame_from_records(matches)

        self.assertFalse(frame.empty)
        for _, row in frame.iterrows():
            match = next(m for m in matches if m["id_fixture"] == row["id_fixture"])
            stats = {s["statistics_team_id"]: s["score_ft"] for s in match["statistics"]}
            total = stats[100] + stats[200]
            self.assertEqual(int(row["total_goals"]), total)
            for th in THRESHOLDS:
                label = f"y_over_{str(th).replace('.', '_')}"
                self.assertEqual(int(row[label]), int(total > th))

    def test_goal_bin_consistent_with_total_goals(self):
        matches = _synthetic_matches(n=60)
        frame = build_totals_frame_from_records(matches)
        expected_bins = total_goals_to_bin(frame["total_goals"].to_numpy())
        np.testing.assert_array_equal(frame["goal_bin"].to_numpy(), expected_bins)

    def test_rows_ordered_by_prediction_at(self):
        matches = _synthetic_matches(n=40)
        frame = build_totals_frame_from_records(matches)
        ordered_dates = pd.to_datetime(frame["prediction_at"], utc=True)
        self.assertTrue((ordered_dates.diff().dropna() >= pd.Timedelta(0)).all())


class TestBuildTotalsEvaluationFrame(unittest.TestCase):
    def test_frame_has_expected_extra_columns_and_valid_probabilities(self):
        matches = _synthetic_matches(n=150)
        frame, feature_columns = build_totals_evaluation_frame(matches)

        self.assertFalse(frame.empty)
        self.assertTrue(len(feature_columns) > 0)
        for label in THRESHOLD_LABELS:
            self.assertIn(f"gd_{label}", frame.columns)
            self.assertIn(f"y_{label}", frame.columns)
            self.assertTrue(((frame[f"gd_{label}"] >= 0.0) & (frame[f"gd_{label}"] <= 1.0)).all())

        # feature_columns non deve contenere colonne meta/target.
        for col in ["id_fixture", "season", "league", "market", "prediction_at", "total_goals", "goal_bin"]:
            self.assertNotIn(col, feature_columns)

    def test_goal_distribution_probabilities_are_monotonic_per_row(self):
        matches = _synthetic_matches(n=150)
        frame, _ = build_totals_evaluation_frame(matches)
        stacked = np.vstack([frame[f"gd_{label}"].to_numpy() for label in THRESHOLD_LABELS])
        diffs = np.diff(stacked, axis=0)
        self.assertTrue(np.all(diffs <= 1e-9))

    def test_empty_when_no_matches(self):
        frame, feature_columns = build_totals_evaluation_frame(matches=[])
        self.assertTrue(frame.empty)
        self.assertEqual(feature_columns, [])


class TestBenchmarkTotalsApproaches(unittest.TestCase):
    def _prepared_frame(self, n=280, seed=13):
        matches = _synthetic_matches(n=n, seed=seed)
        frame, feature_columns = build_totals_evaluation_frame(matches)
        min_train = max(30, int(len(frame) * 0.45))
        min_valid = max(10, int(len(frame) * 0.1))
        from src.ml.validation.temporal_split import expanding_window_splits

        cv_splits = expanding_window_splits(
            frame=frame, time_col="prediction_at", n_splits=4, min_train_size=min_train, min_valid_size=min_valid
        )
        return frame, feature_columns, cv_splits

    def test_report_contains_all_approaches_and_thresholds(self):
        frame, feature_columns, cv_splits = self._prepared_frame()
        report = benchmark_totals_approaches(frame=frame, feature_columns=feature_columns, cv_splits=cv_splits)

        self.assertIsInstance(report, TotalsBenchmarkReport)
        self.assertEqual(set(report.approach_aggregate_scores.keys()), {"binary_independent", "hierarchical", "goal_distribution"})
        self.assertIn(report.best_approach, {"binary_independent", "hierarchical", "goal_distribution"})
        for label in THRESHOLD_LABELS:
            self.assertEqual(
                set(report.threshold_metrics[label].keys()), {"binary_independent", "hierarchical", "goal_distribution"}
            )

    def test_final_probabilities_are_always_monotonic_across_thresholds(self):
        """Acceptance criteria: P(O1.5)>=P(O2.5)>=P(O3.5)>=P(O4.5), SEMPRE,
        indipendentemente da quale approccio ha vinto il confronto."""
        frame, feature_columns, cv_splits = self._prepared_frame()
        report = benchmark_totals_approaches(frame=frame, feature_columns=feature_columns, cv_splits=cv_splits)

        stacked = np.vstack([report.final_probabilities[label] for label in THRESHOLD_LABELS])
        diffs = np.diff(stacked, axis=0)
        self.assertTrue(np.all(diffs <= 1e-9))
        for label in THRESHOLD_LABELS:
            self.assertTrue(((report.final_probabilities[label] >= 0.0) & (report.final_probabilities[label] <= 1.0)).all())

    def test_goal_distribution_never_shows_monotonicity_violations(self):
        frame, feature_columns, cv_splits = self._prepared_frame()
        report = benchmark_totals_approaches(frame=frame, feature_columns=feature_columns, cv_splits=cv_splits)
        self.assertEqual(report.monotonicity_violations_before_projection["goal_distribution"], 0)

    def test_raises_when_no_oof_index_available(self):
        frame, feature_columns, _ = self._prepared_frame()
        with self.assertRaises(ValueError):
            benchmark_totals_approaches(frame=frame, feature_columns=feature_columns, cv_splits=[])


class TestRunTotalsBenchmark(unittest.TestCase):
    def test_end_to_end_benchmark_without_saving(self):
        matches = _synthetic_matches(n=280)
        result = run_totals_benchmark(matches=matches, save_model=False)

        self.assertEqual(result.market, "totals")
        self.assertEqual(result.status, "benchmarked")
        self.assertGreater(result.rows, 0)
        self.assertIn(result.best_approach, {"binary_independent", "hierarchical", "goal_distribution"})
        self.assertEqual(
            set(result.details["approach_aggregate_scores"].keys()), {"binary_independent", "hierarchical", "goal_distribution"}
        )
        self.assertIsNone(result.details["run"])  # save_model=False -> nessuna registrazione

    def test_empty_matches_are_skipped_without_raising(self):
        result = run_totals_benchmark(matches=[], save_model=False)
        self.assertEqual(result.status, "skipped_no_data")
        self.assertEqual(result.rows, 0)


class TestRunTotalsBenchmarkFromDb(unittest.TestCase):
    def test_empty_db_result_is_skipped_without_raising(self):
        with mock.patch("src.ml.markets.totals.totals_market.MatchRepository") as repo_cls:
            repo_cls.return_value.search_filter.return_value = []
            result = run_totals_benchmark_from_db(save_model=False)

        self.assertEqual(result.status, "skipped_no_data")


if __name__ == "__main__":
    unittest.main()

