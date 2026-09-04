import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ml.experts.direct.direct_market_expert import DirectMarketExpert
from src.ml.markets.btts.btts_market import (
    BTTS_OUTCOMES,
    BttsBenchmarkReport,
    build_btts_evaluation_frame,
    build_goal_no_goal_frame_from_records,
    benchmark_btts_approaches,
    btts_probabilities,
    calibrate_raw_probability_oof,
    ensemble_probability,
    run_btts_benchmark,
    run_btts_benchmark_from_db,
    score_distribution_btts_probability,
)


def _make_match(fixture_id: int, date: datetime, home_rating: float, away_rating: float, rng: np.random.RandomState) -> dict:
    home_goals = int(rng.poisson(max(0.15, 1.25 + 0.55 * (home_rating - away_rating))))
    away_goals = int(rng.poisson(max(0.15, 1.05 - 0.45 * (home_rating - away_rating))))

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
        "odds": [{"goal_no_goal": {"yes_bookA": 1.85, "no_bookA": 1.95}}],
    }


def _synthetic_matches(n: int = 220, seed: int = 7) -> list[dict]:
    rng = np.random.RandomState(seed)
    base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    matches = []
    for i in range(n):
        home_rating = float(rng.normal(scale=0.8))
        away_rating = float(rng.normal(scale=0.8))
        matches.append(_make_match(1000 + i, base_date + timedelta(days=i), home_rating, away_rating, rng))
    return matches


def _fitted_direct_expert(matches: list[dict]) -> DirectMarketExpert:
    """Direct expert 'goal_no_goal' di test, addestrato sulle VERE feature
    prodotte da `build_goal_no_goal_frame_from_records` (stessa forma usata
    in produzione da EXP-05/`train_multi_market.py`)."""
    frame = build_goal_no_goal_frame_from_records(matches)
    feature_cols = [c for c in frame.columns if c not in {"id_fixture", "season", "league", "market", "prediction_at", "y"}]
    X = frame[feature_cols]
    y = frame["y"].astype(int)
    model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
    return DirectMarketExpert.from_estimator(market="goal_no_goal", estimator=model, feature_names=feature_cols)


class TestScoreDistributionBttsProbability(unittest.TestCase):
    def test_probability_in_unit_interval(self):
        p = score_distribution_btts_probability(home_lambda=1.4, away_lambda=1.1)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_higher_lambdas_increase_btts_probability(self):
        low = score_distribution_btts_probability(home_lambda=0.3, away_lambda=0.3)
        high = score_distribution_btts_probability(home_lambda=2.5, away_lambda=2.5)
        self.assertGreater(high, low)

    def test_matches_manual_matrix_computation(self):
        from src.ml.experts.goal_distribution.goal_distribution_expert import score_distribution_matrix

        matrix = score_distribution_matrix(home_lambda=1.3, away_lambda=1.2, max_goals=10)
        expected = float(matrix.iloc[1:, 1:].to_numpy().sum())
        self.assertAlmostEqual(score_distribution_btts_probability(1.3, 1.2), expected, places=9)


class TestBttsProbabilities(unittest.TestCase):
    def test_yes_and_no_sum_to_one(self):
        for p_yes in [0.0, 0.13, 0.5, 0.77, 1.0]:
            probs = btts_probabilities(p_yes)
            self.assertEqual(set(probs.keys()), set(BTTS_OUTCOMES))
            self.assertAlmostEqual(probs["Yes"] + probs["No"], 1.0, places=9)

    def test_clips_out_of_range_inputs(self):
        self.assertEqual(btts_probabilities(-0.5)["Yes"], 0.0)
        self.assertEqual(btts_probabilities(1.5)["Yes"], 1.0)


class TestEnsembleProbability(unittest.TestCase):
    def test_weighted_average_matches_manual_formula(self):
        p_a = np.array([0.8, 0.2, 0.5])
        p_b = np.array([0.4, 0.6, 0.5])
        result = ensemble_probability(p_a, p_b, weight_a=0.25)
        expected = 0.25 * p_a + 0.75 * p_b
        np.testing.assert_allclose(result, expected)

    def test_weight_is_clipped_to_unit_interval(self):
        p_a = np.array([1.0, 0.0])
        p_b = np.array([0.0, 1.0])
        result = ensemble_probability(p_a, p_b, weight_a=5.0)
        np.testing.assert_allclose(result, p_a)


class TestCalibrateRawProbabilityOof(unittest.TestCase):
    def _synthetic_probability_series(self, n=300, seed=1):
        rng = np.random.RandomState(seed)
        raw = rng.uniform(0.0, 1.0, size=n)
        y = (rng.uniform(0.0, 1.0, size=n) < raw).astype(int)
        return raw, y

    def _temporal_splits(self, n):
        from src.ml.validation.temporal_split import expanding_window_splits

        frame = pd.DataFrame({"prediction_at": pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")})
        return expanding_window_splits(frame=frame, time_col="prediction_at", n_splits=4, min_train_size=120, min_valid_size=30)

    def test_isotonic_calibration_produces_pre_post_metrics(self):
        raw, y = self._synthetic_probability_series()
        splits = self._temporal_splits(len(raw))
        result = calibrate_raw_probability_oof(raw_probability=raw, y_true=y, cv_splits=splits, method="isotonic")

        self.assertEqual(result.method, "isotonic")
        self.assertEqual(result.sample_size, len(raw))
        self.assertIn("log_loss", result.pre_metrics)
        self.assertIn("log_loss", result.post_metrics)
        self.assertTrue(hasattr(result.calibrator, "predict"))

    def test_sigmoid_calibration_also_produces_metrics(self):
        raw, y = self._synthetic_probability_series(seed=2)
        splits = self._temporal_splits(len(raw))
        result = calibrate_raw_probability_oof(raw_probability=raw, y_true=y, cv_splits=splits, method="sigmoid")

        self.assertEqual(result.method, "sigmoid")
        self.assertIn("brier", result.post_metrics)

    def test_raises_on_empty_series(self):
        with self.assertRaises(ValueError):
            calibrate_raw_probability_oof(raw_probability=[], y_true=[], cv_splits=[])

    def test_raises_when_no_valid_fold(self):
        # Singola classe in ogni fold di training -> nessun fold valido.
        raw = np.array([0.1, 0.2, 0.3, 0.4])
        y = np.array([0, 0, 0, 0])
        with self.assertRaises(ValueError):
            calibrate_raw_probability_oof(raw_probability=raw, y_true=y, cv_splits=[([0, 1], [2, 3])])


class TestBenchmarkBttsApproaches(unittest.TestCase):
    def _dataset(self, n=300, seed=3):
        rng = np.random.RandomState(seed)
        y = rng.binomial(1, 0.5, size=n)
        # p_score_distribution: segnale debole ma reale.
        p_sd = np.clip(0.5 + 0.15 * (y - 0.5) + rng.normal(scale=0.2, size=n), 0.01, 0.99)
        # p_direct_expert: segnale piu' forte.
        p_de = np.clip(0.5 + 0.35 * (y - 0.5) + rng.normal(scale=0.15, size=n), 0.01, 0.99)
        frame = pd.DataFrame({"prediction_at": pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")})
        return y, p_sd, p_de, frame

    def test_report_contains_all_three_approaches_and_best_is_one_of_them(self):
        from src.ml.validation.temporal_split import expanding_window_splits

        y, p_sd, p_de, frame = self._dataset()
        splits = expanding_window_splits(frame=frame, time_col="prediction_at", n_splits=4, min_train_size=120, min_valid_size=30)

        report = benchmark_btts_approaches(
            y_true=y, p_score_distribution=p_sd, p_direct_expert=p_de, cv_splits=splits
        )

        self.assertIsInstance(report, BttsBenchmarkReport)
        self.assertEqual(set(report.approach_metrics.keys()), {"score_distribution", "direct_expert", "ensemble"})
        self.assertIn(report.best_approach, {"score_distribution", "direct_expert", "ensemble"})
        self.assertIsNotNone(report.calibration)

    def test_calibration_skipped_when_disabled(self):
        from src.ml.validation.temporal_split import expanding_window_splits

        y, p_sd, p_de, frame = self._dataset(seed=4)
        splits = expanding_window_splits(frame=frame, time_col="prediction_at", n_splits=4, min_train_size=120, min_valid_size=30)

        report = benchmark_btts_approaches(
            y_true=y, p_score_distribution=p_sd, p_direct_expert=p_de, cv_splits=splits, calibrate_best=False
        )
        self.assertIsNone(report.calibration)

    def test_stronger_direct_expert_signal_is_more_likely_to_win(self):
        from src.ml.validation.temporal_split import expanding_window_splits

        # p_direct_expert quasi perfetto, p_score_distribution quasi rumore puro.
        rng = np.random.RandomState(5)
        n = 300
        y = rng.binomial(1, 0.5, size=n)
        p_sd = np.clip(rng.uniform(0.4, 0.6, size=n), 0.01, 0.99)
        p_de = np.clip(np.where(y == 1, 0.9, 0.1) + rng.normal(scale=0.02, size=n), 0.01, 0.99)
        frame = pd.DataFrame({"prediction_at": pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")})
        splits = expanding_window_splits(frame=frame, time_col="prediction_at", n_splits=4, min_train_size=120, min_valid_size=30)

        report = benchmark_btts_approaches(y_true=y, p_score_distribution=p_sd, p_direct_expert=p_de, cv_splits=splits)
        self.assertEqual(report.best_approach, "direct_expert")


class TestBuildGoalNoGoalFrameFromRecords(unittest.TestCase):
    def test_target_matches_real_result_not_odds(self):
        matches = _synthetic_matches(n=60)
        frame = build_goal_no_goal_frame_from_records(matches)

        self.assertFalse(frame.empty)
        for _, row in frame.iterrows():
            match = next(m for m in matches if m["id_fixture"] == row["id_fixture"])
            stats = {s["statistics_team_id"]: s["score_ft"] for s in match["statistics"]}
            expected_y = int(stats[100] > 0 and stats[200] > 0)
            self.assertEqual(int(row["y"]), expected_y)

    def test_rows_without_goal_no_goal_odds_are_dropped(self):
        matches = _synthetic_matches(n=5)
        matches[0]["odds"] = [{"h2h": {"home_bookA": 1.9}}]
        frame = build_goal_no_goal_frame_from_records(matches)
        self.assertNotIn(matches[0]["id_fixture"], frame["id_fixture"].tolist())


class TestBuildBttsEvaluationFrame(unittest.TestCase):
    def test_frame_has_expected_columns_and_valid_probabilities(self):
        matches = _synthetic_matches(n=150)
        direct_expert = _fitted_direct_expert(matches)

        frame = build_btts_evaluation_frame(matches=matches, direct_expert=direct_expert)

        self.assertFalse(frame.empty)
        expected_cols = {"id_fixture", "season", "league", "prediction_at", "y", "p_score_distribution", "p_direct_expert"}
        self.assertEqual(expected_cols, set(frame.columns))
        self.assertTrue(((frame["p_score_distribution"] >= 0.0) & (frame["p_score_distribution"] <= 1.0)).all())
        self.assertTrue(((frame["p_direct_expert"] >= 0.0) & (frame["p_direct_expert"] <= 1.0)).all())
        self.assertTrue(set(frame["y"].unique()).issubset({0, 1}))

        ordered_dates = pd.to_datetime(frame["prediction_at"], utc=True)
        self.assertTrue((ordered_dates.diff().dropna() >= pd.Timedelta(0)).all())

    def test_empty_when_no_matches(self):
        direct_expert = mock.Mock()
        frame = build_btts_evaluation_frame(matches=[], direct_expert=direct_expert)
        self.assertTrue(frame.empty)


class TestRunBttsBenchmark(unittest.TestCase):
    def test_end_to_end_benchmark_without_saving(self):
        matches = _synthetic_matches(n=220)
        direct_expert = _fitted_direct_expert(matches)

        result = run_btts_benchmark(matches=matches, direct_expert=direct_expert, save_model=False)

        self.assertEqual(result.market, "btts")
        self.assertEqual(result.status, "benchmarked")
        self.assertGreater(result.rows, 0)
        self.assertIn(result.best_approach, {"score_distribution", "direct_expert", "ensemble"})
        self.assertEqual(set(result.details["approach_metrics"].keys()), {"score_distribution", "direct_expert", "ensemble"})
        self.assertIsNotNone(result.details["calibration"])
        self.assertIsNone(result.details["run"])  # save_model=False -> nessuna registrazione

    def test_empty_matches_are_skipped_without_raising(self):
        direct_expert = mock.Mock()
        result = run_btts_benchmark(matches=[], direct_expert=direct_expert, save_model=False)
        self.assertEqual(result.status, "skipped_no_data")
        self.assertEqual(result.rows, 0)


class TestRunBttsBenchmarkFromDb(unittest.TestCase):
    def test_raises_lookup_error_when_no_production_model_and_none_injected(self):
        with mock.patch(
            "src.ml.markets.btts.btts_market.DirectMarketExpert.load_production",
            side_effect=LookupError("nessun modello 'goal_no_goal' in produzione"),
        ):
            with self.assertRaises(LookupError):
                run_btts_benchmark_from_db(direct_expert=None)

    def test_injected_direct_expert_bypasses_load_production(self):
        fake_direct_expert = mock.Mock()
        with mock.patch(
            "src.ml.markets.btts.btts_market.DirectMarketExpert.load_production",
            side_effect=AssertionError("load_production non doveva essere chiamato"),
        ), mock.patch("src.ml.markets.btts.btts_market.MatchRepository") as repo_cls:
            repo_cls.return_value.search_filter.return_value = []
            result = run_btts_benchmark_from_db(direct_expert=fake_direct_expert, save_model=False)

        self.assertEqual(result.status, "skipped_no_data")


if __name__ == "__main__":
    unittest.main()
