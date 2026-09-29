import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ml.markets.market_1x2 import (
    Market1x2Expert,
    OUTCOME_LABELS,
    bookmaker_baseline_from_h2h_odds,
    build_1x2_dataset_from_records,
    label_1x2,
    train_market_1x2,
)


def _make_match(fixture_id: int, date: datetime, home_rating: float, away_rating: float, rng: np.random.RandomState) -> dict:
    home_goals = int(rng.poisson(max(0.15, 1.25 + 0.55 * (home_rating - away_rating))))
    away_goals = int(rng.poisson(max(0.15, 1.05 - 0.45 * (home_rating - away_rating))))

    diff = home_rating - away_rating
    home_odd = max(1.2, 2.2 - diff)
    away_odd = max(1.2, 2.2 + diff)

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
        "odds": [{"h2h": {"home_bookA": home_odd, "draw_bookA": 3.3, "away_bookA": away_odd}}],
    }


def _synthetic_matches(n: int = 220, seed: int = 42) -> list[dict]:
    rng = np.random.RandomState(seed)
    base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    matches = []
    for i in range(n):
        home_rating = float(rng.normal(scale=0.8))
        away_rating = float(rng.normal(scale=0.8))
        matches.append(_make_match(1000 + i, base_date + timedelta(days=i), home_rating, away_rating, rng))
    return matches


class TestLabel1x2(unittest.TestCase):
    def test_home_win(self):
        self.assertEqual(label_1x2(3, 1), "HOME")

    def test_away_win(self):
        self.assertEqual(label_1x2(0, 2), "AWAY")

    def test_draw_is_never_mapped_to_away_or_home(self):
        for goals in [0, 1, 2, 3, 4]:
            self.assertEqual(label_1x2(goals, goals), "DRAW")

    def test_missing_scores_return_none(self):
        self.assertIsNone(label_1x2(None, 1))
        self.assertIsNone(label_1x2(1, None))


class TestBookmakerBaselineFromH2hOdds(unittest.TestCase):
    def test_fair_probabilities_sum_to_one(self):
        payload = bookmaker_baseline_from_h2h_odds({"home_bookA": 2.0, "draw_bookA": 3.4, "away_bookA": 4.0})
        self.assertTrue(payload["is_exclusive"])
        self.assertAlmostEqual(payload["sum_fair_probability"], 1.0, places=6)
        outcomes = {row["outcome"] for row in payload["outcomes"]}
        self.assertEqual(outcomes, {"Home", "Draw", "Away"})

    def test_ignores_non_1x2_outcomes_and_invalid_odds(self):
        payload = bookmaker_baseline_from_h2h_odds({"home_bookA": 2.0, "draw_bookA": 0, "unknown_bookA": "n/a"})
        outcomes = {row["outcome"] for row in payload["outcomes"]}
        self.assertEqual(outcomes, {"Home"})


class TestBuild1x2Dataset(unittest.TestCase):
    def test_dataset_has_only_valid_classes_and_temporal_order(self):
        matches = _synthetic_matches(n=220)
        df = build_1x2_dataset_from_records(matches)

        self.assertFalse(df.empty)
        self.assertTrue(set(df["y"].unique()).issubset(set(OUTCOME_LABELS)))
        # Con 220 partite simulate ci aspettiamo di osservare tutti e tre gli esiti.
        self.assertEqual(set(df["y"].unique()), set(OUTCOME_LABELS))

        ordered_dates = pd.to_datetime(df["prediction_at"], utc=True)
        self.assertTrue((ordered_dates.diff().dropna() >= pd.Timedelta(0)).all())

    def test_no_odds_no_row(self):
        match = _synthetic_matches(n=1)[0]
        match["odds"] = []
        df = build_1x2_dataset_from_records([match])
        self.assertTrue(df.empty)


class TestTrainMarket1x2(unittest.TestCase):
    def test_train_end_to_end_multiclass_pipeline(self):
        df = build_1x2_dataset_from_records(_synthetic_matches(n=220))
        result = train_market_1x2(frame=df, save_model=False)

        self.assertEqual(result.status, "trained")
        self.assertEqual(result.market, "1x2")
        self.assertIsNotNone(result.champion)
        self.assertEqual(result.details["classes"], list(OUTCOME_LABELS))
        self.assertEqual(result.details["classification_type"], "multiclass")

        for model_payload in result.details["models"].values():
            self.assertIn("log_loss", model_payload["probability_metrics"])
            self.assertIn("brier", model_payload["probability_metrics"])
            self.assertIn("ece", model_payload["probability_metrics"])

        calibration = result.details["calibration"]
        self.assertTrue(calibration.get("enabled"))
        self.assertIn("pre_metrics", calibration)
        self.assertIn("post_metrics", calibration)

        baseline = result.details["bookmaker_baseline"]
        self.assertGreater(baseline["available_rows"], 0)

    def test_unexpected_labels_raise_value_error(self):
        df = build_1x2_dataset_from_records(_synthetic_matches(n=30))
        df.loc[0, "y"] = "SOMETHING_ELSE"
        with self.assertRaises(ValueError):
            train_market_1x2(frame=df, save_model=False)

    def test_empty_frame_is_skipped_without_raising(self):
        result = train_market_1x2(frame=pd.DataFrame(), save_model=False)
        self.assertEqual(result.status, "skipped_no_data")


class TestMarket1x2Expert(unittest.TestCase):
    def _fitted_multiclass_estimator(self):
        rng = np.random.RandomState(3)
        n = 150
        X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
        # Classi passate in un ordine "scomodo": sklearn le riordinera' alfabeticamente
        # (AWAY, DRAW, HOME), diverso dall'ordine canonico OUTCOME_LABELS (HOME, DRAW, AWAY).
        y = pd.Series(np.where(X["f1"] > 0.3, "HOME", np.where(X["f1"] < -0.3, "AWAY", "DRAW")))
        model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
        return model, X

    def test_predict_proba_sums_to_one_and_uses_canonical_column_order(self):
        model, X = self._fitted_multiclass_estimator()
        expert = Market1x2Expert.from_estimator(estimator=model, feature_names=["f1", "f2"])

        proba = expert.predict_proba(X)
        self.assertEqual(proba.shape, (len(X), 3))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

        # Confronto manuale: la colonna "HOME" del wrapper deve combaciare con la
        # colonna "HOME" grezza del modello, indipendentemente da come sklearn
        # ordina model.classes_ internamente.
        raw = model.predict_proba(X)
        home_col_raw = raw[:, list(model.classes_).index("HOME")]
        home_col_index = OUTCOME_LABELS.index("HOME")
        np.testing.assert_allclose(proba[:, home_col_index], home_col_raw)

    def test_predict_proba_dict_exposes_three_named_probabilities(self):
        model, X = self._fitted_multiclass_estimator()
        expert = Market1x2Expert.from_estimator(estimator=model, feature_names=["f1", "f2"])

        rows = expert.predict_proba_dict(X.iloc[:3])
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(set(row.keys()), {"HOME", "DRAW", "AWAY"})
            self.assertAlmostEqual(sum(row.values()), 1.0, places=6)

    def test_estimator_without_predict_proba_raises(self):
        class Dummy:
            pass

        with self.assertRaises(TypeError):
            Market1x2Expert.from_estimator(estimator=Dummy())


class _FakeRegistry:
    def __init__(self, production_run=None, latest_run=None):
        self._production_run = production_run
        self._latest_run = latest_run

    def get_production(self, market):
        return self._production_run

    def get_latest(self, market):
        return self._latest_run


class TestMarket1x2ExpertRegistryLifecycle(unittest.TestCase):
    def test_load_production_raises_lookup_error_when_missing(self):
        registry = _FakeRegistry(production_run=None)
        with self.assertRaises(LookupError):
            Market1x2Expert.load_production(registry=registry)

    def test_from_run_raises_file_not_found_for_missing_model_path(self):
        fake_run = {
            "market": "1x2",
            "model_path": "does_not_exist_1x2_model.pkl",
            "feature_names": [],
            "run_id": "1x2_20260101T000000000000Z",
            "current_stage": "production",
        }
        registry = _FakeRegistry(production_run=fake_run)
        with self.assertRaises(FileNotFoundError):
            Market1x2Expert.load_production(registry=registry)


if __name__ == "__main__":
    unittest.main()
