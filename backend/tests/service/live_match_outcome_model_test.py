"""Test per LIVE-03 (addestramento/validazione modelli LIVE separati dal pre-match).

Nessun DB reale qui (a differenza di `live_match_outcome_dataset_test.py`):
un frame sintetico costruito in memoria (`_synthetic_live_frame`), stesso
stile di `_synthetic_matches` in `market_1x2_test.py`, riproduce la FORMA
esatta del dataset prodotto da `build_match_outcome_dataset` (una riga per
snapshot, target 1X2 costante per fixture) ma resta veloce/deterministico.

Il generatore assegna i minuti dei gol (home/away) in [1, 90]: lo scoreline
"parziale" di ogni snapshot e' quindi SEMPRE coerente con quello finale
(mai un gol che sparisce), e l'ultimo snapshot (FT) coincide esattamente col
risultato reale - utile anche per il test di sanity "le metriche vicino al
90' sono migliori di quelle a inizio partita" (comportamento atteso di un
modello live corretto, acceptance criteria "Probabilita' aggiornate").
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ml.live.live_match_outcome_model import (
    LiveMatchOutcomeExpert,
    match_level_temporal_splits,
    minute_window_label,
    train_live_match_outcome_model,
)
from src.ml.markets.market_1x2 import label_1x2


def _synthetic_live_frame(n_matches: int = 40, snapshots_per_match: int = 6, seed: int = 7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    base_kickoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    minutes = [10, 25, 40, 55, 70, 90][:snapshots_per_match]

    rows: list[dict] = []
    for i in range(n_matches):
        fixture_id = 5000 + i
        kickoff = base_kickoff + timedelta(hours=i)
        home_rating = float(rng.normal(scale=0.8))
        away_rating = float(rng.normal(scale=0.8))
        final_home_goals = int(rng.poisson(max(0.2, 1.3 + 0.5 * (home_rating - away_rating))))
        final_away_goals = int(rng.poisson(max(0.2, 1.0 - 0.4 * (home_rating - away_rating))))

        home_goal_minutes = sorted(rng.randint(1, 90, size=final_home_goals).tolist()) if final_home_goals else []
        away_goal_minutes = sorted(rng.randint(1, 90, size=final_away_goals).tolist()) if final_away_goals else []
        outcome = label_1x2(final_home_goals, final_away_goals)

        for idx, minute in enumerate(minutes):
            as_of = kickoff + timedelta(minutes=minute)
            home_goals_so_far = sum(1 for m in home_goal_minutes if m <= minute)
            away_goals_so_far = sum(1 for m in away_goal_minutes if m <= minute)
            is_last = idx == len(minutes) - 1
            status = "FT" if is_last else ("2H" if minute > 45 else "1H")
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "as_of": as_of.isoformat(),
                    "feature_available_at_max": as_of.isoformat(),
                    "status": status,
                    "minute": minute,
                    "minute_extra": 0,
                    "minute_total": float(minute),
                    "home_goals": home_goals_so_far,
                    "away_goals": away_goals_so_far,
                    "goal_diff": home_goals_so_far - away_goals_so_far,
                    "total_goals": home_goals_so_far + away_goals_so_far,
                    "yellow_cards_home": 0,
                    "yellow_cards_away": 0,
                    "yellow_cards_diff": 0,
                    "red_cards_home": 0,
                    "red_cards_away": 0,
                    "red_cards_diff": 0,
                    "match_time": as_of.isoformat(),
                    "y": outcome,
                }
            )

    frame = pd.DataFrame(rows)
    frame["match_time"] = pd.to_datetime(frame["match_time"], utc=True)
    return frame.sort_values(by=["match_time", "fixture_id"]).reset_index(drop=True)


class TestMinuteWindowLabel(unittest.TestCase):
    def test_final_status_is_always_ft_regardless_of_minute(self):
        self.assertEqual(minute_window_label("FT", None), "FT")
        self.assertEqual(minute_window_label("AET", 120.0), "FT")

    def test_unknown_when_minute_total_missing_and_status_not_final(self):
        self.assertEqual(minute_window_label("1H", None), "unknown")

    def test_buckets_by_minute_total(self):
        self.assertEqual(minute_window_label("1H", 10.0), "00_15")
        self.assertEqual(minute_window_label("1H", 20.0), "15_30")
        self.assertEqual(minute_window_label("1H", 44.0), "30_45")
        self.assertEqual(minute_window_label("2H", 50.0), "45_60")
        self.assertEqual(minute_window_label("2H", 70.0), "60_75")
        self.assertEqual(minute_window_label("2H", 80.0), "75_90")
        self.assertEqual(minute_window_label("2H", 95.0), "90_plus")


class TestMatchLevelTemporalSplits(unittest.TestCase):
    def test_splits_respect_match_boundaries_and_chronological_order(self):
        df = _synthetic_live_frame(n_matches=30, snapshots_per_match=4)
        splits = match_level_temporal_splits(df, n_splits=4, min_train_matches=10, min_valid_matches=3)

        self.assertGreater(len(splits), 0)
        for train_idx, valid_idx in splits:
            train_fixtures = set(df.iloc[train_idx]["fixture_id"])
            valid_fixtures = set(df.iloc[valid_idx]["fixture_id"])
            # Nessuna fixture puo' comparire sia in train che in valid nello stesso fold.
            self.assertEqual(train_fixtures & valid_fixtures, set())
            # Cronologia rispettata: tutto il training precede la validation.
            self.assertLessEqual(df.iloc[train_idx]["match_time"].max(), df.iloc[valid_idx]["match_time"].min())

    def test_empty_frame_returns_no_splits(self):
        self.assertEqual(match_level_temporal_splits(pd.DataFrame()), [])

    def test_missing_required_columns_returns_no_splits(self):
        self.assertEqual(match_level_temporal_splits(pd.DataFrame({"other": [1, 2]})), [])


class TestTrainLiveMatchOutcomeModel(unittest.TestCase):
    def test_train_end_to_end_with_match_level_split_and_minute_window_report(self):
        df = _synthetic_live_frame(n_matches=40)
        result = train_live_match_outcome_model(
            frame=df, save_model=False, n_splits=3, min_train_matches=15, min_valid_matches=5
        )

        self.assertEqual(result.status, "trained")
        self.assertEqual(result.market, "1x2_live")
        self.assertIsNotNone(result.champion)
        self.assertGreater(result.matches, 0)
        self.assertEqual(result.details["cv_strategy"], "match_level_expanding_window")
        self.assertGreater(len(result.minute_window_metrics), 0)

        windows = {row["window"] for row in result.minute_window_metrics}
        self.assertIn("FT", windows)
        self.assertIn("00_15", windows)

        # Sanity "probabilita' aggiornate": vicino al 90'/FT lo scoreline e'
        # (quasi) quello finale -> il log_loss atteso e' nettamente migliore
        # rispetto all'inizio partita (scoreline quasi sempre 0-0).
        ft_row = next(row for row in result.minute_window_metrics if row["window"] == "FT")
        early_row = next(row for row in result.minute_window_metrics if row["window"] == "00_15")
        self.assertLess(ft_row["log_loss"], early_row["log_loss"])

        for model_payload in result.details["models"].values():
            self.assertIn("log_loss", model_payload["probability_metrics"])
            self.assertIn("brier", model_payload["probability_metrics"])

    def test_no_fixture_appears_in_both_train_and_valid_within_a_fold(self):
        df = _synthetic_live_frame(n_matches=40)
        splits = match_level_temporal_splits(df, n_splits=3, min_train_matches=15, min_valid_matches=5)
        self.assertGreater(len(splits), 0)
        for train_idx, valid_idx in splits:
            self.assertEqual(set(df.iloc[train_idx]["fixture_id"]) & set(df.iloc[valid_idx]["fixture_id"]), set())

    def test_empty_frame_is_skipped_without_raising(self):
        result = train_live_match_outcome_model(frame=pd.DataFrame(), save_model=False)
        self.assertEqual(result.status, "skipped_no_data")

    def test_unexpected_labels_raise_value_error(self):
        df = _synthetic_live_frame(n_matches=20)
        df.loc[0, "y"] = "SOMETHING_ELSE"
        with self.assertRaises(ValueError):
            train_live_match_outcome_model(frame=df, save_model=False)

    def test_insufficient_matches_is_skipped_without_raising(self):
        df = _synthetic_live_frame(n_matches=5)
        result = train_live_match_outcome_model(
            frame=df, save_model=False, n_splits=3, min_train_matches=15, min_valid_matches=5
        )
        self.assertEqual(result.status, "skipped_insufficient_matches_for_temporal_cv")


class TestLiveMatchOutcomeExpert(unittest.TestCase):
    def _fitted_multiclass_estimator(self):
        rng = np.random.RandomState(3)
        n = 150
        X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
        y = pd.Series(np.where(X["f1"] > 0.3, "HOME", np.where(X["f1"] < -0.3, "AWAY", "DRAW")))
        model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
        return model, X

    def test_predict_proba_sums_to_one(self):
        model, X = self._fitted_multiclass_estimator()
        expert = LiveMatchOutcomeExpert.from_estimator(estimator=model, feature_names=["f1", "f2"])

        proba = expert.predict_proba(X)
        self.assertEqual(proba.shape, (len(X), 3))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_dict_exposes_three_named_probabilities(self):
        model, X = self._fitted_multiclass_estimator()
        expert = LiveMatchOutcomeExpert.from_estimator(estimator=model, feature_names=["f1", "f2"])

        rows = expert.predict_proba_dict(X.iloc[:3])
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(set(row.keys()), {"HOME", "DRAW", "AWAY"})
            self.assertAlmostEqual(sum(row.values()), 1.0, places=6)

    def test_estimator_without_predict_proba_raises(self):
        class Dummy:
            pass

        with self.assertRaises(TypeError):
            LiveMatchOutcomeExpert.from_estimator(estimator=Dummy())

    def test_load_production_raises_lookup_error_when_missing(self):
        class _FakeRegistry:
            def get_production(self, market):
                return None

        with self.assertRaises(LookupError):
            LiveMatchOutcomeExpert.load_production(registry=_FakeRegistry())

    def test_load_latest_raises_lookup_error_when_missing(self):
        class _FakeRegistry:
            def get_latest(self, market):
                return None

        with self.assertRaises(LookupError):
            LiveMatchOutcomeExpert.load_latest(registry=_FakeRegistry())


if __name__ == "__main__":
    unittest.main()

