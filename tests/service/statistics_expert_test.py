import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from src.ml.experts.statistics.statistics_expert import StatisticsExpert


def _match(fixture_id, date_match, home_shots, away_shots, home_ft, away_ft, season=2026, status="FT"):
    return {
        "id_fixture": fixture_id,
        "season": season,
        "status": status,
        "date_match": date_match,
        "id_team_home": 1,
        "id_team_away": 2,
        "mean_statistics": [
            {"id_team": 1, "mean_Shots on Goal": home_shots},
            {"id_team": 2, "mean_Shots on Goal": away_shots},
        ],
        "statistics": [
            {"statistics_team_id": 1, "score_ft": home_ft},
            {"statistics_team_id": 2, "score_ft": away_ft},
        ],
    }


class TestStatisticsExpert(unittest.TestCase):
    def test_build_dataset_has_no_odds_columns_and_correct_labels(self):
        matches = [
            _match(1, "2026-01-01T18:00:00+00:00", home_shots=6, away_shots=3, home_ft=2, away_ft=1),
            _match(2, "2026-01-08T18:00:00+00:00", home_shots=2, away_shots=5, home_ft=0, away_ft=2),
            _match(3, "2026-01-15T18:00:00+00:00", home_shots=4, away_shots=4, home_ft=1, away_ft=1),
        ]
        expert = StatisticsExpert()
        frame = expert.build_dataset_from_records(matches, outcome="home_win")

        self.assertEqual(len(frame), 3)
        self.assertFalse(any("odd" in col.lower() for col in frame.columns))
        self.assertTrue(any(col.startswith("stat_") for col in frame.columns))
        self.assertListEqual(frame["y"].tolist(), [1, 0, 0])  # solo la 1a partita e' vittoria home

    def test_build_dataset_rejects_unsupported_outcome(self):
        expert = StatisticsExpert()
        with self.assertRaises(ValueError):
            expert.build_dataset_from_records([_match(1, "2026-01-01T18:00:00+00:00", 5, 5, 1, 0)], outcome="not_a_real_outcome")

    def test_build_dataset_skips_matches_without_mean_statistics(self):
        broken_match = _match(1, "2026-01-01T18:00:00+00:00", 5, 5, 1, 0)
        broken_match["mean_statistics"] = None
        expert = StatisticsExpert()
        frame = expert.build_dataset_from_records([broken_match])
        self.assertTrue(frame.empty)

    def _synthetic_matches(self, n=150, seed=0):
        rng = np.random.RandomState(seed)
        base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        matches = []
        for i in range(n):
            diff = rng.normal(0, 1.2)
            home_shots = 5.0 + diff
            away_shots = 5.0 - diff
            home_win_prob = 1.0 / (1.0 + np.exp(-diff))
            home_wins = rng.random_sample() < home_win_prob
            home_ft, away_ft = (2, 1) if home_wins else (1, 2)
            date_match = (base_date + timedelta(days=i)).isoformat()
            matches.append(
                _match(i + 1, date_match, home_shots=home_shots, away_shots=away_shots, home_ft=home_ft, away_ft=away_ft)
            )
        return matches

    def test_fit_and_evaluate_returns_temporal_probability_metrics(self):
        expert = StatisticsExpert()
        frame = expert.build_dataset_from_records(self._synthetic_matches(), outcome="home_win")

        report = expert.fit_and_evaluate(frame, n_splits=3, min_train_size=80, min_valid_size=20)

        self.assertEqual(report["cv_strategy"], "expanding_window")
        self.assertGreater(report["folds"], 0)
        self.assertIn("log_loss", report["probability_metrics"])
        self.assertIn("brier", report["probability_metrics"])
        self.assertIn("ece", report["probability_metrics"])

    def test_embedding_features_after_fit_returns_probabilities(self):
        expert = StatisticsExpert()
        frame = expert.build_dataset_from_records(self._synthetic_matches(), outcome="home_win")
        expert.fit_and_evaluate(frame, n_splits=3, min_train_size=80, min_valid_size=20)

        embedding = expert.embedding_features(frame.iloc[:10])
        self.assertEqual(len(embedding), 10)
        self.assertTrue((embedding >= 0.0).all() and (embedding <= 1.0).all())

    def test_embedding_features_before_fit_raises(self):
        expert = StatisticsExpert()
        frame = expert.build_dataset_from_records(self._synthetic_matches(n=5), outcome="home_win")
        with self.assertRaises(RuntimeError):
            expert.embedding_features(frame)


if __name__ == "__main__":
    unittest.main()
