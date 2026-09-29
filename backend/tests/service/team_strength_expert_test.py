import unittest

from src.ml.experts.team_strength.team_strength_expert import (
    PRIOR_GOALS,
    TeamStrengthExpert,
)


def _match(fixture_id, date_match, home_id, away_id, home_ft, away_ft, season=2026, status="FT"):
    return {
        "id_fixture": fixture_id,
        "season": season,
        "status": status,
        "date_match": date_match,
        "id_team_home": home_id,
        "id_team_away": away_id,
        "statistics": [
            {"statistics_team_id": home_id, "score_ft": home_ft},
            {"statistics_team_id": away_id, "score_ft": away_ft},
        ],
    }


class TestTeamStrengthExpert(unittest.TestCase):
    def setUp(self):
        # A=1, B=2, C=3
        self.matches = [
            _match(1, "2026-01-01T18:00:00+00:00", home_id=1, away_id=2, home_ft=2, away_ft=1),
            _match(2, "2026-01-08T18:00:00+00:00", home_id=2, away_id=1, home_ft=0, away_ft=0),
            _match(3, "2026-01-15T18:00:00+00:00", home_id=1, away_id=3, home_ft=1, away_ft=1),
        ]

    def test_cold_start_uses_league_prior_and_no_leakage(self):
        expert = TeamStrengthExpert()
        frame = expert.build_ratings_dataset(self.matches)

        self.assertEqual(len(frame), 3)

        first_row = frame.iloc[0]
        # Prima partita in assoluto per A e B: nessuno storico -> prior puro.
        self.assertAlmostEqual(first_row["home_team_attack_rating"], PRIOR_GOALS, places=6)
        self.assertAlmostEqual(first_row["away_team_attack_rating"], PRIOR_GOALS, places=6)
        self.assertEqual(first_row["home_team_matches_played"], 0.0)
        self.assertEqual(first_row["away_team_matches_played"], 0.0)
        self.assertAlmostEqual(first_row["home_team_rolling_form"], 0.5, places=6)

    def test_ratings_reflect_only_past_matches(self):
        expert = TeamStrengthExpert()
        frame = expert.build_ratings_dataset(self.matches)

        second_row = frame.iloc[1]  # match 2: B(home) vs A(away)
        # B ha giocato 1 sola partita (persa 1-2 in trasferta) prima di questa.
        # Nota: lo snapshot arrotonda a 4 decimali per leggibilita' (vedi _TeamState.snapshot).
        expected_b_attack = round((1 + 5 * PRIOR_GOALS) / (1 + 5), 4)
        expected_a_attack = round((2 + 5 * PRIOR_GOALS) / (1 + 5), 4)
        self.assertAlmostEqual(second_row["home_team_attack_rating"], expected_b_attack, places=4)
        self.assertAlmostEqual(second_row["away_team_attack_rating"], expected_a_attack, places=4)
        self.assertEqual(second_row["home_team_matches_played"], 1.0)
        self.assertEqual(second_row["away_team_matches_played"], 1.0)

        third_row = frame.iloc[2]  # match 3: A(home) vs C(away)
        # C non ha mai giocato prima -> ancora prior puro anche se e' la 3a riga del frame.
        self.assertAlmostEqual(third_row["away_team_attack_rating"], PRIOR_GOALS, places=4)
        self.assertEqual(third_row["away_team_matches_played"], 0.0)
        # A ha giocato 2 partite (2-1 casa, 0-0 fuori) prima di questa.
        expected_a_attack_3 = round((2 + 0 + 5 * PRIOR_GOALS) / (2 + 5), 4)
        self.assertAlmostEqual(third_row["home_team_attack_rating"], expected_a_attack_3, places=4)

    def test_current_ratings_reflects_full_history(self):
        expert = TeamStrengthExpert()
        ratings = expert.current_ratings(self.matches)

        self.assertEqual(ratings[1]["team_matches_played"], 3.0)  # A ha giocato 3 partite
        self.assertEqual(ratings[2]["team_matches_played"], 2.0)  # B ha giocato 2 partite
        self.assertEqual(ratings[3]["team_matches_played"], 1.0)  # C ha giocato 1 partita
        self.assertIn("rating_version", ratings[1])

    def test_version_is_deterministic(self):
        expert_a = TeamStrengthExpert()
        expert_b = TeamStrengthExpert()

        self.assertEqual(expert_a.VERSION, expert_b.VERSION)
        self.assertEqual(TeamStrengthExpert.VERSION, expert_a.VERSION)

        frame_a = expert_a.build_ratings_dataset(self.matches)
        frame_b = expert_b.build_ratings_dataset(self.matches)
        self.assertTrue((frame_a["rating_version"] == frame_b["rating_version"]).all())

    def test_backtest_home_signal_returns_metrics(self):
        expert = TeamStrengthExpert()
        frame = expert.build_ratings_dataset(self.matches)

        report = expert.backtest_home_signal(frame, min_matches_played=0)

        self.assertEqual(report["n"], 3)
        self.assertIsNotNone(report["accuracy"])
        self.assertIsNotNone(report["baseline_home_win_rate"])
        self.assertGreaterEqual(report["accuracy"], 0.0)
        self.assertLessEqual(report["accuracy"], 1.0)

    def test_backtest_home_signal_handles_empty_frame(self):
        expert = TeamStrengthExpert()
        empty_report = expert.backtest_home_signal(expert.build_ratings_dataset([]))

        self.assertEqual(empty_report["n"], 0)
        self.assertIsNone(empty_report["accuracy"])


if __name__ == "__main__":
    unittest.main()

