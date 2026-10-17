"""Test del dataset gialli: regole del bersaglio, arbitro, assenza di leakage. Dati inventati e minuscoli."""

from __future__ import annotations

import unittest

import pandas as pd

from src.data.player.cards_dataset import FEATURE_COLUMNS, FEATURE_COLUMNS_V2, TARGET, build_dataset
from tests.service.player_scoring_dataset_test import COLS, _ev, _lineups, _matches


def _events():
    rows = [
        # F1: gialli a 204 (30'), secondo giallo a 205 (80'), rosso diretto a 206, giallo ai supplementari a 207
        _ev(1, "Card", "Yellow Card", 30, 2, 204),
        _ev(1, "Card", "Second Yellow card", 80, 2, 205),
        _ev(1, "Card", "Red Card", 60, 2, 206),
        _ev(1, "Card", "Yellow Card", 105, 2, 207),
        _ev(1, "Card", "Yellow Card", 50, 1, 103),
        # F2: 204 ancora ammonito
        _ev(2, "Card", "Yellow Card", 40, 2, 204),
        # F3, F4: eventi presenti ma nessun giallo a 204
        _ev(3, "Goal", "Normal Goal", 10, 1, 110),
        _ev(4, "Goal", "Normal Goal", 10, 1, 110),
        _ev(5, "Goal", "Normal Goal", 10, 1, 110),
    ]
    return pd.DataFrame(rows, columns=COLS)


REFS = pd.DataFrame({"fixture_id": [1, 2, 3, 4, 5], "referee": ["A", "A", "B", "A", None]})


class CardsDatasetTest(unittest.TestCase):
    def setUp(self):
        self.ds = build_dataset(_matches(), _lineups(), _events(), REFS)

    def row(self, fixture, player):
        r = self.ds[(self.ds.fixture_id == fixture) & (self.ds.player_id == player)]
        return r.iloc[0] if len(r) else None

    def test_target_rules(self):
        self.assertEqual(self.row(1, 204).yellow, 1)
        self.assertEqual(self.row(1, 205).yellow, 1)   # il secondo giallo e' un'ammonizione
        self.assertEqual(self.row(1, 206).yellow, 0)   # rosso diretto: non e' un giallo
        self.assertEqual(self.row(1, 207).yellow, 0)   # supplementari: fuori dai tempi regolamentari
        self.assertEqual(self.row(1, 103).yellow, 1)

    def test_goalkeepers_and_incomplete_lineups_excluded(self):
        self.assertTrue(self.ds[self.ds.player_id.isin([101, 201])].empty)
        self.assertNotIn(5, set(self.ds.fixture_id))  # F5 ha 21 titolari

    def test_no_leakage_into_current_match(self):
        # Alla prima partita nessuno ha storia: nessun conteggio passato.
        first = self.ds[self.ds.fixture_id == 1]
        self.assertTrue(first["prior_yellows_last5"].isna().all())
        # Il giallo di 204 in F1 compare solo dalla partita successiva.
        self.assertEqual(self.row(2, 204).prior_yellows_last5, 1)
        self.assertEqual(self.row(3, 204).prior_yellows_last5, 2)

    def test_referee_feature_uses_only_previous_matches(self):
        # Arbitro A: F1 ha 4 gialli validi (204, 205, 103 + il giallo a 105' non conta -> 3).
        # Alla F1 l'arbitro non e' noto, a F2 lo e' (1 partita arbitrata).
        self.assertEqual(self.row(1, 204).ref_known, 0)
        self.assertEqual(self.row(2, 204).ref_known, 1)
        # F2: (3 gialli + K*media_globale) / (1 + K), con media globale = 3 gialli dalla sola F1.
        k = 5.0
        self.assertAlmostEqual(self.row(2, 204).ref_yc_per_match, (3 + k * 3) / (1 + k))
        # Arbitro mai visto (B alla F3): nessuna partita precedente
        self.assertEqual(self.row(3, 204).ref_known, 0)

    def test_team_yellow_form_is_prior_only(self):
        self.assertTrue(self.ds[self.ds.fixture_id == 1]["team_yc10"].isna().all())

    def test_no_infinite_values_and_all_features_present(self):
        self.assertEqual(list(self.ds.columns[-len(FEATURE_COLUMNS_V2) - 1:]), FEATURE_COLUMNS_V2 + [TARGET])
        self.assertFalse(self.ds[FEATURE_COLUMNS_V2].isin([float("inf"), float("-inf")]).any().any())

    def test_fouls_features_are_prior_only(self):
        # 204: falli 3 in F1 (90'), 5 in F2: alla F1 non c'e' passato, alla F2 conta solo F1.
        stats = pd.DataFrame({
            "fixture_id": [1, 2], "player_id": [204, 204], "shots_total": [0, 0], "shots_on": [0, 0],
            "key_passes": [0, 0], "pen_won": [0, 0], "rating": [6.5, 6.5],
            "fouls_committed": [3, 5], "fouls_drawn": [1, 2],
        })
        ds = build_dataset(_matches(), _lineups(), _events(), REFS, stats=stats)
        f1 = ds[(ds.fixture_id == 1) & (ds.player_id == 204)].iloc[0]
        f2 = ds[(ds.fixture_id == 2) & (ds.player_id == 204)].iloc[0]
        self.assertTrue(pd.isna(f1.fouls_c_p90_l10))
        self.assertEqual(f1.stats_games_l10, 0)
        self.assertAlmostEqual(f2.fouls_c_p90_l10, 3.0)   # 3 falli in 90'
        self.assertEqual(f2.stats_games_l10, 1)

    def test_without_stats_the_new_columns_are_empty(self):
        self.assertTrue(self.ds["fouls_c_p90_l10"].isna().all())
        self.assertTrue((self.ds["stats_games_l10"] == 0).all())

    def test_empty_inputs(self):
        empty = build_dataset(_matches().iloc[0:0], _lineups().iloc[0:0], _events().iloc[0:0])
        self.assertTrue(empty.empty)
        self.assertIn(TARGET, empty.columns)

    def test_upcoming_fixture_gets_starters_only(self):
        matches = pd.concat([_matches(), pd.DataFrame([(6, "2025-09-14T18:00:00+00:00", 135, 2025, 1, 2)],
                                                      columns=_matches().columns)], ignore_index=True)
        lineups = pd.concat([_lineups(), _lineups()[_lineups().fixture_id == 4].assign(fixture_id=6)], ignore_index=True)
        ds = build_dataset(matches, lineups, _events(), REFS, upcoming_fixture_ids=[6])
        rows = ds[ds.fixture_id == 6]
        self.assertEqual(len(rows), 20)  # 22 titolari meno i 2 portieri; la panchina (112, 212) non c'e'
        self.assertTrue(rows[rows.player_id.isin([112, 212])].empty)
