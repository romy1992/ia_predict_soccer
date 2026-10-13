"""Test del dataset marcatori: regole del bersaglio, minuti, assenza di leakage, divisioni per tempo.
Dati inventati e minuscoli: verificano il CODICE, non la qualita' di un modello."""

from __future__ import annotations

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.data.player.scoring_dataset import (
    FEATURE_COLUMNS, TARGET, _appearances, build_dataset, temporal_cv_splits,
)

DATES = {1: "2025-08-10T18:00:00+00:00", 2: "2025-08-17T18:00:00+00:00", 3: "2025-08-24T18:00:00+00:00",
         4: "2025-08-31T18:00:00+00:00", 5: "2025-09-07T18:00:00+00:00"}


def _matches():
    rows = [(f, d, 135, 2025, 1 if f % 2 else 2, 2 if f % 2 else 1) for f, d in DATES.items()]
    return pd.DataFrame(rows, columns=["fixture_id", "date_match", "league", "season", "home_id", "away_id"])


def _lineup(fixture_id, drop_starter=None):
    """Squadra 1: 101 G, 102-105 D, 106-109 M, 110-111 F, panchina 112. Squadra 2: 201..211, panchina 212."""
    rows = []
    for base, team in ((100, 1), (200, 2)):
        pos = ["G"] + ["D"] * 4 + ["M"] * 4 + ["F"] * 2
        for i, p in enumerate(pos, start=1):
            rows.append((fixture_id, base + i, team, p, True))
        rows.append((fixture_id, base + 12, team, "F", False))
    df = pd.DataFrame(rows, columns=["fixture_id", "player_id", "team_id", "position", "is_starter"])
    return df[df["player_id"] != drop_starter] if drop_starter else df


def _ev(fixture, typ, detail, minute, team, player, assist=None, comments=None):
    return (fixture, typ, detail, comments, minute, team, player, assist)


COLS = ["fixture_id", "event_type", "event_detail", "comments", "elapsed", "team_id", "player_id", "assist_id"]


def _events(extra_f4=()):
    rows = [
        # F1
        _ev(1, "Goal", "Normal Goal", 20, 1, 110),
        _ev(1, "Goal", "Own Goal", 30, 1, 102),                        # autogol: non e' un marcatore
        _ev(1, "Goal", "Normal Goal", 90, 1, 109),                     # 90' (recupero): conta
        _ev(1, "Goal", "Normal Goal", 105, 1, 108),                    # supplementare: non conta
        _ev(1, "Goal", "Penalty", 120, 1, 106, comments="Penalty Shootout"),   # lotteria: non conta
        _ev(1, "subst", "Substitution 1", 60, 1, 111, assist=112),     # esce 111, entra 112
        _ev(1, "Card", "Red Card", 70, 2, 205),
        # F2: panchinaro 112 non entra
        _ev(2, "Goal", "Normal Goal", 10, 1, 110),
        _ev(2, "Goal", "Penalty", 55, 1, 110),
        # F3, F5: nessun gol ma eventi presenti
        _ev(3, "Card", "Yellow Card", 40, 2, 203),
        _ev(5, "Card", "Yellow Card", 40, 2, 203),
        # F4
        _ev(4, "Card", "Yellow Card", 40, 2, 204),
    ]
    rows.extend(extra_f4)
    return pd.DataFrame(rows, columns=COLS)


def _lineups():
    parts = [_lineup(f) for f in (1, 2, 3, 4)] + [_lineup(5, drop_starter=211)]  # F5: 21 titolari
    return pd.concat(parts, ignore_index=True)


class ScoringDatasetTest(unittest.TestCase):
    def setUp(self):
        self.matches, self.lineups, self.events = _matches(), _lineups(), _events()
        self.ds = build_dataset(self.matches, self.lineups, self.events)

    def row(self, fixture, player):
        r = self.ds[(self.ds.fixture_id == fixture) & (self.ds.player_id == player)]
        return r.iloc[0] if len(r) else None

    def test_target_rules(self):
        self.assertEqual(self.row(1, 110)[TARGET], 1)      # gol normale
        self.assertEqual(self.row(1, 109)[TARGET], 1)      # gol al 90'
        self.assertEqual(self.row(1, 102)[TARGET], 0)      # autogol
        self.assertEqual(self.row(1, 108)[TARGET], 0)      # supplementare
        self.assertEqual(self.row(1, 106)[TARGET], 0)      # lotteria finale
        self.assertEqual(self.row(2, 110)[TARGET], 1)      # gol + rigore: sempre 1 riga, bersaglio 1

    def test_rows_excluded(self):
        self.assertIsNone(self.row(1, 101))                # portiere
        self.assertIsNone(self.row(2, 112))                # panchinaro mai entrato
        self.assertEqual(len(self.ds[self.ds.fixture_id == 5]), 0)   # 21 titolari
        self.assertIsNotNone(self.row(1, 112))             # panchinaro entrato al 60'

    def test_minutes_from_substitutions_and_red_card(self):
        a = _appearances(self.matches, self.lineups, self.events)
        m = a[a.fixture_id == 1].set_index("player_id")["minutes"]
        self.assertEqual(m[111], 60)    # uscito al 60'
        self.assertEqual(m[112], 30)    # entrato al 60' fino al 90'
        self.assertEqual(m[205], 70)    # rosso al 70'
        self.assertEqual(m[110], 90)

    def test_features_use_only_the_past(self):
        first, second = self.row(1, 110), self.row(2, 110)
        self.assertEqual(first["n_prior_apps"], 0)
        self.assertTrue(pd.isna(first["prior_goals_last5"]))
        self.assertEqual(second["n_prior_apps"], 1)
        self.assertEqual(second["prior_goals_last5"], 1)       # il gol di F1, non quelli di F2
        third = self.row(3, 110)
        self.assertEqual(third["prior_goals_last5"], 3)        # F1: 1 gol, F2: gol + rigore = 2
        self.assertEqual(third["pens_last20"], 1)

    def test_changing_the_future_does_not_change_the_past(self):
        future = _events(extra_f4=[_ev(4, "Goal", "Normal Goal", 5, 1, 110), _ev(4, "Goal", "Normal Goal", 50, 2, 210)])
        changed = build_dataset(self.matches, self.lineups, future)
        before = self.ds[self.ds.fixture_id <= 3].reset_index(drop=True)
        after = changed[changed.fixture_id <= 3].reset_index(drop=True)
        assert_frame_equal(before[FEATURE_COLUMNS], after[FEATURE_COLUMNS])
        assert_frame_equal(before[[TARGET]], after[[TARGET]])

    def test_team_form_is_a_mean_of_previous_matches(self):
        # squadra 1: gol non autogol F1=2, F2=2, F3=0 -> media 4/3 alla partita F4
        r = self.row(4, 110)
        self.assertAlmostEqual(r["team_gf10"], 4 / 3)
        self.assertTrue(pd.isna(self.row(2, 110)["team_gf10"]))   # meno di 3 partite alle spalle

    def test_columns_and_probability_inputs_are_numeric(self):
        for col in FEATURE_COLUMNS:
            self.assertIn(col, self.ds.columns)
        self.assertTrue(set(self.ds[TARGET].unique()) <= {0, 1})
        self.assertTrue(((self.ds["rate_last10_shrunk"] > 0)).all())

    def test_temporal_splits_never_mix_a_match_and_respect_time(self):
        splits = temporal_cv_splits(self.ds, n_folds=2)
        self.assertGreaterEqual(len(splits), 1)
        for train, test in splits:
            self.assertTrue(set(train).isdisjoint(test))
            self.assertLess(self.ds.loc[train, "match_date"].max(), self.ds.loc[test, "match_date"].min())
            self.assertTrue(set(self.ds.loc[train, "fixture_id"]).isdisjoint(set(self.ds.loc[test, "fixture_id"])))


if __name__ == "__main__":
    unittest.main()


class ScoringDatasetFromDbTest(unittest.TestCase):
    """Stessa logica ma letta da un DB SQLite in-memory: verifica che le SELECT funzionino."""

    def test_build_from_session(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        from src.data.player.player_lineup_models import Base, PlayerLineup
        from src.data.player.player_models import PlayerMatchEvent
        from src.data.player.scoring_dataset import build_from_session
        from src.service_ia.model.match import Match

        engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine, tables=[Match.__table__, PlayerLineup.__table__, PlayerMatchEvent.__table__])
        s = sessionmaker(bind=engine)()
        for r in _matches().itertuples():
            s.add(Match(id_fixture=r.fixture_id, date_match=r.date_match, league_match=r.league, season=r.season,
                        id_team_home=r.home_id, id_team_away=r.away_id, status="FT"))
        for r in _lineups().itertuples():
            s.add(PlayerLineup(fixture_id=r.fixture_id, player_id=r.player_id, team_id=r.team_id,
                               position=r.position, is_starter=r.is_starter))
        for i, r in enumerate(_events().itertuples()):
            s.add(PlayerMatchEvent(id_event=f"e{i}", fixture_id=r.fixture_id, event_type=r.event_type,
                                   event_detail=r.event_detail, comments=r.comments, elapsed_minute=r.elapsed,
                                   team_id=r.team_id, player_id=r.player_id, assist_id=r.assist_id))
        s.commit()
        ds = build_from_session(s, [135], [2025], ["FT"])
        expected = build_dataset(_matches(), _lineups(), _events())
        self.assertEqual(len(ds), len(expected))
        self.assertEqual(int(ds[TARGET].sum()), int(expected[TARGET].sum()))
