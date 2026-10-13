"""Predizioni marcatori pre-partita: formazione ufficiale -> probabilita' -> revisioni append-only.
DB SQLite in-memory e modello minuscolo: verificano il CODICE, non la qualita' del modello."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.data.player.player_lineup_models import Base, PlayerLineup
from src.data.player.player_models import PlayerMatchEvent
from src.data.player.scorer_prediction_models import PlayerScorerPrediction
from src.data.player.scoring_dataset import FEATURE_COLUMNS, TARGET, build_dataset
from src.service_ia.model.match import Match
from src.service_ia.scorer import model_store, prediction_service
from tests.service.player_scoring_dataset_test import _events, _lineup, _lineups, _matches


def _bundle(history: pd.DataFrame) -> dict:
    medians = history[FEATURE_COLUMNS].median()
    model = LogisticRegression(max_iter=500).fit(history[FEATURE_COLUMNS].fillna(medians).to_numpy(), history[TARGET])
    import sklearn
    return {"model": model, "medians": medians, "feature_columns": list(FEATURE_COLUMNS),
            "sklearn_version": sklearn.__version__, "version": "test"}


class ScorerPredictionTest(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine, tables=[Match.__table__, PlayerLineup.__table__,
                                                 PlayerMatchEvent.__table__, PlayerScorerPrediction.__table__])
        self.s = sessionmaker(bind=engine)()
        self.now = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)
        for r in _matches().itertuples():
            self.s.add(Match(id_fixture=r.fixture_id, date_match=r.date_match, league_match=135, season=2025,
                             id_team_home=r.home_id, id_team_away=r.away_id, status="FT"))
        for r in _lineups().itertuples():
            if r.fixture_id != 5:
                self.s.add(self._pl(r))
        for i, r in enumerate(_events().itertuples()):
            self.s.add(PlayerMatchEvent(id_event=f"e{i}", fixture_id=r.fixture_id, event_type=r.event_type,
                                        event_detail=r.event_detail, comments=r.comments, elapsed_minute=r.elapsed,
                                        team_id=r.team_id, player_id=r.player_id, assist_id=r.assist_id))
        # partita futura (fixture 99) tra 2 ore, formazione ufficiale gia' pubblicata
        ko = (self.now + timedelta(hours=2)).isoformat()
        self.s.add(Match(id_fixture=99, date_match=ko, league_match=135, season=2026,
                         id_team_home=1, id_team_away=2, status="NS"))
        for r in _lineup(99).itertuples():
            self.s.add(self._pl(r))
        self.s.commit()
        hist = build_dataset(_matches().iloc[:4], _lineups().query("fixture_id<=4"),
                             _events().query("fixture_id<=4"))
        self.bundle = _bundle(hist)

    @staticmethod
    def _pl(r):
        return PlayerLineup(fixture_id=r.fixture_id, player_id=r.player_id, team_id=r.team_id,
                            position=r.position, is_starter=r.is_starter, player_name=f"P{r.player_id}",
                            team_name=f"T{r.team_id}")

    def test_predicts_only_outfield_starters_and_appends_one_revision(self):
        out = prediction_service.predict_upcoming(self.s, self.bundle, leagues=[135], now=self.now,
                                                  finished_statuses=["FT"])
        self.assertEqual(out["predicted"], 1)
        rows = self.s.execute(select(PlayerScorerPrediction)).scalars().all()
        self.assertEqual(len(rows), 20)  # 22 titolari - 2 portieri
        self.assertTrue(all(r.revision == 1 and r.fixture_id == 99 for r in rows))
        self.assertTrue(all(0.0 < r.probability < 1.0 for r in rows))
        self.assertEqual({r.position for r in rows}, {"F", "M", "D"})

    def test_same_lineup_is_not_rewritten_but_changed_lineup_gets_new_revision(self):
        prediction_service.predict_upcoming(self.s, self.bundle, leagues=[135], now=self.now, finished_statuses=["FT"])
        again = prediction_service.predict_upcoming(self.s, self.bundle, leagues=[135], now=self.now,
                                                    finished_statuses=["FT"])
        self.assertEqual((again["predicted"], again["skipped_same_lineup"]), (0, 1))
        # cambia un titolare: 110 esce, entra la riserva 112
        self.s.query(PlayerLineup).filter_by(fixture_id=99, player_id=110).update({"is_starter": False})
        self.s.query(PlayerLineup).filter_by(fixture_id=99, player_id=112).update({"is_starter": True})
        self.s.commit()
        third = prediction_service.predict_upcoming(self.s, self.bundle, leagues=[135], now=self.now,
                                                    finished_statuses=["FT"])
        self.assertEqual(third["predicted"], 1)
        revs = {r.revision for r in self.s.execute(select(PlayerScorerPrediction)).scalars()}
        self.assertEqual(revs, {1, 2})

    def test_no_lineup_no_predictions(self):
        self.s.query(PlayerLineup).filter_by(fixture_id=99).delete()
        self.s.commit()
        out = prediction_service.predict_upcoming(self.s, self.bundle, leagues=[135], now=self.now,
                                                  finished_statuses=["FT"])
        self.assertEqual(out["predicted"], 0)

    def test_far_matches_are_ignored(self):
        out = prediction_service.predict_upcoming(self.s, self.bundle, leagues=[135], now=self.now - timedelta(days=2),
                                                  finished_statuses=["FT"])
        self.assertEqual(out["fixtures"], 0)

    def test_fetch_official_lineups_only_when_missing_and_complete(self):
        self.s.query(PlayerLineup).filter_by(fixture_id=99).delete()
        self.s.commit()
        calls = []

        class Provider:
            def get_fixture_lineups(self, fid):
                calls.append(fid)
                return []  # non ancora pubblicata

        class Repo:
            def has_lineup_for_fixture(self, fid):
                return False

            def save_lineups(self, rows):
                raise AssertionError("non deve salvare formazioni incomplete")

        out = prediction_service.fetch_official_lineups(self.s, Provider(), Repo(), leagues=[135], now=self.now + timedelta(minutes=70))
        self.assertEqual((out["lineups_fetched"], out["lineups_not_ready"], calls), (0, 1, [99]))

    def test_settlement_and_latest_view(self):
        prediction_service.predict_upcoming(self.s, self.bundle, leagues=[135], now=self.now, finished_statuses=["FT"])
        latest = prediction_service.latest_predictions(self.s, fixture_id=99)
        self.assertEqual(len(latest), 20)
        self.assertEqual(latest, sorted(latest, key=lambda d: -d["probability"]))
        self.assertEqual(prediction_service.settle_predictions(self.s, ["FT"]), {"settled_fixtures": 0})  # NS: non conclusa
        # la partita finisce: 110 segna al 20', 111 solo ai supplementari (non conta)
        self.s.query(Match).filter_by(id_fixture=99).update({"status": "FT"})
        for i, (m, pid) in enumerate(((20, 110), (105, 111))):
            self.s.add(PlayerMatchEvent(id_event=f"x{i}", fixture_id=99, event_type="Goal", event_detail="Normal Goal",
                                        elapsed_minute=m, team_id=1, player_id=pid))
        self.s.commit()
        self.assertEqual(prediction_service.settle_predictions(self.s, ["FT"]), {"settled_fixtures": 1})
        by = {r.player_id: r.scored for r in self.s.execute(select(PlayerScorerPrediction)).scalars()}
        self.assertTrue(by[110])
        self.assertFalse(by[111])
        self.assertFalse(by[102])

    def test_model_store_rejects_tiny_datasets(self):
        with self.assertRaises(ValueError):
            model_store.train_bundle(pd.DataFrame({c: [0.0] for c in FEATURE_COLUMNS + [TARGET, "match_date", "fixture_id"]}))

    def test_probabilities_vary_with_features(self):
        hist = build_dataset(_matches().iloc[:4], _lineups().query("fixture_id<=4"), _events().query("fixture_id<=4"))
        p = model_store.predict_proba(self.bundle, hist)
        self.assertGreater(float(np.ptp(p)), 0.0)


if __name__ == "__main__":
    unittest.main()
