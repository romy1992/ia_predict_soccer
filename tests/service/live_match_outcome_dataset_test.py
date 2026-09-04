"""Test per LIVE-03 (dataset per l'addestramento di modelli LIVE separati).

Vero SQLite in-memory (stesso pattern di `live_feature_store_test.py`,
LIVE-02): esercita `build_match_outcome_dataset` end-to-end, incluso il
filtro sulle sole fixture GIA' CONCLUSE (stato finale) e la coerenza del
target 1X2 (costante per tutte le righe della stessa fixture, derivato dal
risultato REALE finale).
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.data.live.live_models as live_models  # noqa: F401  (registra le tabelle live su Base.metadata)
from src.data.live.live_models import Base, LiveFixtureSnapshot
from src.ml.live.live_feature_store import LiveFeatureStore
from src.ml.live.live_match_outcome_dataset import build_match_outcome_dataset, completed_fixture_snapshots
from src.ml.markets.market_1x2 import label_1x2
from src.repository.live_data_repository import LiveDataRepository
from src.repository.prediction_ledger_repository import PredictionLedgerRepository

T0 = datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc)


def _snapshot(fixture_id, captured_at, minute, status, home_goals, away_goals, home_team_id=33, away_team_id=34):
    return LiveFixtureSnapshot(
        fixture_id=fixture_id,
        captured_at=captured_at,
        status=status,
        elapsed_minute=minute,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_goals=home_goals,
        away_goals=away_goals,
    )


def _make_in_memory_engine():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return engine


class TestBuildMatchOutcomeDataset(unittest.TestCase):
    def setUp(self):
        self.engine = _make_in_memory_engine()
        self.session_factory = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        self._live_patch = mock.patch("src.repository.live_data_repository.SessionLocal", new=self.session_factory)
        self._ledger_patch = mock.patch(
            "src.repository.prediction_ledger_repository.SessionLocal", new=self.session_factory
        )
        self._live_patch.start()
        self._ledger_patch.start()
        self.addCleanup(self._live_patch.stop)
        self.addCleanup(self._ledger_patch.stop)

        self.live_repo = LiveDataRepository()
        self.store = LiveFeatureStore(
            live_repository=self.live_repo, prediction_repository=PredictionLedgerRepository()
        )

    def test_completed_fixture_snapshots_excludes_not_yet_finished(self):
        self.live_repo.save_fixture_snapshots(
            [
                _snapshot(100, T0, 90, "FT", 2, 1),
                _snapshot(200, T0, 60, "2H", 0, 0),  # non ancora conclusa -> esclusa
            ]
        )
        completed = completed_fixture_snapshots(self.live_repo)
        self.assertEqual(set(completed.keys()), {100})

    def test_dataset_has_one_row_per_snapshot_with_constant_target_per_fixture(self):
        self.live_repo.save_fixture_snapshots(
            [
                _snapshot(100, T0, 10, "1H", 0, 0),
                _snapshot(100, T0 + timedelta(minutes=40), 50, "2H", 1, 0),
                _snapshot(100, T0 + timedelta(minutes=80), 90, "FT", 2, 0),
            ]
        )
        df = build_match_outcome_dataset(feature_store=self.store, live_repository=self.live_repo)

        self.assertEqual(len(df), 3)
        self.assertEqual(set(df["fixture_id"]), {100})
        self.assertEqual(df["y"].nunique(), 1)
        self.assertEqual(df["y"].iloc[0], label_1x2(2, 0))
        self.assertTrue((df["match_time"].diff().dropna() >= timedelta(0)).all())

    def test_unfinished_fixtures_are_excluded_from_dataset(self):
        self.live_repo.save_fixture_snapshots([_snapshot(200, T0, 60, "2H", 0, 0)])
        df = build_match_outcome_dataset(feature_store=self.store, live_repository=self.live_repo)
        self.assertTrue(df.empty)

    def test_multiple_completed_fixtures_each_keep_their_own_target(self):
        self.live_repo.save_fixture_snapshots(
            [
                _snapshot(100, T0, 90, "FT", 2, 0),
                _snapshot(101, T0 + timedelta(minutes=5), 90, "FT", 0, 0, home_team_id=35, away_team_id=36),
            ]
        )
        df = build_match_outcome_dataset(feature_store=self.store, live_repository=self.live_repo)
        by_fixture = df.set_index("fixture_id")["y"].to_dict()
        self.assertEqual(by_fixture[100], "HOME")
        self.assertEqual(by_fixture[101], "DRAW")


if __name__ == "__main__":
    unittest.main()

