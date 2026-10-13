"""Test statistiche individuali per partita: mapping puro + repository su
SQLite in-memory (stesso stile di `player_lineup_service_test.py`)."""

from __future__ import annotations

import unittest
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.data.player import player_stats_models  # noqa: F401  (registra la tabella su Base.metadata)
from src.data.player.player_stats_models import Base, PlayerMatchStats
from src.data.player.player_stats_service import extract_player_stats
from src.repository.player_data_repository import PlayerDataRepository


def _raw_block(team_id=505):
    return {
        "team": {"id": team_id, "name": "Inter"},
        "players": [
            {
                "player": {"id": 10, "name": "Striker"},
                "statistics": [{
                    "games": {"minutes": 90, "rating": "7.6", "captain": True, "substitute": False, "position": "F"},
                    "offsides": 2,
                    "shots": {"total": 5, "on": 3},
                    "goals": {"total": 1, "assists": 1, "saves": None},
                    "passes": {"total": 20, "key": 2},
                    "dribbles": {"attempts": 4, "success": 2},
                    "fouls": {"drawn": 3, "committed": 1},
                    "cards": {"yellow": 1, "red": 0},
                    "penalty": {"won": 1, "commited": 0, "scored": 1, "missed": 0, "saved": None},
                }],
            },
            {"player": {"id": 11, "name": "Bench"}, "statistics": [{"games": {"minutes": None, "rating": None}, "shots": {}}]},
            {"player": {"id": None, "name": "Ghost"}, "statistics": [{"games": {"minutes": 5}}]},
            {"player": {"id": 12, "name": "NoStats"}, "statistics": []},
        ],
    }


class TestExtractPlayerStats(unittest.TestCase):
    def test_maps_full_row(self):
        rows = extract_player_stats(100, _raw_block())
        row = next(r for r in rows if r.player_id == 10)
        self.assertEqual((row.fixture_id, row.team_id, row.minutes, row.rating), (100, 505, 90, 7.6))
        self.assertEqual((row.shots_total, row.shots_on, row.goals, row.assists), (5, 3, 1, 1))
        self.assertEqual((row.key_passes, row.dribbles_attempts, row.fouls_drawn), (2, 4, 3))
        self.assertEqual((row.pen_won, row.pen_committed, row.pen_scored, row.pen_missed), (1, 0, 1, 0))
        self.assertEqual((row.offsides, row.yellow, row.red), (2, 1, 0))
        self.assertTrue(row.captain)
        self.assertFalse(row.substitute)
        self.assertEqual((row.position, row.source), ("F", "api_sports"))

    def test_missing_values_become_none_and_invalid_skipped(self):
        rows = extract_player_stats(100, _raw_block())
        self.assertEqual({r.player_id for r in rows}, {10, 11})
        bench = next(r for r in rows if r.player_id == 11)
        self.assertIsNone(bench.minutes)
        self.assertIsNone(bench.rating)
        self.assertIsNone(bench.shots_total)
        self.assertIsNone(bench.captain)

    def test_bad_values_and_empty_block(self):
        raw = {"team": {"id": 1}, "players": [
            {"player": {"id": 1}, "statistics": [{"games": {"minutes": "x", "rating": "n/a"}}]}]}
        row = extract_player_stats(1, raw)[0]
        self.assertIsNone(row.minutes)
        self.assertIsNone(row.rating)
        self.assertEqual(extract_player_stats(1, {}), [])


class TestPlayerStatsRepositoryWithDb(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        patch = mock.patch("src.repository.player_data_repository.SessionLocal",
                           new=sessionmaker(bind=engine, autocommit=False, autoflush=False))
        patch.start()
        self.addCleanup(patch.stop)
        self.repo = PlayerDataRepository()

    def test_save_is_idempotent_and_updates(self):
        self.assertFalse(self.repo.has_stats_for_fixture(100))
        self.repo.save_player_stats(extract_player_stats(100, _raw_block()))
        self.repo.save_player_stats(extract_player_stats(100, _raw_block()))
        saved = self.repo.list_stats_for_fixture(100)
        self.assertEqual(len(saved), 2)
        self.assertTrue(self.repo.has_stats_for_fixture(100))
        self.assertIsNotNone(saved[0].created_at)

        changed = _raw_block()
        changed["players"][0]["statistics"][0]["shots"]["total"] = 9
        self.repo.save_player_stats(extract_player_stats(100, changed))
        self.assertEqual(next(r for r in self.repo.list_stats_for_fixture(100) if r.player_id == 10).shots_total, 9)

    def test_dedupes_same_key_in_batch_and_captured_ids(self):
        rows = extract_player_stats(100, _raw_block())
        rows.append(PlayerMatchStats(fixture_id=100, player_id=10, shots_total=1))
        self.repo.save_player_stats(rows)  # niente IntegrityError
        self.assertEqual(self.repo.captured_stats_fixture_ids([100, 200]), {100})
        self.assertEqual(self.repo.captured_stats_fixture_ids([]), set())


if __name__ == "__main__":
    unittest.main()
