"""Test per le formazioni (cantiere "giocatori che segnano") - stesso
stile di `player_event_service_test.py`: funzioni pure di mapping, poi
repository con vero SQLite in-memory."""

from __future__ import annotations

import unittest
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.data.player.player_lineup_models as player_lineup_models  # noqa: F401  (registra player_lineup su Base.metadata)
from src.data.player.player_lineup_models import Base, PlayerLineup
from src.data.player.player_lineup_service import extract_lineup_players
from src.repository.player_data_repository import PlayerDataRepository


def _raw_team_lineup(team_id=532, team_name="Valencia", formation="4-4-2"):
    return {
        "team": {"id": team_id, "name": team_name},
        "formation": formation,
        "startXI": [
            {"player": {"id": 1, "name": "Keeper", "pos": "G"}},
            {"player": {"id": 2, "name": "Defender", "pos": "D"}},
        ],
        "substitutes": [
            {"player": {"id": 3, "name": "Sub Striker", "pos": "F"}},
        ],
    }


class TestExtractLineupPlayers(unittest.TestCase):
    def test_maps_starters_and_substitutes(self):
        rows = extract_lineup_players(100, _raw_team_lineup())

        self.assertEqual(len(rows), 3)
        starters = {row.player_id for row in rows if row.is_starter}
        subs = {row.player_id for row in rows if not row.is_starter}
        self.assertEqual(starters, {1, 2})
        self.assertEqual(subs, {3})

    def test_fields_mapped_correctly(self):
        rows = extract_lineup_players(100, _raw_team_lineup(team_id=532, team_name="Valencia", formation="4-4-2"))
        keeper = next(row for row in rows if row.player_id == 1)

        self.assertEqual(keeper.fixture_id, 100)
        self.assertEqual(keeper.player_name, "Keeper")
        self.assertEqual(keeper.team_id, 532)
        self.assertEqual(keeper.team_name, "Valencia")
        self.assertEqual(keeper.position, "G")
        self.assertTrue(keeper.is_starter)
        self.assertEqual(keeper.formation, "4-4-2")
        self.assertEqual(keeper.source, "api_sports")

    def test_skips_entries_without_player_id(self):
        raw = {
            "team": {"id": 1, "name": "X"},
            "formation": "4-4-2",
            "startXI": [{"player": {"id": None, "name": "Ghost"}}],
            "substitutes": [],
        }
        self.assertEqual(extract_lineup_players(100, raw), [])

    def test_missing_lists_default_to_empty(self):
        raw = {"team": {"id": 1, "name": "X"}, "formation": "4-4-2"}
        self.assertEqual(extract_lineup_players(100, raw), [])


def _make_in_memory_engine():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return engine


class TestPlayerLineupRepositoryWithDb(unittest.TestCase):
    """Vero SQLite in-memory, stesso principio di
    `TestPlayerDataRepositoryWithDb` in `player_event_service_test.py`."""

    def setUp(self):
        self.engine = _make_in_memory_engine()
        self.session_factory = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        self._session_local_patch = mock.patch(
            "src.repository.player_data_repository.SessionLocal", new=self.session_factory
        )
        self._session_local_patch.start()
        self.addCleanup(self._session_local_patch.stop)
        self.repo = PlayerDataRepository()

    def test_save_lineups_dedupes_duplicate_composite_key_within_same_batch(self):
        """Bug reale 2026-10-08 (fixture 1601534): API-Sports ha assegnato
        lo STESSO player_id a due giocatori DIVERSI nella stessa risposta
        (errore di qualita' dati lato provider, non un duplicato letterale)
        - la PK composita (fixture_id, player_id) non puo' rappresentare
        entrambi. Prima del fix, il bulk insert andava in UniqueViolation
        su player_lineup_pkey."""
        raw = _raw_team_lineup()
        rows = extract_lineup_players(100, raw)
        # Forza la collisione osservata in produzione: stesso player_id,
        # nomi diversi, nella stessa chiamata a save_lineups.
        colliding = PlayerLineup(
            fixture_id=100, player_id=1, player_name="Altro Giocatore",
            team_id=532, team_name="Valencia", position="F", is_starter=False, formation="4-4-2",
        )
        rows.append(colliding)

        self.repo.save_lineups(rows)  # non deve sollevare IntegrityError

        saved = self.repo.list_lineup_for_fixture(100)
        self.assertEqual(len({(row.fixture_id, row.player_id) for row in saved}), 3)

    def test_has_lineup_for_fixture_and_captured_lineup_fixture_ids(self):
        rows = extract_lineup_players(100, _raw_team_lineup())

        self.assertFalse(self.repo.has_lineup_for_fixture(100))
        self.repo.save_lineups(rows)
        self.assertTrue(self.repo.has_lineup_for_fixture(100))

        captured = self.repo.captured_lineup_fixture_ids([100, 200])
        self.assertEqual(captured, {100})

    def test_captured_lineup_fixture_ids_empty_input(self):
        self.assertEqual(self.repo.captured_lineup_fixture_ids([]), set())


if __name__ == "__main__":
    unittest.main()
