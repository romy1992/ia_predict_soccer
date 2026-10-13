"""Test per il cantiere "giocatori che segnano" (storico eventi-per-
giocatore) - stesso stile di `live_data_service_test.py::TestPureMapping`:
funzioni pure di mapping, nessun DB/rete. Piu' in basso, stesso principio di
`TestLiveDataRepositoryWithDb`: vero SQLite in-memory per il repository."""

from __future__ import annotations

import unittest
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.data.player.player_models as player_models  # noqa: F401  (registra player_match_event su Base.metadata)
from src.data.player.player_models import Base, player_event_id
from src.data.player.player_event_service import extract_player_event
from src.repository.player_data_repository import PlayerDataRepository


def _raw_event(elapsed=34, event_type="Goal", detail="Normal Goal", team_id=33, player_id=1, player_name="Player X"):
    return {
        "time": {"elapsed": elapsed, "extra": None},
        "team": {"id": team_id, "name": "Home FC"},
        "player": {"id": player_id, "name": player_name},
        "assist": {"id": 2, "name": "Assist Man"},
        "type": event_type,
        "detail": detail,
        "comments": None,
    }


class TestExtractPlayerEvent(unittest.TestCase):
    def test_maps_fields_and_id_is_deterministic(self):
        raw = _raw_event(elapsed=34, event_type="Goal", detail="Normal Goal", team_id=33, player_id=1, player_name="Player X")
        event_a = extract_player_event(100, raw)
        event_b = extract_player_event(100, raw)

        self.assertEqual(event_a.id_event, event_b.id_event)
        self.assertEqual(event_a.fixture_id, 100)
        self.assertEqual(event_a.elapsed_minute, 34)
        self.assertEqual(event_a.event_type, "Goal")
        self.assertEqual(event_a.event_detail, "Normal Goal")
        self.assertEqual(event_a.team_id, 33)
        self.assertEqual(event_a.player_id, 1)
        self.assertEqual(event_a.player_name, "Player X")
        self.assertEqual(event_a.assist_id, 2)
        self.assertEqual(event_a.assist_name, "Assist Man")
        self.assertEqual(event_a.source, "api_sports")

    def test_id_stable_even_without_player_id(self):
        """Fallback sul nome quando l'id giocatore manca (es. sostituzioni
        con payload incompleto) - l'id resta comunque deterministico."""
        raw = _raw_event(player_id=None, player_name="Solo Nome")
        event_a = extract_player_event(100, raw)
        event_b = extract_player_event(100, raw)
        self.assertIsNone(event_a.player_id)
        self.assertEqual(event_a.id_event, event_b.id_event)

    def test_player_event_id_differs_for_different_players_same_fixture_minute(self):
        """Due gol nello stesso minuto/fixture ma di giocatori diversi non
        devono collassare sullo stesso id (motivo per cui si include
        `player_id` nell'hash, a differenza di `live_event_id`)."""
        id_player_1 = player_event_id(100, "Goal", "Normal Goal", 34, None, 33, 1, "Player X")
        id_player_2 = player_event_id(100, "Goal", "Normal Goal", 34, None, 33, 2, "Player Y")
        self.assertNotEqual(id_player_1, id_player_2)

    def test_player_event_id_differs_for_different_events(self):
        id_goal = player_event_id(100, "Goal", "Normal Goal", 34, None, 33, 1, "Player X")
        id_card = player_event_id(100, "Card", "Yellow Card", 34, None, 33, 1, "Player X")
        id_other_fixture = player_event_id(101, "Goal", "Normal Goal", 34, None, 33, 1, "Player X")
        self.assertNotEqual(id_goal, id_card)
        self.assertNotEqual(id_goal, id_other_fixture)


def _make_in_memory_engine():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return engine


class TestPlayerDataRepositoryWithDb(unittest.TestCase):
    """Vero SQLite in-memory (non mock) - stesso principio di
    `live_data_service_test.py::TestLiveDataRepositoryWithDb`."""

    def setUp(self):
        self.engine = _make_in_memory_engine()
        self.session_factory = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        self._session_local_patch = mock.patch(
            "src.repository.player_data_repository.SessionLocal", new=self.session_factory
        )
        self._session_local_patch.start()
        self.addCleanup(self._session_local_patch.stop)
        self.repo = PlayerDataRepository()

    def test_save_events_dedupes_duplicate_id_within_same_batch(self):
        """Bug reale 2026-10-07: API-Sports ha restituito lo STESSO evento
        due volte nella stessa risposta (stesso id deterministico) - prima
        del fix, il bulk insert andava in UniqueViolation su `id_event`
        perche' `session.merge` controlla solo il DB, non gli altri oggetti
        gia' mersi in questa sessione non ancora flushata."""
        raw = _raw_event(elapsed=90, event_type="Card", detail="Yellow Card", team_id=548, player_id=61774, player_name="Orri")
        duplicated_payload = [raw, dict(raw)]  # stesso identico evento, due volte

        events = [extract_player_event(1570401, item) for item in duplicated_payload]
        self.assertEqual(len({e.id_event for e in events}), 1)  # stesso id, come osservato in produzione

        self.repo.save_events(events)  # non deve sollevare IntegrityError

        rows = self.repo.list_events_for_fixture(1570401)
        self.assertEqual(len(rows), 1)

    def test_has_events_for_fixture_and_merge_is_idempotent_across_calls(self):
        raw = _raw_event(elapsed=34, event_type="Goal", player_id=1, player_name="Player X")
        event = extract_player_event(100, raw)

        self.assertFalse(self.repo.has_events_for_fixture(100))
        self.repo.save_events([event])
        self.assertTrue(self.repo.has_events_for_fixture(100))

        self.repo.save_events([event])  # ri-processing: stesso id, upsert sul posto
        rows = self.repo.list_events_for_fixture(100)
        self.assertEqual(len(rows), 1)

    def test_captured_fixture_ids_bulk_check(self):
        """Performance fix 2026-10-07: il backfill deve poter scartare
        migliaia di fixture gia' catturate con UNA query, non una per
        fixture (`has_events_for_fixture` in loop impiegava minuti solo
        per lo skip, prima di toccare anche una sola fixture nuova)."""
        event_100 = extract_player_event(100, _raw_event(player_id=1, player_name="Player X"))
        event_200 = extract_player_event(200, _raw_event(player_id=2, player_name="Player Y"))
        self.repo.save_events([event_100, event_200])

        captured = self.repo.captured_fixture_ids([100, 200, 300])

        self.assertEqual(captured, {100, 200})

    def test_captured_fixture_ids_empty_input(self):
        self.assertEqual(self.repo.captured_fixture_ids([]), set())


if __name__ == "__main__":
    unittest.main()
