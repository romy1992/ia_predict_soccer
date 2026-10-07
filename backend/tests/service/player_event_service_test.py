"""Test per il cantiere "giocatori che segnano" (storico eventi-per-
giocatore) - stesso stile di `live_data_service_test.py::TestPureMapping`:
funzioni pure di mapping, nessun DB/rete."""

from __future__ import annotations

import unittest

from src.data.player.player_event_service import extract_player_event
from src.data.player.player_models import player_event_id


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


if __name__ == "__main__":
    unittest.main()
