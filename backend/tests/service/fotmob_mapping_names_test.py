"""Abbinamento nomi FotMob -> API-Sports: casi osservati nei dati reali (cognomi troncati, ordine invertito,
omonimi nella stessa rosa) e lettere che NFKD non scompone."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.service_ia.fotmob.mapping import add_appearance, empty_roster, map_player
from src.service_ia.fotmob.names import normalize_name

TEAM = 10
NOW = datetime(2026, 10, 15, tzinfo=timezone.utc)


def roster(*players):
    r = empty_roster()
    for pid, name, days_ago in players:
        add_appearance(r, team_id=TEAM, player_id=pid, name=name, position="M", last_seen=NOW - timedelta(days=days_ago))
    return r


class MappingNamesTest(unittest.TestCase):
    def test_special_letters_are_transliterated(self):
        self.assertEqual(normalize_name("Martin Ødegaard"), "martin odegaard")
        self.assertEqual(normalize_name("Łukasz Fabiański"), "lukasz fabianski")
        r = roster((1, "M. Odegaard", 3))
        self.assertEqual(map_player("Martin Ødegaard", TEAM, r).api_player_id, 1)

    def test_truncated_second_surname_matches_when_unique(self):
        r = roster((1, "N. Aséko Nkili", 3), (2, "F. Redondo Solari", 3), (3, "X. Mandza Tsiendi", 3))
        self.assertEqual(map_player("Noël Aséko", TEAM, r).method, "surname_subset")
        self.assertEqual(map_player("Federico Redondo", TEAM, r).api_player_id, 2)
        self.assertEqual(map_player("Xavier Mandza", TEAM, r).api_player_id, 3)

    def test_inverted_name_order_matches(self):
        r = roster((7, "Lee Kang In", 3))
        m = map_player("Kang-In Lee", TEAM, r)
        self.assertEqual((m.api_player_id, m.method), (7, "name_tokens"))

    def test_namesakes_use_recency_only_when_one_is_long_gone(self):
        recent_vs_old = roster((1, "C. Romero", 5), (2, "C. Romero", 400))
        m = map_player("Cristian Romero", TEAM, recent_vs_old)
        self.assertEqual((m.api_player_id, m.warning), (1, "recency_pick"))
        both_active = roster((1, "C. Romero", 5), (2, "C. Romero", 20))
        self.assertEqual(map_player("Cristian Romero", TEAM, both_active).warning, "ambiguous")

    def test_ambiguous_subset_and_unknown_stay_unmapped(self):
        r = roster((1, "J. Silva Alves", 3), (2, "J. Silva Costa", 3))
        self.assertIsNone(map_player("Joao Silva", TEAM, r).api_player_id)
        self.assertEqual(map_player("Andrés Lopez", TEAM, r).warning, "unmapped")

    def test_existing_exact_and_initial_matches_are_unchanged(self):
        r = roster((1, "Lautaro Martinez", 3), (2, "H. Calhanoglu", 3))
        self.assertEqual(map_player("Lautaro Martinez", TEAM, r).method, "exact_name")
        self.assertEqual(map_player("Hakan Calhanoglu", TEAM, r).method, "last_name_initial")


if __name__ == "__main__":
    unittest.main()
