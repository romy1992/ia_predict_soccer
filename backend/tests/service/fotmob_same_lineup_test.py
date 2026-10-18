"""Una formazione prevista identica non genera una nuova revisione (i portieri non sono nelle righe salvate)."""

from __future__ import annotations

import unittest

from src.service_ia.fotmob.predict import _same_saved_lineup


def starters():
    out = [{"player_id": 1, "position": "G"}, {"player_id": 12, "position": "G"}]
    out += [{"player_id": 100 + i, "position": ("D" if i < 4 else "M" if i < 8 else "F")} for i in range(20)]
    return out[:22]


class SameSavedLineupTest(unittest.TestCase):
    def test_same_eleven_without_goalkeepers_is_skipped(self):
        s = starters()
        saved = {int(p["player_id"]) for p in s if p["position"] != "G"}
        self.assertTrue(_same_saved_lineup((2, saved, "predicted"), s))

    def test_changed_player_is_not_skipped(self):
        s = starters()
        saved = {int(p["player_id"]) for p in s if p["position"] != "G"}
        saved.discard(100)
        saved.add(999)
        self.assertFalse(_same_saved_lineup((2, saved, "predicted"), s))

    def test_official_or_missing_previous_is_not_a_skip_here(self):
        s = starters()
        saved = {int(p["player_id"]) for p in s if p["position"] != "G"}
        self.assertFalse(_same_saved_lineup((2, saved, "official"), s))
        self.assertFalse(_same_saved_lineup(None, s))

    def test_without_roles_falls_back_to_exact_check_later(self):
        s = [{"player_id": i, "position": None} for i in range(22)]
        self.assertFalse(_same_saved_lineup((1, set(range(22)), "predicted"), s))


if __name__ == "__main__":
    unittest.main()
