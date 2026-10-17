"""Anagrafica giocatori: sincronizzazione API-Sports, collegamenti FotMob, ricerca con data di nascita,
formazione ufficiale/prevista e apprendimento. SQLite in-memory, dati inventati."""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api import players_routes
from src.data.player.player_lineup_models import Base, PlayerLineup
from src.service_ia.fotmob.models import FotmobLineupPlayer, FotmobLineupSnapshot
from src.service_ia.players import identity as ids
from src.service_ia.players import lineups
from src.service_ia.players.models import PlayerIdentity, PlayerProviderLink

NOW = datetime(2026, 10, 17, 12, tzinfo=timezone.utc)


class _Base(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
        tables = [PlayerLineup.__table__, PlayerIdentity.__table__, PlayerProviderLink.__table__,
                  FotmobLineupSnapshot.__table__, FotmobLineupPlayer.__table__]
        Base.metadata.create_all(engine, tables=tables)
        self.s = sessionmaker(bind=engine)()

    def add_lineup(self, fixture_id, rows):
        for pid, name, team, pos, starter in rows:
            self.s.add(PlayerLineup(fixture_id=fixture_id, player_id=pid, player_name=name, team_id=team,
                                    team_name=f"T{team}", position=pos, is_starter=starter, formation="4-3-3"))
        self.s.commit()



class IdentityTest(_Base):
    def test_sync_creates_one_identity_per_api_player_and_is_idempotent(self):
        self.add_lineup(1, [(11, "L. Martinez", 10, "F", True), (12, "H. Calhanoglu", 10, "M", True)])
        self.add_lineup(2, [(11, "Lautaro Martinez", 10, "F", True)])  # stesso giocatore, nome piu' recente
        first = ids.sync_api_sports_identities(self.s)
        again = ids.sync_api_sports_identities(self.s)
        self.assertEqual(first["identities_created"], 2)
        self.assertEqual(again["identities_created"], 0)
        self.assertEqual(self.s.query(PlayerIdentity).count(), 2)
        link = ids.get_link(self.s, ids.API, 11)
        self.assertEqual((link.provider_name, link.method), ("Lautaro Martinez", "provider_id"))

    def test_fotmob_link_is_saved_reused_and_conflicts_do_not_overwrite(self):
        self.add_lineup(1, [(11, "L. Martinez", 10, "F", True), (13, "L. Martinez", 20, "F", True)])
        ids.sync_api_sports_identities(self.s)
        self.assertEqual(ids.link_fotmob_to_api(self.s, fotmob_id=900, fotmob_name="Lautaro Martinez",
                                                api_player_id=11, method="last_name_initial"), "created")
        self.assertEqual(ids.api_id_for_fotmob(self.s, 900), 11)
        # lo stesso ID FotMob proposto per un altro giocatore con fiducia piu' bassa: resta il primo
        self.assertEqual(ids.link_fotmob_to_api(self.s, fotmob_id=900, fotmob_name="x", api_player_id=13,
                                                method="surname_subset"), "conflict")
        self.assertEqual(ids.api_id_for_fotmob(self.s, 900), 11)
        self.assertIsNone(ids.api_id_for_fotmob(self.s, 12345))

    def test_manual_link_is_never_overwritten_by_automatic(self):
        self.add_lineup(1, [(11, "Rodri", 10, "M", True), (14, "Rodrigo", 10, "M", True)])
        ids.sync_api_sports_identities(self.s)
        ids.register_external_link(self.s, api_sports_id=11, fotmob_id=777, name="Rodri", source="manual")
        status = ids.link_fotmob_to_api(self.s, fotmob_id=777, fotmob_name="Rodri", api_player_id=14, method="exact_name")
        self.assertEqual(status, "protected")
        self.assertEqual(ids.api_id_for_fotmob(self.s, 777), 11)

    def test_resolve_by_id_name_and_birth_date(self):
        self.add_lineup(1, [(21, "J. Silva", 10, "M", True), (22, "J. Silva", 10, "D", True)])
        ids.sync_api_sports_identities(self.s)
        ids.register_external_link(self.s, api_sports_id=21, name="J. Silva", birth_date="1999-03-04")
        ids.register_external_link(self.s, api_sports_id=22, name="J. Silva", birth_date="2001-07-09")
        ambiguous = ids.resolve(self.s, name="Joao Silva", team_id=10)
        self.assertEqual(ambiguous["status"], "ambiguous")
        self.assertEqual(len(ambiguous["candidates"]), 2)
        found = ids.resolve(self.s, name="Joao Silva", team_id=10, birth_date="2001-07-09")
        self.assertEqual((found["status"], found["match"]["providers"]["api_sports"]["id"]), ("found", "22"))
        self.assertTrue(found["birth_date_confirmed"])
        self.assertEqual(ids.resolve(self.s, provider="api_sports", provider_id=21)["method"], "provider_id")
        self.assertEqual(ids.resolve(self.s, name="Nessuno Esiste")["status"], "not_found")
        # data diversa da tutte: nessun candidato
        self.assertEqual(ids.resolve(self.s, name="Joao Silva", birth_date="1980-01-01")["status"], "not_found")

    def official(self):
        rows = []
        for team, base in ((10, 100), (20, 200)):
            rows += [(base + i, f"Player {base + i}", team, "M", True) for i in range(11)]
        self.add_lineup(50, rows)

    def snapshot(self, mapped_missing=None, lineup_type="predicted"):
        snap = FotmobLineupSnapshot(fixture_id=50, fotmob_match_id=5, lineup_type=lineup_type, fetched_at=NOW,
                                    reliability="partially_validated", warnings=[], content_hash="h", starter_count=22,
                                    mapped_starters=21, payload={}, home_team_id=10, away_team_id=20,
                                    home_formation="4-3-3", away_formation="4-4-2")
        self.s.add(snap)
        self.s.flush()
        for side, base in (("home", 100), ("away", 200)):
            for i in range(11):
                api_id = None if (mapped_missing and base + i == mapped_missing) else base + i
                self.s.add(FotmobLineupPlayer(snapshot_id=snap.id, side=side, role="starter", fotmob_player_id=9000 + base + i,
                                              name=f"F {base + i}", api_player_id=api_id, map_method="exact_name"))
        self.s.commit()

    def test_lineup_prefers_official_but_falls_back_to_predicted(self):
        self.snapshot(mapped_missing=105)
        ids.sync_api_sports_identities(self.s)  # nessuna formazione ufficiale ancora
        predicted = lineups.lineup_for_fixture(self.s, 50)
        self.assertEqual((predicted["lineup_type"], predicted["source"]), ("predicted", "fotmob"))
        self.assertEqual(sum(len(t["starters"]) for t in predicted["teams"]), 22)
        self.assertTrue(any(p["api_sports_id"] is None for t in predicted["teams"] for p in t["starters"]))
        self.official()
        official = lineups.lineup_for_fixture(self.s, 50)
        self.assertEqual((official["lineup_type"], official["source"]), ("official", "api_sports"))
        self.assertEqual(lineups.lineup_for_fixture(self.s, 50, prefer="predicted")["lineup_type"], "predicted")
        self.assertEqual(lineups.lineup_for_fixture(self.s, 999)["teams"], [])

    def test_learns_missing_starter_by_elimination_and_confirms_the_rest(self):
        self.official()
        self.snapshot(mapped_missing=105)
        ids.sync_api_sports_identities(self.s)
        report = lineups.learn_from_official(self.s, 50)
        self.assertEqual((report["eliminated"], report["unresolved"]), (1, 0))
        self.assertEqual(report["confirmed"], 21)
        self.assertEqual(ids.api_id_for_fotmob(self.s, 9105), 105)
        self.assertEqual(ids.get_link(self.s, ids.FOTMOB, 9105).method, "official_elimination")

    def test_two_missing_starters_are_not_guessed(self):
        self.official()
        self.snapshot(mapped_missing=105)
        self.s.query(FotmobLineupPlayer).filter_by(fotmob_player_id=9106).update({"api_player_id": None})
        self.s.commit()
        ids.sync_api_sports_identities(self.s)
        report = lineups.learn_from_official(self.s, 50)
        self.assertEqual((report["eliminated"], report["unresolved"]), (0, 2))
        self.assertIsNone(ids.api_id_for_fotmob(self.s, 9105))

    def test_service_key_check(self):
        with mock.patch.dict(os.environ, {"PLAYERS_API_KEY": "s3cret"}):
            self.assertTrue(players_routes.service_key_ok("s3cret"))
            self.assertFalse(players_routes.service_key_ok("wrong"))
            self.assertFalse(players_routes.service_key_ok(None))
        with mock.patch.dict(os.environ, {"PLAYERS_API_KEY": ""}):
            self.assertFalse(players_routes.service_key_ok(""))


if __name__ == "__main__":
    unittest.main()


class SyncMapTest(_Base):
    def test_fotmob_sync_saves_the_link_and_reuses_it_without_the_name(self):
        from src.service_ia.fotmob.mapping import add_appearance, empty_roster
        from src.service_ia.fotmob.parse import ParsedPlayer
        from src.service_ia.fotmob.sync import _map

        self.add_lineup(1, [(11, "L. Martinez", 10, "F", True)])
        ids.sync_api_sports_identities(self.s)
        roster = empty_roster()
        add_appearance(roster, team_id=10, player_id=11, name="L. Martinez", position="F", last_seen=NOW)
        player = ParsedPlayer(fotmob_id=900, name="Lautaro Martinez", position_id=None, shirt=None, market_value=None)
        first = _map(self.s, player, 10, roster)
        self.assertEqual((first.api_player_id, first.method), (11, "last_name_initial"))
        self.assertEqual(ids.api_id_for_fotmob(self.s, 900), 11)
        # nome scritto in modo irriconoscibile: si usa il collegamento salvato
        renamed = ParsedPlayer(fotmob_id=900, name="El Toro", position_id=None, shirt=None, market_value=None)
        again = _map(self.s, renamed, 10, roster)
        self.assertEqual((again.api_player_id, again.method, again.position), (11, "saved_link", "F"))


class ServiceKeyMiddlewareTest(unittest.TestCase):
    def test_players_routes_need_login_or_service_key(self):
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        with mock.patch.dict(os.environ, {"PLAYERS_API_KEY": "k"}):
            self.assertEqual(client.get("/players/lookup", params={"name": "x"}).status_code, 401)
            self.assertEqual(client.get("/players/lookup", params={"name": "x"}, headers={"X-Service-Key": "bad"}).status_code, 401)
            # la chiave non apre le altre rotte
            self.assertEqual(client.get("/scorers/day", params={"date": "2026-10-17"}, headers={"X-Service-Key": "k"}).status_code, 401)
            with mock.patch.object(players_routes, "SessionLocal") as session_cls, \
                    mock.patch.object(players_routes.identity, "resolve", return_value={"status": "not_found", "candidates": []}):
                session_cls.return_value.__enter__.return_value = object()
                ok = client.get("/players/lookup", params={"name": "x"}, headers={"X-Service-Key": "k"})
            self.assertEqual((ok.status_code, ok.json()["status"]), (200, "not_found"))


class RichLineupTest(_Base):
    def snap(self):
        snap = FotmobLineupSnapshot(fixture_id=60, fotmob_match_id=6, lineup_type="predicted", fetched_at=NOW,
                                    reliability="partially_validated", warnings=[], content_hash="h", starter_count=2,
                                    mapped_starters=1, payload={}, home_team_id=10, away_team_id=20)
        self.s.add(snap)
        self.s.flush()
        rows = [
            ("home", "starter", 9001, "Mapped Guy", 11, "exact_name", "9"),
            ("home", "starter", 9002, "Unknown Guy", None, None, "27"),
            ("home", "bench", 9003, "Bench Guy", 13, "last_name_initial", "30"),
            ("home", "unavailable", 9004, "Hurt Guy", 14, "exact_name", "5"),
        ]
        for side, role, fid, name, api, method, shirt in rows:
            self.s.add(FotmobLineupPlayer(
                snapshot_id=snap.id, side=side, role=role, fotmob_player_id=fid, name=name, api_player_id=api,
                map_method=method, shirt_number=shirt,
                unavailability_type="injury" if role == "unavailable" else None,
                expected_return="A few weeks" if role == "unavailable" else None,
                return_certainty="doubtful" if role == "unavailable" else None))
        self.s.commit()

    def test_each_player_carries_status_method_confidence_shirt_and_absence(self):
        self.add_lineup(1, [(11, "Mapped Guy", 10, "M", True), (13, "Bench Guy", 10, "D", False),
                            (14, "Hurt Guy", 10, "F", False)])
        ids.sync_api_sports_identities(self.s)
        self.snap()
        out = lineups.lineup_for_fixture(self.s, 60)
        self.assertEqual((out["lineup_type"], out["unmapped_starters"]), ("predicted", 1))
        home = out["teams"][0]
        starter = {p["fotmob_id"]: p for p in home["starters"]}
        self.assertEqual((starter[9001]["status"], starter[9001]["method"], starter[9001]["confidence"]),
                         ("starter", "exact_name", 0.9))
        self.assertEqual((starter[9002]["mapped"], starter[9002]["api_sports_id"], starter[9002]["confidence"]),
                         (False, None, None))
        self.assertEqual((starter[9002]["name"], starter[9002]["shirt_number"]), ("Unknown Guy", "27"))
        self.assertEqual(home["bench"][0]["status"], "bench")
        hurt = home["unavailable"][0]
        self.assertEqual(hurt["status"], "unavailable")
        self.assertEqual((hurt["absence"]["type"], hurt["absence"]["expected_return"], hurt["absence"]["certainty"]),
                         ("injury", "A few weeks", "doubtful"))

    def test_official_players_are_certain(self):
        self.add_lineup(2, [(1000 + i, f"P{i}", 10 if i < 11 else 20, "M", True) for i in range(22)])
        ids.sync_api_sports_identities(self.s)
        p = lineups.lineup_for_fixture(self.s, 2)["teams"][0]["starters"][0]
        self.assertEqual((p["status"], p["method"], p["confidence"], p["mapped"]), ("starter", "provider_id", 1.0, True))

    def test_bulk_lineups_and_bulk_links(self):
        self.add_lineup(1, [(11, "A. One", 10, "M", True), (12, "B. Two", 10, "M", True)])
        ids.sync_api_sports_identities(self.s)
        bulk = lineups.lineups_for_fixtures(self.s, [1, 1, 999])
        self.assertEqual(bulk["count"], 2)
        self.assertEqual([f["fixture_id"] for f in bulk["fixtures"]], [1, 999])
        out = ids.register_external_links(self.s, [
            {"api_sports_id": 11, "fotmob_id": 500, "name": "Alpha One", "birth_date": "2000-01-02"},
            {"api_sports_id": 12, "fotmob_id": 500, "name": "Beta Two"},   # stesso fotmob_id su un'altra persona
            {"api_sports_id": None},
        ])
        self.assertEqual(out["received"], 3)
        self.assertEqual(out["errors"], 1)
        self.assertEqual(ids.api_id_for_fotmob(self.s, 500), 12)  # il manuale/esterno piu' recente vince, nessun errore
