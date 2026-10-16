"""FotMob: parsing, validazione, job giornaliero. Nessuna chiamata di rete."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.data.player.player_lineup_models import PlayerLineup
from src.data.player.player_models import PlayerMatchEvent
from src.data.player.scorer_prediction_models import PlayerScorerPrediction
from src.service_ia.fotmob.client import FotmobAuthError, FotmobClient, FotmobUnavailableError
from src.service_ia.fotmob.models import FotmobLineupPlayer, FotmobLineupSnapshot, FotmobMatchMap
from src.service_ia.fotmob.names import names_match
from src.service_ia.fotmob.parse import iter_matches, parse_lineup, return_certainty
from src.service_ia.fotmob.sync import sync_predicted_lineups
from src.service_ia.model.match import Base, Match
from src.service_ia.scorer import prediction_service
from tests.service.player_scoring_dataset_test import _events, _lineup, _lineups, _matches
from tests.service.scorer_prediction_test import _bundle


NOW = datetime(2026, 10, 16, 8, 0, tzinfo=timezone.utc)
KICK = (NOW + timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _xi(base: int, names: list[str]) -> list[dict]:
    return [{"id": base + index, "name": name, "positionId": 21, "shirtNumber": str(index)} for index, name in enumerate(names, start=1)]


HOME_NAMES = [f"P{100 + index}" for index in range(1, 12)]
AWAY_NAMES = [f"P{200 + index}" for index in range(1, 12)]


def _lineup_payload(**overrides) -> dict:
    body = {
        "lineup": {
            "lineupType": "predicted",
            "source": "enetpulse",
            "homeTeam": {
                "id": 9885, "name": "Juventus", "formation": "4-2-3-1",
                "starters": _xi(7000, HOME_NAMES),
                "unavailable": [],
                "coach": {"name": "Allenatore Casa"},
            },
            "awayTeam": {
                "id": 8543, "name": "Lazio", "formation": "4-3-3",
                "starters": _xi(8000, AWAY_NAMES),
                "unavailable": [{
                    "id": 846033, "name": "Vinicius Junior",
                    "unavailability": {
                        "type": "suspension",
                        "expectedReturn": "Early November 2026",
                        "expectedReturnDate": "2026-11-08T17:00:00Z",
                    },
                }],
            },
        }
    }
    body["lineup"].update(overrides)
    return body


def _by_date() -> dict:
    return {"leagues": [
        {"id": 55, "name": "Serie A", "matches": [{
            "id": 5749705,
            "home": {"id": 9885, "name": "Juventus"},
            "away": {"id": 8543, "name": "Lazio"},
            "status": {"utcTime": KICK},
        }]},
        {"id": 53, "name": "Ligue 1", "matches": [{
            "id": 5900001,
            "home": {"id": 9847, "name": "Paris Saint-Germain"},
            "away": {"id": 9829, "name": "Marseille"},
            "status": {"utcTime": KICK},
        }]},
    ]}


class _Memory:
    def setUp(self):
        engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine, tables=[
            Match.__table__, PlayerLineup.__table__, PlayerMatchEvent.__table__,
            PlayerScorerPrediction.__table__, FotmobMatchMap.__table__,
            FotmobLineupSnapshot.__table__, FotmobLineupPlayer.__table__,
        ])
        self.session = sessionmaker(bind=engine)()

    def _history(self):
        for row in _matches().itertuples():
            if row.fixture_id == 5:
                continue
            self.session.add(Match(
                id_fixture=row.fixture_id, date_match=row.date_match, league_match=135, season=2025,
                id_team_home=row.home_id, id_team_away=row.away_id, status="FT",
                name_home="Juventus", name_away="Lazio",
            ))
        for row in _lineups().itertuples():
            if row.fixture_id == 5:
                continue
            self.session.add(PlayerLineup(
                fixture_id=row.fixture_id, player_id=row.player_id, team_id=row.team_id,
                position=row.position, is_starter=bool(row.is_starter),
                player_name=f"P{row.player_id}", team_name="Juventus" if row.team_id == 1 else "Lazio",
                source="api_sports",
            ))
        for index, row in enumerate(_events().itertuples()):
            if row.fixture_id == 5:
                continue
            self.session.add(PlayerMatchEvent(
                id_event=f"e{index}", fixture_id=row.fixture_id, event_type=row.event_type,
                event_detail=row.event_detail, comments=row.comments, elapsed_minute=row.elapsed,
                team_id=row.team_id, player_id=row.player_id, assist_id=row.assist_id,
            ))

    def _upcoming(self):
        self.session.add(Match(
            id_fixture=99, date_match=KICK, league_match=135, season=2026, status="NS",
            id_team_home=1, id_team_away=2, name_home="Juventus", name_away="Lazio",
        ))
        self.session.commit()


class FotmobParseTest(unittest.TestCase):
    def test_by_date_reads_serie_a_and_ligue_1(self):
        matches = iter_matches(_by_date())
        self.assertEqual([item.match_id for item in matches], [5749705, 5900001])
        self.assertEqual(matches[1].league_name, "Ligue 1")
        self.assertEqual(matches[1].home_name, "Paris Saint-Germain")

    def test_predicted_lineup_keeps_starters_and_uncertain_absence(self):
        payload = _lineup_payload()
        payload["lineup"]["awayTeam"]["unavailable"].append({
            "id": 1, "name": "Doubtful Player",
            "unavailability": {"type": "injury", "expectedReturn": "Doubtful"},
        })
        parsed = parse_lineup(payload)
        self.assertEqual(parsed.lineup_type, "predicted")
        self.assertEqual(len(parsed.home.starters), 11)
        self.assertEqual(parsed.home.starters[0].name, "P101")
        dated = parsed.away.unavailable[0]
        self.assertEqual(dated.return_certainty, "dated")
        self.assertEqual(return_certainty("Back in training", None), "uncertain")
        self.assertEqual(return_certainty("A few weeks", None), "uncertain")
        self.assertEqual(parsed.away.unavailable[1].return_certainty, "uncertain")

    def test_confirmed_is_not_predicted_and_unknown_stays_unknown(self):
        confirmed = parse_lineup(_lineup_payload(lineupType="confirmed"))
        other = parse_lineup(_lineup_payload(lineupType="provisional"))
        self.assertEqual(confirmed.lineup_type, "confirmed")
        self.assertEqual(other.lineup_type, "unknown")

    def test_missing_lineup_is_none(self):
        self.assertIsNone(parse_lineup({"lineup": None}))
        self.assertIsNone(parse_lineup(None))


class FotmobNameTest(unittest.TestCase):
    def test_aliases_match_and_city_does_not_match_united(self):
        self.assertTrue(names_match("Bayern München", "Bayern Munich"))
        self.assertTrue(names_match("Paris Saint-Germain FC", "Paris Saint Germain"))
        self.assertTrue(names_match("Atlético Madrid", "Atletico Madrid"))
        self.assertFalse(names_match("Manchester United", "Manchester City"))


class FotmobClientTest(unittest.TestCase):
    def test_retries_429_then_stops_without_leaking_the_key(self):
        calls = []

        def transport(path, params):
            calls.append(path)
            return 429, {"message": "slow down"}

        client = FotmobClient(api_key="super-secret-key", transport=transport, sleep=lambda _seconds: None, max_retries=3)
        with self.assertRaises(FotmobUnavailableError) as raised:
            client.matches_by_date("20261018")
        self.assertNotIn("super-secret-key", str(raised.exception))
        self.assertEqual(len(calls), 3)

    def test_401_is_not_retried(self):
        calls = []

        def transport(path, params):
            calls.append(1)
            return 401, {}

        client = FotmobClient(api_key="super-secret-key", transport=transport, sleep=lambda _seconds: None)
        with self.assertRaises(FotmobAuthError) as raised:
            client.lineups(5749705)
        self.assertEqual(calls, [1])
        self.assertNotIn("super-secret-key", str(raised.exception))


class FotmobSyncTest(_Memory, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._history()
        self._upcoming()

    def _client(self, lineup=None):
        payload = _lineup_payload() if lineup is None else lineup

        def transport(path, params):
            if path.endswith("/lineups"):
                return 200, payload
            return 200, _by_date()

        return FotmobClient(api_key="test-key", transport=transport, sleep=lambda _seconds: None)

    def test_validated_snapshot_does_not_write_official_lineup_table(self):
        report = sync_predicted_lineups(
            self.session, self._client(), now=NOW, load_models=False, refresh_hours=20,
        )
        self.assertEqual(report["validated"], 1)
        self.assertEqual(report["snapshots"], 1)
        self.assertEqual(report["requests"], 2)
        official = self.session.execute(select(PlayerLineup).where(PlayerLineup.fixture_id == 99)).scalars().all()
        self.assertEqual(official, [])
        snapshot = self.session.execute(select(FotmobLineupSnapshot)).scalar_one()
        self.assertEqual(snapshot.lineup_type, "predicted")
        self.assertEqual(snapshot.fotmob_match_id, 5749705)
        self.assertEqual(snapshot.reliability, "validated")
        players = self.session.execute(select(FotmobLineupPlayer)).scalars().all()
        starters = [row for row in players if row.role == "starter"]
        self.assertEqual(len(starters), 22)
        self.assertTrue(all(row.api_player_id is not None for row in starters))
        self.assertTrue(any(row.role == "unavailable" and row.return_certainty == "dated" for row in players))

    def test_second_run_the_same_day_does_not_call_the_provider_again(self):
        sync_predicted_lineups(self.session, self._client(), now=NOW, load_models=False)
        again = sync_predicted_lineups(
            self.session, self._client(), now=NOW + timedelta(hours=2), load_models=False,
        )
        self.assertEqual(again["skipped_fresh"], 1)
        self.assertEqual(again["requests"], 0)
        self.assertEqual(self.session.execute(select(FotmobLineupSnapshot)).scalars().all().__len__(), 1)

    def test_short_lineup_is_rejected_and_not_a_prediction_source(self):
        payload = _lineup_payload()
        payload["lineup"]["homeTeam"]["starters"] = payload["lineup"]["homeTeam"]["starters"][:10]
        report = sync_predicted_lineups(self.session, self._client(payload), now=NOW, load_models=False)
        self.assertEqual(report["rejected"], 1)
        self.assertEqual(report["predicted_scorers"], 0)
        snapshot = self.session.execute(select(FotmobLineupSnapshot)).scalar_one()
        self.assertIn("casa:titolari_10", snapshot.warnings)

    def test_starter_also_unavailable_is_rejected(self):
        payload = _lineup_payload()
        starter = payload["lineup"]["homeTeam"]["starters"][0]
        payload["lineup"]["homeTeam"]["unavailable"] = [{
            "id": starter["id"], "name": starter["name"],
            "unavailability": {"type": "injury", "expectedReturn": "Doubtful"},
        }]
        report = sync_predicted_lineups(self.session, self._client(payload), now=NOW, load_models=False)
        self.assertEqual(report["rejected"], 1)

    def test_empty_provider_lineup_keeps_the_job_alive(self):
        report = sync_predicted_lineups(self.session, self._client({"lineup": None}), now=NOW, load_models=False)
        self.assertEqual(report["no_lineup"], 1)
        self.assertEqual(report["rejected"], 1)
        self.assertEqual(self.session.execute(select(PlayerLineup).where(PlayerLineup.fixture_id == 99)).scalars().all(), [])

    def test_wrong_team_in_the_lineup_drops_the_match_map(self):
        payload = _lineup_payload()
        payload["lineup"]["homeTeam"]["name"] = "Real Madrid"
        sync_predicted_lineups(self.session, self._client(payload), now=NOW, load_models=False)
        self.assertIsNone(self.session.get(FotmobMatchMap, 99))
        snapshot = self.session.execute(select(FotmobLineupSnapshot)).scalar_one()
        self.assertEqual(snapshot.reliability, "rejected")
        self.assertIn("identita_partita", snapshot.warnings)


class FotmobInitialPredictionTest(_Memory, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._history()
        self._upcoming()
        history = __import__("src.data.player.scoring_dataset", fromlist=["build_dataset"]).build_dataset(
            _matches().iloc[:4], _lineups().query("fixture_id<=4"), _events().query("fixture_id<=4"),
        )
        self.bundle = _bundle(history)

    def _client(self):
        def transport(path, params):
            if path.endswith("/lineups"):
                return 200, _lineup_payload()
            return 200, _by_date()

        return FotmobClient(api_key="test-key", transport=transport, sleep=lambda _seconds: None)

    def test_validated_xi_writes_predicted_revision_and_official_still_appends(self):
        report = sync_predicted_lineups(
            self.session, self._client(), now=NOW, load_models=False, scorer_bundle=self.bundle, cards_bundle=None,
        )
        self.assertEqual(report["predicted_scorers"], 1)
        rows = self.session.execute(select(PlayerScorerPrediction)).scalars().all()
        self.assertTrue(rows)
        self.assertTrue(all(row.lineup_type == "predicted" and row.revision == 1 for row in rows))
        for row in _lineup(99).itertuples():
            self.session.add(PlayerLineup(
                fixture_id=99, player_id=row.player_id, team_id=row.team_id, position=row.position,
                is_starter=bool(row.is_starter), player_name=f"P{row.player_id}", source="api_sports",
            ))
        self.session.commit()
        official = prediction_service.predict_upcoming(
            self.session, self.bundle, leagues=[135], now=NOW + timedelta(hours=28), finished_statuses=["FT"],
        )
        self.assertEqual(official["predicted"], 1)
        revs = {row.revision for row in self.session.execute(select(PlayerScorerPrediction)).scalars()}
        types = {row.lineup_type for row in self.session.execute(select(PlayerScorerPrediction)).scalars()}
        self.assertEqual(revs, {1, 2})
        self.assertEqual(types, {"predicted", "official"})


if __name__ == "__main__":
    unittest.main()
