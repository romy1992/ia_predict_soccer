import unittest
from unittest.mock import patch

from src.service_ia.model.match import Match
from src.service_ia.pre_processing.download_match_service import calculate_mean, download_import_matches


class FakeProvider:
    def __init__(self, fixtures=None, statistics=None, odds=None):
        self._fixtures = fixtures or []
        self._statistics = statistics or []
        self._odds = odds or []

    def get_fixtures(self, **params):
        return list(self._fixtures)

    def get_fixture_statistics(self, fixture_id):
        return list(self._statistics)

    def get_fixture_odds(self, fixture_id):
        return list(self._odds)


def _sample_fixture(fixture_id=1326590, status="FT"):
    return {
        "fixture": {
            "id": fixture_id,
            "date": "2026-09-01T18:45:00+00:00",
            "referee": "Referee Test",
            "status": {"short": status},
        },
        "league": {"id": 135, "round": "Regular Season - 1"},
        "teams": {
            "home": {"id": 505, "name": "Inter"},
            "away": {"id": 487, "name": "Lazio"},
        },
        "score": {
            "halftime": {"home": 1, "away": 0},
            "fulltime": {"home": 2, "away": 1},
        },
    }


def _sample_statistics():
    return [
        {
            "team": {"id": 505},
            "statistics": [
                {"type": "Shots on Goal", "value": 5},
                {"type": "Total Shots", "value": 11},
                {"type": "Corner Kicks", "value": 7},
            ],
        },
        {
            "team": {"id": 487},
            "statistics": [
                {"type": "Shots on Goal", "value": 3},
                {"type": "Total Shots", "value": 9},
                {"type": "Corner Kicks", "value": 4},
            ],
        },
    ]


def _sample_odds_payload():
    return [
        {
            "update": "2026-09-01T17:00:00+00:00",
            "bookmakers": [
                {
                    "name": "BookA",
                    "bets": [
                        {
                            "id": 1,
                            "name": "Match Winner",
                            "values": [
                                {"value": "Home", "odd": "1.80"},
                                {"value": "Draw", "odd": "3.20"},
                                {"value": "Away", "odd": "4.20"},
                            ],
                        }
                    ],
                }
            ],
        }
    ]


class TestDownloadMatch(unittest.TestCase):
    @patch("src.service_ia.pre_processing.download_match_service.BET_BOOKMAKERS", [{"id": 1}])
    @patch("src.service_ia.pre_processing.download_match_service.form_last_5_tot", return_value=None)
    @patch("src.service_ia.pre_processing.download_match_service.repo_snapshot.save_many")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.save_all")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.save")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.filter_by")
    def test_download_import_matches_insert_report(
        self,
        mock_filter_by,
        mock_save,
        mock_save_all,
        mock_snapshot_save,
        _mock_form,
    ):
        mock_filter_by.return_value.first.return_value = None
        provider = FakeProvider(
            fixtures=[_sample_fixture()],
            statistics=_sample_statistics(),
            odds=_sample_odds_payload(),
        )

        report = download_import_matches(
            seasons=[2026],
            leagues=[135],
            fixture_date="2026-09-01",
            statuses="FT",
            provider=provider,
        )

        self.assertEqual(report["fixtures_seen"], 1)
        self.assertEqual(report["inserted"], 1)
        self.assertEqual(report["updated"], 0)
        self.assertEqual(report["failed"], 0)
        mock_save.assert_not_called()
        mock_save_all.assert_called_once()
        mock_snapshot_save.assert_called_once()

    @patch("src.service_ia.pre_processing.download_match_service.BET_BOOKMAKERS", [{"id": 1}])
    @patch("src.service_ia.pre_processing.download_match_service.form_last_5_tot", return_value=None)
    @patch("src.service_ia.pre_processing.download_match_service.repo_snapshot.save_many")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.save_all")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.save")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.filter_by")
    def test_download_import_matches_update_report(
        self,
        mock_filter_by,
        mock_save,
        _mock_save_all,
        _mock_snapshot_save,
        _mock_form,
    ):
        existing = Match(
            id_match_fk="existing-id",
            id_fixture=1326590,
            id_team_home=505,
            id_team_away=487,
            name_home="Inter",
            name_away="Lazio",
            date_match="2026-09-01T18:45:00+00:00",
            status="FT",
            season=2026,
            current_league=135,
            league_match=135,
            round="Regular Season - 1",
        )
        mock_filter_by.return_value.first.return_value = existing

        provider = FakeProvider(
            fixtures=[_sample_fixture()],
            statistics=_sample_statistics(),
            odds=_sample_odds_payload(),
        )

        report = download_import_matches(
            seasons=[2026],
            leagues=[135],
            fixture_date="2026-09-01",
            statuses="FT",
            provider=provider,
        )

        self.assertEqual(report["inserted"], 0)
        self.assertEqual(report["updated"], 1)
        self.assertEqual(report["failed"], 0)
        mock_save.assert_called_once()

    @patch("src.service_ia.pre_processing.download_match_service.BET_BOOKMAKERS", [{"id": 1}])
    @patch("src.service_ia.pre_processing.download_match_service.form_last_5_tot", return_value=None)
    @patch("src.service_ia.pre_processing.download_match_service.repo_snapshot.save_many")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.save_all")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.save")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.filter_by")
    def test_download_import_matches_continue_on_fixture_error(
        self,
        mock_filter_by,
        _mock_save,
        _mock_save_all,
        _mock_snapshot_save,
        _mock_form,
    ):
        mock_filter_by.return_value.first.return_value = None
        broken_fixture = {"fixture": {"id": 999}, "league": {"id": 135}, "teams": {}}
        provider = FakeProvider(
            fixtures=[broken_fixture, _sample_fixture(1326591)],
            statistics=_sample_statistics(),
            odds=_sample_odds_payload(),
        )

        report = download_import_matches(
            seasons=[2026],
            leagues=[135],
            fixture_date="2026-09-01",
            statuses="FT",
            provider=provider,
        )

        self.assertEqual(report["fixtures_seen"], 2)
        self.assertGreaterEqual(report["failed"], 1)
        self.assertGreaterEqual(report["inserted"], 1)

    @patch("src.service_ia.pre_processing.download_match_service.repo_match.massive_update_bulk")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.search_filter")
    def test_calculate_mean_runs_bulk_update(self, mock_search_filter, mock_update_bulk):
        match = Match(
            id_match_fk="m1",
            id_fixture=1,
            id_team_home=10,
            id_team_away=20,
            name_home="A",
            name_away="B",
            date_match="2026-01-01T12:00:00+00:00",
            season=2026,
            status="FT",
        )
        match.statistics = []

        match_2 = Match(
            id_match_fk="m2",
            id_fixture=2,
            id_team_home=10,
            id_team_away=30,
            name_home="A",
            name_away="C",
            date_match="2026-01-02T12:00:00+00:00",
            season=2026,
            status="FT",
        )
        match_2.statistics = []

        mock_search_filter.return_value = [match, match_2]

        calculate_mean(with_season=2026, force_mean=True, teams=[10])

        mock_search_filter.assert_called()
        mock_update_bulk.assert_called()


if __name__ == "__main__":
    unittest.main()
