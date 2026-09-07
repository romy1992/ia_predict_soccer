import unittest
from unittest.mock import patch

from src.service_ia.model.match import Match, Statistics
from src.service_ia.pre_processing.api_sports_provider import ApiSportsQuotaExceededError
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

    @patch("src.service_ia.pre_processing.download_match_service.repo_match.massive_update_bulk")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.search_filter")
    def test_calculate_mean_with_teams_does_not_recompute_opponent_mean(self, mock_search_filter, mock_update_bulk):
        """Bug fix 2026-09-07 (era un TODO esplicito nel codice): con
        `teams=[10]`, il filtro OR usato per popolare `all_match` include
        anche le partite dell'avversario 20 (andata/ritorno contro la 10).
        PRIMA del fix, 20 finiva comunque in `id_teams` e la sua media
        veniva ricalcolata usando SOLO le partite contro la 10 (sottoinsieme
        parziale, mai lo storico stagionale reale di 20) - sovrascrivendo un
        valore sbagliato. Con `teams` specificato il ricalcolo deve restare
        limitato ESATTAMENTE alle squadre richieste: per il match di
        ritorno, `mean_statistics` deve contenere SOLO l'aggiornamento della
        10 (un dict singolo), MAI una lista con dentro anche la 20."""
        match_1 = Match(
            id_match_fk="m1", id_fixture=1, id_team_home=10, id_team_away=20,
            name_home="A", name_away="B", date_match="2026-01-01T12:00:00+00:00",
            season=2026, status="FT",
        )
        match_1.statistics = [
            Statistics(id_statistics_fk="s1", statistics_team_id=10),
            Statistics(id_statistics_fk="s2", statistics_team_id=20),
        ]

        match_2 = Match(
            id_match_fk="m2", id_fixture=2, id_team_home=20, id_team_away=10,
            name_home="B", name_away="A", date_match="2026-02-01T12:00:00+00:00",
            season=2026, status="FT",
        )
        match_2.statistics = [
            Statistics(id_statistics_fk="s3", statistics_team_id=20),
            Statistics(id_statistics_fk="s4", statistics_team_id=10),
        ]

        mock_search_filter.return_value = [match_1, match_2]

        calculate_mean(with_season=2026, force_mean=True, teams=[10])

        list_obj = mock_update_bulk.call_args[0][0]
        entry_m2 = next(e for e in list_obj if e["id_match_fk"] == "m2")
        mean_stats = entry_m2["mean_statistics"]

        self.assertIsInstance(mean_stats, dict)  # MAI una lista (che indicherebbe il ricalcolo anche di 20)
        self.assertEqual(mean_stats["id_team"], 10)


class FakeProviderQuotaExceededOnOdds(FakeProvider):
    """Simula la quota GIORNALIERA che si esaurisce DURANTE il processing
    di una fixture (su `get_fixture_odds`), non durante `get_fixtures`."""

    def get_fixture_odds(self, fixture_id):
        raise ApiSportsQuotaExceededError("quota esaurita (test)")


class TestDownloadImportMatchesQuotaExceededMidway(unittest.TestCase):
    """BUGFIX 2026-09-06: quando la quota si esaurisce DENTRO il loop
    per-fixture (get_fixture_statistics/get_fixture_odds), PRIMA veniva
    inghiottita dal blocco `except Exception` generico e il loop
    CONTINUAVA su tutte le fixture/leghe rimanenti senza mai impostare
    `quota_exceeded=True` - causa principale per cui "Aggiorna tutto"
    restava "in esecuzione" per decine di minuti/ore invece di fermarsi
    subito, come gia' avveniva per l'eccezione sollevata da
    `get_fixtures`."""

    @patch("src.service_ia.pre_processing.download_match_service.BET_BOOKMAKERS", [{"id": 1}])
    @patch("src.service_ia.pre_processing.download_match_service.form_last_5_tot", return_value=None)
    @patch("src.service_ia.pre_processing.download_match_service.repo_snapshot.save_many")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.save_all")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.save")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.filter_by")
    def test_stops_immediately_and_flags_quota_exceeded(
        self,
        mock_filter_by,
        _mock_save,
        _mock_save_all,
        _mock_snapshot_save,
        _mock_form,
    ):
        mock_filter_by.return_value.first.return_value = None
        # 3 fixture nella stessa lega: la quota si esaurisce SULLA PRIMA
        # (get_fixture_odds) - le altre 2 non devono essere processate.
        provider = FakeProviderQuotaExceededOnOdds(
            fixtures=[_sample_fixture(1326590), _sample_fixture(1326591), _sample_fixture(1326592)],
            statistics=_sample_statistics(),
        )

        report = download_import_matches(
            seasons=[2026],
            leagues=[135],
            fixture_date="2026-09-01",
            statuses="FT",
            provider=provider,
        )

        self.assertTrue(report["quota_exceeded"])
        # Nessuna fixture completata: la primissima ha gia' fatto scattare
        # l'interruzione immediata (nessun `continue` sulle successive).
        self.assertEqual(report["inserted"], 0)
        self.assertEqual(report["updated"], 0)

    @patch("src.service_ia.pre_processing.download_match_service.BET_BOOKMAKERS", [{"id": 1}])
    @patch("src.service_ia.pre_processing.download_match_service.form_last_5_tot", return_value=None)
    @patch("src.service_ia.pre_processing.download_match_service.repo_snapshot.save_many")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.save_all")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.save")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.filter_by")
    def test_stops_across_multiple_leagues(
        self,
        mock_filter_by,
        _mock_save,
        _mock_save_all,
        _mock_snapshot_save,
        _mock_form,
    ):
        """La quota esaurita durante la lega 135 deve fermare ANCHE il
        tentativo sulla lega successiva (136) - mai una chiamata HTTP in
        piu' dopo che la quota giornaliera e' gia' segnalata esaurita."""
        mock_filter_by.return_value.first.return_value = None
        provider = FakeProviderQuotaExceededOnOdds(
            fixtures=[_sample_fixture(1326590)],
            statistics=_sample_statistics(),
        )

        report = download_import_matches(
            seasons=[2026],
            leagues=[135, 136],
            fixture_date="2026-09-01",
            statuses="FT",
            provider=provider,
        )

        self.assertTrue(report["quota_exceeded"])
        # Una sola lega vista: fixtures_seen conta SOLO la prima lega (135),
        # la seconda (136) non viene nemmeno interrogata.
        self.assertEqual(report["fixtures_seen"], 1)


if __name__ == "__main__":
    unittest.main()
