import json
import os
import unittest
from unittest.mock import patch, call, ANY


# ATTENZIONE: patcha dove VIENE USATA, non dove è definita!
# Nel tuo file fai: from src.service_ia.utility.request_api import base_api_statistics
# quindi il target è: src.service_ia.pre_processing.download_match.base_api_statistics
class TestDownloadMatch(unittest.TestCase):

    @patch("src.service_ia.pre_processing.download_match.repo_match.insert_massive")
    @patch("src.service_ia.pre_processing.download_match.base_api_statistics")
    @patch("src.service_ia.pre_processing.statistics_service.base_api_statistics")   # <-- /predictions
    def test_download_import_matches(self, mock_form, mock_api, mock_insert):

        def read_open_file(path):
            with open(path, 'r', encoding='utf-8') as file:
                return json.load(file)

        # side_effect che cambia in base a path/params
        def fake_api(path, params=None):
            if path == "fixtures":
                # Ritorna il match
                return read_open_file('json_test/fixtures_sports_api_test.json')
            elif path == "fixtures/statistics":
                # ritorna la lista di statistiche per home/away
                return read_open_file('json_test/fixtures_statistics_sports_api_test.json')
            elif path == "/odds":
                # Ritorna le quote
                return read_open_file('json_test/odds_sports_api_test.json')
            # elif path == '/predictions':
            #     # Per predizioni
            #     return read_open_file('json_test/predictions_sports_api_test.json')
            return []

        mock_api.side_effect = fake_api
        mock_form.return_value = read_open_file('json_test/predictions_sports_api_test.json')
        mock_insert.return_value = None

        # import tardivo per usare il mock già attivo (facoltativo)
        from src.service_ia.pre_processing.download_match import download_import_matches
        download_import_matches()

        # Verifiche: è stata chiamata? con quale sequenza?
        expected_calls = [
            call(path="fixtures", params=ANY),  # prima prende le fixture
            call(path="fixtures/statistics", params={"fixture": 1}),  # poi stats (perché "statistics" era vuoto)
            call(path="/odds", params={"fixture": 1}),  # poi le quote
        ]
        # La funzione è chiamata più volte nel loop: controlla ordine per la prima fixture
        mock_api.assert_has_calls(expected_calls, any_order=False)
        mock_form.assert_called()  # /predictions usato da form_last_5_tot

        # ha salvato a DB una sola volta?
        mock_insert.assert_called_once()
