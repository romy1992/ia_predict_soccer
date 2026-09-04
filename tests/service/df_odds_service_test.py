import os
import json
import unittest
from unittest.mock import patch

import pandas as pd

from src.service_ia.pre_processing.df_odds_service import aggregate_events_into_dataset


class TestOddsService(unittest.TestCase):
    @patch("src.service_ia.pre_processing.df_odds_service.pd.DataFrame.to_csv")
    @patch("src.service_ia.pre_processing.df_odds_service.pd.read_csv")
    @patch("src.service_ia.pre_processing.df_odds_service.base_api_odds")
    def test_aggregate_events_into_dataset_ok(self, mock_base_api_odds, mock_read_csv, mock_to_csv):
        # NOTA: il dataset reale (odds_dataset.csv) NON deve mai essere letto né scritto da questo test:
        # è un file di produzione che cresce nel tempo (causa di ParserError/out of memory quando l'intera
        # suite viene eseguita) e la funzione lo sovrascrive nel blocco `finally`. Si usa quindi un piccolo
        # DataFrame di controllo al posto della lettura reale e si neutralizza la scrittura su disco.
        test_json_path = os.path.join(os.path.dirname(__file__), 'json_test', 'odds_api_test.json')
        with open(test_json_path, 'r', encoding='utf-8') as file:
            mock_base_api_odds.return_value = json.load(file)

        fake_df = pd.DataFrame([{
            'id': 'evt1',
            'ids_dates': None,
            'sport_key': 'soccer_italy_serie_a',
            'commence_time': '2024-12-14T14:00:00Z',
            'home_team': 'Cagliari',
            'away_team': 'Atalanta BC',
        }])
        mock_read_csv.return_value = fake_df

        aggregate_events_into_dataset()

        mock_read_csv.assert_called_once()
        mock_base_api_odds.assert_called()
        mock_to_csv.assert_called_once()
