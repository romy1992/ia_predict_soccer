import json
import unittest
from unittest.mock import patch

from src.service_ia.pre_processing.df_odds_service import aggregate_events_into_dataset


class TestOddsService(unittest.TestCase):
    @patch("src.service_ia.pre_processing.df_odds_service.base_api_odds")
    def test_aggregate_events_into_dataset_ok(self, mock_base_api_odds):
        with open('json_test/odds_api_test.json', 'r', encoding='utf-8') as file:
            mock_base_api_odds.return_value = json.load(file)

        aggregate_events_into_dataset()

        mock_base_api_odds.assert_called()
