import json
import unittest
from unittest.mock import patch, call, ANY

from src.service_ia.model.match import Match, Statistics, Odds
from src.service_ia.pre_processing.download_match_service import calculate_mean
from src.service_ia.utility.utils import convert_dict_match_to_orm


def base_test(mock_form, mock_api):
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
        elif path == "odds":
            # Ritorna le quote
            return read_open_file('json_test/odds_sports_api_test.json')
        # elif path == '/predictions':
        #     # Per predizioni
        #     return read_open_file('json_test/predictions_sports_api_test.json')
        return []

    mock_api.side_effect = fake_api
    mock_form.return_value = read_open_file('json_test/predictions_sports_api_test.json')


# ATTENZIONE: patcha dove VIENE USATA, non dove è definita!
# Nel tuo file fai: from src.service_ia.utility.request_api import base_api_statistics
# quindi il target è: src.service_ia.pre_processing.download_match.base_api_statistics
class TestDownloadMatch(unittest.TestCase):

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)

    # ordine inverso negli argomenti del test: l'ultimo @patch è il primo parametro!
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.save_all")
    @patch("src.service_ia.pre_processing.download_match_service.base_api_statistics")
    @patch("src.service_ia.mapper.statistic_mapper.base_api_statistics")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.filter_by")
    def test_download_import_matches_ok(self, mock_get, mock_form, mock_api, mock_insert):
        base_test(mock_form, mock_api)
        mock_insert.return_value = None
        mock_get.return_value = None

        from src.service_ia.pre_processing.download_match_service import download_import_matches
        download_import_matches(seasons=[2024], leagues=[135])

        # Verifiche: è stata chiamata? con quale sequenza?
        expected_calls = [
            call(path="fixtures", params=ANY),  # prima prende le fixture
            call(path="fixtures/statistics", params={"fixture": 1326590}),  # poi stats (perché "statistics" era vuoto)
            call(path="odds", params={"fixture": 1326590}),  # poi le quote
        ]
        # La funzione è chiamata più volte nel loop: controlla ordine per la prima fixture
        mock_api.assert_has_calls(expected_calls, any_order=False)
        mock_form.assert_called()  # /predictions usato da form_last_5_tot

        # ha salvato a DB una sola volta?
        mock_insert.assert_called_once()

    @patch("src.service_ia.pre_processing.download_match_service.repo_match.save_all")
    @patch("src.service_ia.pre_processing.download_match_service.base_api_statistics")
    @patch("src.service_ia.mapper.statistic_mapper.base_api_statistics")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.filter_by")
    def test_download_import_matches_ko(self, mock_get, mock_form, mock_api, mock_insert):
        base_test(mock_form, mock_api)
        mock_insert.side_effect = Exception("DB KO")
        mock_get.return_value = None

        from src.service_ia.pre_processing.download_match_service import download_import_matches
        download_import_matches(seasons=[2024], leagues=[135])
        mock_insert.assert_called_once()

        # opzionale: verifica che venga usato il fallback su file
        import os
        assert os.path.exists("error_save_dict.json")
        # ripulisci
        os.remove("error_save_dict.json")

    @patch("src.service_ia.pre_processing.download_match_service.repo_match.save_all")
    @patch("src.service_ia.pre_processing.download_match_service.base_api_statistics")
    @patch("src.service_ia.mapper.statistic_mapper.base_api_statistics")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.filter_by")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.save")
    def test_download_import_matches_update(self, mock_save, mock_get, mock_form, mock_api, mock_insert):
        dict_match = {'id_match_fk': "fd7529f4-d623-4c33-b779-8bb8ea0bd45e", 'id_fixture': 1326590,
                      'name_home': 'Inter',
                      'id_team_home': 505, 'name_away': 'Lazio', 'id_team_away': 487,
                      'date_match': '2025-08-11T00:00:00+00:00', 'current_league': 135, 'league_match': 253,
                      'referee': 'Armando Villarrea, USA', 'round': 'Regular Season - 26', 'season': 2024}

        dict_stat = [
            {'id_statistics_fk': "00011ad6-4f15-4c61-b0a7-74710ceb3886",
             'id_match': "fd7529f4-d623-4c33-b779-8bb8ea0bd45e", 'statistics_team_id': 505,
             'score_ht': 1,
             'score_ft': 4, 'shots': {'Total Shots': 4}, 'fouls': 5, 'corners': 3, 'offside': 1,
             'bass_possession': 51.0, 'yellow_cards': 2, 'red_cards': 0, 'goal_keeper': 6,
             'passes': {'Total passes': 489, 'Passes accurate': 428, 'Passes %': 88.0},
             'form': {'form_last_5': 0.0, 'form_att_last_5': 0.32, 'form_def_last_5': 0.21,
                      'form_goal_for_average_last_5': 1.2, 'form_goal_against_average_last_5': 3.0},
             'for_': {'form_goal_for_average_last_5': 1.2, 'for_total_goal_home': 11,
                      'for_total_goal_away': 9,
                      'for_total_goal': 20, 'for_total_average_goals_home': 1.1,
                      'for_total_average_goals_away': 1.0, 'for_total_average_goals': 1.1,
                      'for_0-15_goal_total': 3, 'for_0-15_goal_percentage': '11.54%',
                      'for_16-30_goal_total': 2,
                      'for_16-30_goal_percentage': '7.69%', 'for_31-45_goal_total': 3,
                      'for_31-45_goal_percentage': '11.54%', 'for_46-60_goal_total': 3,
                      'for_46-60_goal_percentage': '11.54%', 'for_61-75_goal_total': 4,
                      'for_61-75_goal_percentage': '15.38%', 'for_76-90_goal_total': 7,
                      'for_76-90_goal_percentage': '26.92%', 'for_106-120_goal_total': 4,
                      'for_106-120_goal_percentage': '15.38%', 'for_0.5_over': 15,
                      'for_0.5_under': 4,
                      'for_1.5_over': 4, 'for_1.5_under': 15, 'for_2.5_over': 1, 'for_2.5_under': 18,
                      'for_3.5_over': 0, 'for_3.5_under': 19, 'for_4.5_over': 0,
                      'for_4.5_under': 19},
             'against': {'form_goal_against_average_last_5': 3.0, 'against_total_goal_home': 12,
                         'against_total_goal_away': 29, 'against_total_goal': 41,
                         'against_total_average_goals_home': 1.2,
                         'against_total_average_goals_away': 3.2,
                         'against_total_average_goals': 2.2, 'against_0-15_goal_total': 3,
                         'against_0-15_goal_percentage': '6.38%', 'against_16-30_goal_total': 4,
                         'against_16-30_goal_percentage': '8.51%', 'against_31-45_goal_total': 3,
                         'against_31-45_goal_percentage': '6.38%', 'against_46-60_goal_total': 11,
                         'against_46-60_goal_percentage': '23.40%', 'against_61-75_goal_total': 6,
                         'against_61-75_goal_percentage': '12.77%', 'against_76-90_goal_total': 12,
                         'against_76-90_goal_percentage': '25.53%', 'against_106-120_goal_total': 8,
                         'against_106-120_goal_percentage': '17.02%', 'against_0.5_over': 16,
                         'against_0.5_under': 3, 'against_1.5_over': 13, 'against_1.5_under': 6,
                         'against_2.5_over': 6, 'against_2.5_under': 13, 'against_3.5_over': 3,
                         'against_3.5_under': 16, 'against_4.5_over': 2, 'against_4.5_under': 17},
             'preview_matches': {'wins_home': 3, 'wins_away': 0, 'draws_home': 2, 'draws_away': 1,
                                 'loses_home': 5, 'loses_away': 8},
             'comparison': {'comparison_form': 0.0, 'comparison_att': 0.33, 'comparison_def': 0.25,
                            'comparison_poisson_distribution': 0.09, 'comparison_h2h': 0.71,
                            'comparison_goals': 0.61, 'comparison_tot': 0.39799999999999996},
             'generic_statistics': {'expected_goals': 0, 'goals_prevented': 0, 'Assists': 0,
                                    'Counter Attacks': 0, 'Cross Attacks': 0, 'Free Kicks': 0,
                                    'Goals': 0,
                                    'Goal Attempts': 0, 'Substitutions': 0, 'Throwins': 0,
                                    'Medical Treatment': 0},
             'predict': {'predict_winner_predict_id': 10281,
                         'predict_winner_predict_name': 'New England II',
                         'predict_win_or_draw': False, 'predict_under_over': '+1.5',
                         'predict_goal_home': '-3.5', 'predict_goal_away': '-1.5',
                         'predict_advice': 'Combo Winner : New England II and +1.5 goals',
                         'predict_percent_home': 0.45, 'predict_percent_draw': 0.45,
                         'predict_percent_away': 0.1}},
            {'id_statistics_fk': "0003325c-304a-4da0-a124-299440576cc5",
             'id_match': "fd7529f4-d623-4c33-b779-8bb8ea0bd45e", 'statistics_team_id': 487,
             'score_ht': 1,
             'score_ft': 1, 'shots': {'Total Shots': 13}, 'fouls': 11, 'corners': 8, 'offside': 1,
             'bass_possession': 49.0, 'yellow_cards': 5, 'red_cards': 0, 'goal_keeper': 0,
             'passes': {'Total passes': 452, 'Passes accurate': 400, 'Passes %': 88.0},
             'form': {'form_last_5': 0.0, 'form_att_last_5': 0.32, 'form_def_last_5': 0.21,
                      'form_goal_for_average_last_5': 1.2, 'form_goal_against_average_last_5': 3.0},
             'for_': {'form_goal_for_average_last_5': 1.2, 'for_total_goal_home': 11,
                      'for_total_goal_away': 9,
                      'for_total_goal': 20, 'for_total_average_goals_home': 1.1,
                      'for_total_average_goals_away': 1.0, 'for_total_average_goals': 1.1,
                      'for_0-15_goal_total': 3, 'for_0-15_goal_percentage': '11.54%',
                      'for_16-30_goal_total': 2,
                      'for_16-30_goal_percentage': '7.69%', 'for_31-45_goal_total': 3,
                      'for_31-45_goal_percentage': '11.54%', 'for_46-60_goal_total': 3,
                      'for_46-60_goal_percentage': '11.54%', 'for_61-75_goal_total': 4,
                      'for_61-75_goal_percentage': '15.38%', 'for_76-90_goal_total': 7,
                      'for_76-90_goal_percentage': '26.92%', 'for_106-120_goal_total': 4,
                      'for_106-120_goal_percentage': '15.38%', 'for_0.5_over': 15,
                      'for_0.5_under': 4,
                      'for_1.5_over': 4, 'for_1.5_under': 15, 'for_2.5_over': 1, 'for_2.5_under': 18,
                      'for_3.5_over': 0, 'for_3.5_under': 19, 'for_4.5_over': 0,
                      'for_4.5_under': 19},
             'against': {'form_goal_against_average_last_5': 3.0, 'against_total_goal_home': 12,
                         'against_total_goal_away': 29, 'against_total_goal': 41,
                         'against_total_average_goals_home': 1.2,
                         'against_total_average_goals_away': 3.2,
                         'against_total_average_goals': 2.2, 'against_0-15_goal_total': 3,
                         'against_0-15_goal_percentage': '6.38%', 'against_16-30_goal_total': 4,
                         'against_16-30_goal_percentage': '8.51%', 'against_31-45_goal_total': 3,
                         'against_31-45_goal_percentage': '6.38%', 'against_46-60_goal_total': 11,
                         'against_46-60_goal_percentage': '23.40%', 'against_61-75_goal_total': 6,
                         'against_61-75_goal_percentage': '12.77%', 'against_76-90_goal_total': 12,
                         'against_76-90_goal_percentage': '25.53%', 'against_106-120_goal_total': 8,
                         'against_106-120_goal_percentage': '17.02%', 'against_0.5_over': 16,
                         'against_0.5_under': 3, 'against_1.5_over': 13, 'against_1.5_under': 6,
                         'against_2.5_over': 6, 'against_2.5_under': 13, 'against_3.5_over': 3,
                         'against_3.5_under': 16, 'against_4.5_over': 2, 'against_4.5_under': 17},
             'preview_matches': {'wins_home': 3, 'wins_away': 0, 'draws_home': 2, 'draws_away': 1,
                                 'loses_home': 5, 'loses_away': 8},
             'comparison': {'comparison_form': 0.0, 'comparison_att': 0.33, 'comparison_def': 0.25,
                            'comparison_poisson_distribution': 0.09, 'comparison_h2h': 0.71,
                            'comparison_goals': 0.61, 'comparison_tot': 0.39799999999999996},
             'generic_statistics': {'expected_goals': 0, 'goals_prevented': 0, 'Assists': 0,
                                    'Counter Attacks': 0, 'Cross Attacks': 0, 'Free Kicks': 0,
                                    'Goals': 0,
                                    'Goal Attempts': 0, 'Substitutions': 0, 'Throwins': 0,
                                    'Medical Treatment': 0},
             'predict': {'predict_winner_predict_id': 10281,
                         'predict_winner_predict_name': 'New England II',
                         'predict_win_or_draw': False, 'predict_under_over': '+1.5',
                         'predict_goal_home': '-3.5', 'predict_goal_away': '-1.5',
                         'predict_advice': 'Combo Winner : New England II and +1.5 goals',
                         'predict_percent_home': 0.45, 'predict_percent_draw': 0.45,
                         'predict_percent_away': 0.1}}]

        dict_odds = [{'id_odds_fk': "0000cb55-39f9-43d4-b613-b61cd7cad14f",
                      'id_match': "fd7529f4-d623-4c33-b779-8bb8ea0bd45e", 'odds_from': 'sports-api',
                      'h2h': {'home_10Bet': '2.88', 'draw_10Bet': '3.10', 'away_10Bet': '2.35',
                              'home_Bet365': '2.80',
                              'draw_Bet365': '3.10', 'away_Bet365': '2.25', 'home_Marathonbet': '2.80',
                              'draw_Marathonbet': '3.10', 'away_Marathonbet': '2.33', 'home_Unibet': '2.63',
                              'draw_Unibet': '3.15', 'away_Unibet': '2.43', 'home_Betfair': '2.88',
                              'draw_Betfair': '3.00', 'away_Betfair': '2.25', 'home_Pinnacle': '2.99',
                              'draw_Pinnacle': '3.17', 'away_Pinnacle': '2.30', 'home_1xBet': '2.86',
                              'draw_1xBet': '3.10', 'away_1xBet': '2.35', 'home_Betano': '3.00',
                              'draw_Betano': '3.15',
                              'away_Betano': '2.27', 'home_Tipico': '2.95', 'draw_Tipico': '2.90',
                              'away_Tipico': '2.30'},
                      'under_over_1_5': {'over 1.5_10Bet': '1.40', 'under 1.5_10Bet': '2.80',
                                         'over 1.5_Marathonbet': '1.34', 'under 1.5_Marathonbet': '2.78',
                                         'over 1.5_Unibet': '1.38', 'under 1.5_Unibet': '2.80',
                                         'over 1.5_Betfair': '1.36', 'under 1.5_Betfair': '2.88',
                                         'over 1.5_Pinnacle': '1.39', 'under 1.5_Pinnacle': '2.81',
                                         'over 1.5_1xBet': '1.34', 'under 1.5_1xBet': '2.78',
                                         'over 1.5_Betano': '1.39',
                                         'under 1.5_Betano': '2.57', 'over 1.5_Tipico': '1.35',
                                         'under 1.5_Tipico': '2.90'},
                      'under_over_2_5': {'over 2.5_10Bet': '2.20', 'under 2.5_10Bet': '1.62',
                                         'over 2.5_Bet365': '2.20',
                                         'under 2.5_Bet365': '1.65', 'over 2.5_Marathonbet': '2.18',
                                         'under 2.5_Marathonbet': '1.60', 'over 2.5_Unibet': '2.18',
                                         'under 2.5_Unibet': '1.58', 'over 2.5_Betfair': '2.25',
                                         'under 2.5_Betfair': '1.57', 'over 2.5_Pinnacle': '2.28',
                                         'under 2.5_Pinnacle': '1.60', 'over 2.5_1xBet': '2.18',
                                         'under 2.5_1xBet': '1.60', 'over 2.5_Betano': '2.12',
                                         'under 2.5_Betano': '1.57', 'over 2.5_Tipico': '2.15',
                                         'under 2.5_Tipico': '1.62'},
                      'under_over_3_5': {'over 3.5_10Bet': '4.20', 'under 3.5_10Bet': '1.20',
                                         'over 3.5_Marathonbet': '4.05', 'under 3.5_Marathonbet': '1.17',
                                         'over 3.5_Unibet': '3.95', 'under 3.5_Unibet': '1.20',
                                         'over 3.5_Betfair': '4.50', 'under 3.5_Betfair': '1.17',
                                         'over 3.5_1xBet': '4.05', 'under 3.5_1xBet': '1.17',
                                         'over 3.5_Betano': '3.55',
                                         'under 3.5_Betano': '1.21', 'over 3.5_Tipico': '3.90',
                                         'under 3.5_Tipico': '1.20'},
                      'under_over_4_5': {'over 4.5_Unibet': '7.50', 'under 4.5_Unibet': '1.05',
                                         'over 4.5_Betfair': '9.00', 'under 4.5_Betfair': '1.03',
                                         'over 4.5_1xBet': '9.00', 'under 4.5_1xBet': '1.03'},
                      'under_over_home_away': None,
                      'goal_no_goal': {'no_goal__10Bet': '1.80', 'no_goal__Marathonbet': '1.80',
                                       'no_goal__Unibet': '1.77', 'no_goal__Betfair': '1.80',
                                       'no_goal__Pinnacle': '1.81', 'no_goal__1xBet': '1.83',
                                       'no_goal__Betano': '1.75',
                                       'no_goal__Tipico': '1.80'},
                      'corners': {'over 8_Bet365': '1.92', 'under 8_Bet365': '1.88', 'over 7.5_Bet365': '1.67',
                                  'under 7.5_Bet365': '2.10', 'over 8_1xBet': '1.92', 'under 8_1xBet': '1.88',
                                  'over 8.5_1xBet': '2.12', 'under 8.5_1xBet': '1.64', 'over 9_1xBet': '2.54',
                                  'under 9_1xBet': '1.43', 'over 9.5_1xBet': '2.86', 'under 9.5_1xBet': '1.34',
                                  'over 10_1xBet': '3.76', 'under 10_1xBet': '1.18', 'over 10.5_1xBet': '4.15',
                                  'under 10.5_1xBet': '1.15', 'over 11_1xBet': '6.05', 'under 11_1xBet': '1.04',
                                  'over 7_1xBet': '1.44', 'under 7_1xBet': '2.52', 'over 7.5_1xBet': '1.67',
                                  'under 7.5_1xBet': '2.08', 'over 5.5_1xBet': '1.12', 'under 5.5_1xBet': '4.50',
                                  'over 11.5_1xBet': '6.52', 'under 11.5_1xBet': '1.03', 'over 6_1xBet': '1.16',
                                  'under 6_1xBet': '4.00', 'over 6.5_1xBet': '1.34', 'under 6.5_1xBet': '2.86',
                                  'over 7.5_Betano': '1.70', 'under 7.5_Betano': '2.07'}, 'cards': None,
                      'dc': {'1X_10Bet': '1.48', '12_10Bet': '1.28', 'draw/away_10Bet': '1.38',
                             '1X_Bet365': '1.53',
                             '12_Bet365': '1.28', 'draw/away_Bet365': '1.33', '1X_Marathonbet': '1.48',
                             '12_Marathonbet': '1.28', 'draw/away_Marathonbet': '1.33', '1X_Unibet': '1.52',
                             '12_Unibet': '1.33', 'draw/away_Unibet': '1.46', '1X_Pinnacle': '1.60',
                             '12_Pinnacle': '1.34', 'draw/away_Pinnacle': '1.37', '1X_1xBet': '1.48',
                             '12_1xBet': '1.28',
                             'draw/away_1xBet': '1.33', '1X_Betano': '1.60', '12_Betano': '1.31',
                             'draw/away_Betano': '1.37', '1X_Tipico': '1.55', '12_Tipico': '1.20',
                             'draw/away_Tipico': '1.35'}}]

        match = Match(**dict_match)
        stat_ = Statistics(**dict_stat[0])
        stat__ = Statistics(**dict_stat[1])
        odds = Odds(**dict_odds[0])
        match.statistics = [stat_, stat__]
        match.odds = [odds]
        match = match

        base_test(mock_form, mock_api)
        mock_get.return_value = match
        mock_save.return_value = None
        mock_insert.return_value = None

        from src.service_ia.pre_processing.download_match_service import download_import_matches
        download_import_matches(seasons=[2024], leagues=[135])

        # Verifiche: è stata chiamata? con quale sequenza?
        expected_calls = [
            call(path="fixtures", params=ANY),  # prima prende le fixture
            call(path="fixtures/statistics", params={"fixture": 1326590}),  # poi stats (perché "statistics" era vuoto)
            call(path="odds", params={"fixture": 1326590}),  # poi le quote
        ]
        # La funzione è chiamata più volte nel loop: controlla ordine per la prima fixture
        mock_api.assert_has_calls(expected_calls, any_order=False)
        mock_form.assert_called()  # /predictions usato da form_last_5_tot
        mock_save.assert_called()

    @patch("src.service_ia.pre_processing.download_match_service.repo_match.search_filter")
    @patch("src.service_ia.pre_processing.download_match_service.repo_match.search_by_id_fixture_None_and_season")
    def test_calculate_mean(self, mock_search_filter, mock_update_bulk):
        with open('json_test/all_match_mean.json', 'r', encoding='utf-8') as file:
            dict_json = json.load(file)
            matches_all = convert_dict_match_to_orm(dict_json)

        mock_search_filter.return_value = matches_all
        mock_update_bulk.return_value = None
        calculate_mean(with_season=2024, force_mean=True, teams=[487])
        mock_search_filter.assert_called()
