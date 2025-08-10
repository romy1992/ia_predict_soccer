"""
Classe che nasce per l'esigenza di scaricare tutti i prossimi match che ci saranno, con statistiche e quote, e
inserirli a db.
Si punterà a una sola piattaforma, quella di API SPORTS che contiene anche le quote che però durano una settimana.
L'idea sarebbe di attingere quotidianamente a una serie di leghe dell'anno corrente per recuperare tutti i dati
di pre-processing in modo da poter addestrare cin grosse quantità anche in futuro.
"""
import json
import logging
from datetime import datetime, timedelta

import pandas as pd

from repository.match_repository import MatchRepository
from service_ia.model.match import Match, Statistics, Odds
from service_ia.pre_processing.statistics_service import form_last_5_tot, get_attribute_statistics
from service_ia.utility.request_api import base_api_statistics

logging.basicConfig(level=logging.DEBUG)

LEAGUES = [135, 136, 140, 78, 39, 94, 203, 2, 3, 848, 492]
SEASONS = [2025]

with open('../json/bet.json', 'r', encoding='utf-8') as file:
    BET_BOOKMAKERS = json.load(file)

# ( 0 -> Oggi , - 1 -> Ieri, - 2 -> altro ieri , + 1 -> Domani , + 2 DopoDomani)

# Formato richiesto è esempio:"2025-02-12"
format_data = '%Y-%m-%d'
current_data = datetime.now()
# Scegliere da che giorno indietro si vuole andare per recuperare le partite
from_date = (current_data - timedelta(days=3)).strftime(format_data)
# Fino a ...
to_date = (current_data - timedelta(days=0)).strftime(format_data)

# Scegliere i giorni indietro che si vuole andare per recuperare le partite (ESEGUIRA' solo un giorno)
date = (current_data - timedelta(days=0)).strftime(format_data)
date_manual = '2025-03-07'

# FT è partita finita
# AET è per partita finita ai supplementari (QUINDI PER COPPE)
# PEN è per partita finita ai rigori (QUINDI PER COPPE)
status_list = ['FT', 'AET', 'PEN']

# SOLO QUOTE DI EVENTI AMMESSI
ids_bets = [id_bet['id'] for id_bet in BET_BOOKMAKERS]

list_dict_match = []
repo_match = MatchRepository()


def get_statistics():
    # Recupero le statistiche dal nodo fixture principale
    stat = fixture.get('statistics') or []
    # E se non ci sono chiama l'alternativa
    if len(stat) == 0:
        stat = base_api_statistics(path='fixtures/statistics', params={'fixture': id_fix})
    return stat


def map_base_match():
    teams_home = fixture['teams']['home']
    teams_away = fixture['teams']['away']

    def get_val(team, value):
        return team[value]

    return {
        'id_fixture': id_fix,
        'name_home': get_val(teams_home, 'name'),
        'id_team_home': get_val(teams_home, 'id'),
        'name_away': get_val(teams_away, 'name'),
        'id_team_away': get_val(teams_away, 'id'),
        'date_match': fixture['fixture']['date'],
        'current_league': league,
        'league_match': fixture['league']['id'],
        'referee': fixture['fixture']['referee'],
        'round': fixture['league']['round'],
        'season': season
    }


def map_statistic(stat, team):
    id_team = stat['team']['id']
    stat = stat['statistics']

    attribute_stat = get_attribute_statistics(stat)
    attribute_stat.update(form_last_5_tot(id_fix, id_team))

    def search_key_value(index_key):
        return {key: value for key, value in attribute_stat.items() if pd.notna(value) and index_key in key}

    return {
        'statistics_team_id': id_team,
        'score_ht': fixture['score']['halftime'][team],
        'score_ft': fixture['score']['fulltime'][team],
        'shots': search_key_value('Shots'),
        'fouls': attribute_stat['Fouls'],
        'corners': attribute_stat['Corner Kicks'],
        'offside': attribute_stat['Offsides'],
        'bass_possession': attribute_stat['Ball Possession'],
        'yellow_cards': attribute_stat['Yellow Cards'],
        'red_cards': attribute_stat['Red Cards'],
        'goal_keeper': attribute_stat['Goalkeeper Saves'],
        'passes': search_key_value('asses'),
        'form': search_key_value('form_'),
        'for_': search_key_value('for_'),
        'against': search_key_value('against_'),
        'preview_matches': {
            'wins_home': attribute_stat['wins_home'],
            'wins_away': attribute_stat['wins_away'],
            'draws_home': attribute_stat['draws_home'],
            'draws_away': attribute_stat['draws_away'],
            'loses_home': attribute_stat['loses_home'],
            'loses_away': attribute_stat['loses_away']
        },
        'comparison': search_key_value('comparison_'),
        'predict': search_key_value('predict_'),
        'generic_statistics': {
            'expected_goals': attribute_stat['expected_goals'],
            'goals_prevented': attribute_stat['goals_prevented'],
            'Assists': attribute_stat['Assists'],
            'Counter Attacks': attribute_stat['Counter Attacks'],
            'Cross Attacks': attribute_stat['Cross Attacks'],
            'Free Kicks': attribute_stat['Free Kicks'],
            'Goals': attribute_stat['Goals'],
            'Goal Attempts': attribute_stat['Goal Attempts'],
            'Substitutions': attribute_stat['Substitutions'],
            'Throwins': attribute_stat['Throwins'],
            'Medical Treatment': attribute_stat['Medical Treatment']
        },
        'mean_season_at_today': {

        }
    }


def map_odds():
    """
    Mappa le quote dei bookmakers
    :return: nuovo dizionario di quote
    """

    def switch_bet(bet, alternate_bet):
        match bet:
            case 'Match Winner':
                return 'h2h'
            case 'Goals Over/Under':
                if alternate_bet == 'Over 1.5' or 'Under 1.5':
                    return 'under_over_1_5'
                elif alternate_bet == 'Over 2.5' or 'Under 2.5':
                    return 'under_over_2_5'
                elif alternate_bet == 'Over 3.5' or 'Under 3.5':
                    return 'under_over_3_5'
                elif alternate_bet == 'Over 4.5' or 'Under 4.5':
                    return 'under_over_4_5'
            case 'Both Teams Score':
                return 'goal_no_goal'
            case 'Corners Over Under':
                return 'corners'
            case 'Cards Over/Under':
                return 'cards'
            case 'Double Chance':
                return 'dc'

    # fixture_bookmakers = base_api_statistics(path='/odds', params={'fixture': id_fix})
    with open("response_test.json", "r", encoding="utf-8") as f:
        fixture_bookmakers = json.load(f)['response']

    odd_bet = {
        'odds_from': 'sports-api'
    }
    if len(fixture_bookmakers) > 0:
        # Inizia a creare il dizionario prima di aggiungere le quote
        bookmakers_filters = [bookmaker for bookmaker in fixture_bookmakers[0]['bookmakers']]

        for bookmaker in bookmakers_filters:
            # Crea il dizionario della fixture aggregando tutti gli eventi con le sue quote
            name_book = bookmaker['name']
            filter_bet = [bet for bet in bookmaker['bets'] if bet['id'] in ids_bets]
            for filter_bet_name in filter_bet:
                for value in filter_bet_name['values']:
                    alternate_value = str(value['value']).lower()
                    if alternate_value in ['yes', 'no']:
                        alternate_value = 'goal_' if alternate_value == 'Yes' else 'no_goal_'
                    elif alternate_value in ['home/draw', 'home/away', 'Draw/away']:
                        alternate_value = '1X' if alternate_value == 'home/draw' else '12' if alternate_value == 'home/away' else 'X2'

                    name_bet = switch_bet(bet=filter_bet_name['name'], alternate_bet=value['value'])
                    if odd_bet.get(name_bet):
                        odd_bet[name_bet].update({f'{alternate_value}_{name_book}': value['odd']})
                    else:
                        odd_bet.update({
                            name_bet: {f'{alternate_value}_{name_book}': value['odd']}
                        })
    return odd_bet


map_odds()


# try:
#     for season in SEASONS:
#         logging.info(f'<<< Start season {season} >>>')
#
#         for league in LEAGUES:
#             logging.info(f'<<< Start season {season} for league {league} >>>')
#
#             fixtures = base_api_statistics(
#                 path='fixtures',
#                 params={'from': from_date, 'to': to_date, 'status': status_list, 'league': league,
#                         # 'date': date
#                         })
#             for fixture in fixtures:
#                 id_fix = fixture['fixture']['id']
#
#                 # Mappa la base del match
#                 dict_match = map_base_match()
#
#                 # Mappa una serie di statistiche
#                 statistics = get_statistics()
#                 if len(statistics) > 0:
#                     logging.info(f'Statistics match {id_fix} : {statistics}')
#                     dict_match.update(
#                         {'statistics': [
#                             Statistics(**map_statistic(statistic, '') for statistic in statistics)]})
#
#                 # Mappa le quote
#                 dict_match.update({'odds': Odds(**map_odds())})
#
#                 if len(dict_match) > 0:
#                     match = Match(**dict_match)
#
#                     list_dict_match.append(match)
#
#
# except Exception as e:
#     logging.error('Errore durante il download : ', str(e))
# finally:
#     try:
#         # Salva tutto
#         repo_match.insert_massive(list_dict_match)
#     except Exception as e_db:
#         logging.info('Errore durante il salvataggio a db.Salvato in un file temporaneo :', str(e_db))
#         # Salvataggio in un file JSON
#         with open("error_save_dict.json", "w", encoding="utf-8") as f:
#             json.dump(list_dict_match, f, ensure_ascii=False, indent=4)


def re_processor_error():  # TODO
    # Lettura da file JSON
    with open("error_save_dict.json", "r", encoding="utf-8") as f:
        dict_error = json.load(f)

# def added_odds():
#     dataset_odds = pd.read_csv(name_odds_base)
#
#     ids_bookmakers = [ids_book['id'] for ids_book in BOOKMAKERS_SPORTS]
#     ids_bets = [id_bet['id'] for id_bet in BET_BOOKMAKERS]
#     for league in leagues:
#         fixtures = base_api_statistics(
#             path='fixtures',
#             params={
#                 'from': from_date, 'to': to_date,
#                 'status': status_list,
#                 'league': league,
#                 # 'date': date
#             })
#         odds_bet = []
#         for fixture in fixtures:
#             id_fixture = fixture['fixture']['id']
#             data_fix = fixture['fixture']['date']
#             league_id = fixture['league']['id']
#             name_league = fixture['league']['name']
#             season = fixture['league']['season']
#             round_fixture = fixture['league']['round']
#             home_id = fixture['teams']['home']['id']
#             home_team = fixture['teams']['home']['name']
#             away_id = fixture['teams']['away']['id']
#             away_team = fixture['teams']['away']['name']
#             fixture_bookmakers = base_api_statistics(path='/odds', params={'fixture': id_fixture})
#
#             if len(fixture_bookmakers) > 0:
#                 # Inizia a creare il dizionario prima di aggiungere le quote
#                 bookmakers_filters = [bookmaker for bookmaker in fixture_bookmakers[0]['bookmakers'] if
#                                       bookmaker['id'] in ids_bookmakers]
#
#                 odd_bet = {
#                     'id_fixture_from_stat': id_fixture,
#                     'api_from': 'sports-api',
#                     'sport_key': league_id,
#                     'sport_title': name_league,
#                     'commence_time': data_fix,
#                     'home_id': home_id,
#                     'home_team': home_team,
#                     'away_id': away_id,
#                     'away_team': away_team,
#                     'season': season,
#                     'round_fixture': round_fixture,
#                 }
#
#                 for bookmaker in bookmakers_filters:
#                     # Crea il dizionario della fixture aggregando tutti gli eventi con le sue quote
#                     name_book = bookmaker['name']
#                     filter_bet = [bet for bet in bookmaker['bets'] if bet['id'] in ids_bets]
#                     for filter_bet_name in filter_bet:
#                         for value in filter_bet_name['values']:
#                             odd_bet.update({
#                                 f'{name_book}_{filter_bet_name['name']}_{value['value']}': value['odd']
#                             })
#
#                 odds_bet.append(odd_bet)
#
#             if len(odds_bet) > 0:
#                 # Inserisci e modifica il dataset attuale solo se c'è almeno un elemento
#                 odd_bet_dt = pd.DataFrame(odds_bet)
#                 concat_odds = pd.concat([dataset_odds, odd_bet_dt], axis=0)
#                 print(concat_odds)
#                 # TODO concat_odds.to_csv(name_odds_base, index=False)
