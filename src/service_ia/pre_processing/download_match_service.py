"""
Classe che nasce per l'esigenza di scaricare tutti i prossimi match che ci saranno, con statistiche e quote, e
inserirli a db.
Si punterà a una sola piattaforma, quella di API SPORTS che contiene anche le quote che però durano una settimana.
L'idea sarebbe di attingere quotidianamente a una serie di leghe dell'anno corrente per recuperare tutti i dati
di pre-processing in modo da poter addestrare cin grosse quantità anche in futuro.
"""
import json
import logging
import os
from datetime import datetime, timedelta

import pandas as pd

from src.repository.match_repository import MatchRepository
from src.service_ia.mapper.statistic_mapper import get_attribute_statistics, form_last_5_tot
from src.service_ia.model.match import Match, Statistics, Odds
from src.service_ia.utility.request_api import base_api_statistics

logging.basicConfig(level=logging.DEBUG)

LEAGUES = [135, 136, 140, 78, 39, 94, 203, 2, 3, 848, 492]
SEASONS = [2025]

BASE_DIR = os.path.dirname(__file__)
BET_FILE = os.path.join(BASE_DIR, '..', 'json', 'bet.json')

with open(os.path.abspath(BET_FILE), 'r', encoding='utf-8') as file:
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

repo_match = MatchRepository()


def map_base_match(match, id_fix, fixture, league, season):
    teams_home = fixture['teams']['home']
    teams_away = fixture['teams']['away']

    def get_val(team, value):
        return team[value]

    return {
        'id_match_fk': match.id_match_fk if match else None,
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


def map_statistic(match, stat, team, id_fix, fixture):
    """
    Mappa statistiche, forma fisica, comparison e prediction
    :return: nuovo dizionario di statistiche
    """
    id_team = stat['team']['id']
    stat = stat['statistics']

    attribute_stat = get_attribute_statistics(stat)
    attribute_stat.update(form_last_5_tot(id_fix, id_team))

    def search_key_value(index_key):
        return {key: value for key, value in attribute_stat.items() if pd.notna(value) and index_key in key}

    def default_attribute(attribute):
        return attribute_stat.get(attribute) or 0

    id_stat = None
    id_match = None
    if match and match.statistics:
        id_match = match.id_match_fk
        id_stat = [st for st in match.statistics if st.statistics_team_id == id_team]
        if len(id_stat) > 0:
            id_stat = id_stat[0]

    return {
        'id_statistics_fk': id_stat,
        'id_match': id_match,
        'statistics_team_id': id_team,
        'score_ht': fixture['score']['halftime'][team],
        'score_ft': fixture['score']['fulltime'][team],
        'shots': search_key_value('Total Shots'),
        'fouls': default_attribute('Fouls'),
        'corners': default_attribute('Corner Kicks'),
        'offside': default_attribute('Offsides'),
        'bass_possession': default_attribute('Ball Possession'),
        'yellow_cards': default_attribute('Yellow Cards'),
        'red_cards': default_attribute('Red Cards'),
        'goal_keeper': default_attribute('Goalkeeper Saves'),
        'passes': search_key_value('asses'),
        'form': search_key_value('form_'),
        'for_': search_key_value('for_'),
        'against': search_key_value('against_'),
        'preview_matches': {
            'wins_home': default_attribute('wins_home'),
            'wins_away': default_attribute('wins_away'),
            'draws_home': default_attribute('draws_home'),
            'draws_away': default_attribute('draws_away'),
            'loses_home': default_attribute('loses_home'),
            'loses_away': default_attribute('loses_away')
        },
        'comparison': search_key_value('comparison_'),
        'predict': search_key_value('predict_'),
        'generic_statistics': {
            'expected_goals': default_attribute('expected_goals'),
            'goals_prevented': default_attribute('goals_prevented'),
            'Assists': default_attribute('Assists'),
            'Counter Attacks': default_attribute('Counter Attacks'),
            'Cross Attacks': default_attribute('Cross Attacks'),
            'Free Kicks': default_attribute('Free Kicks'),
            'Goals': default_attribute('Goals'),
            'Goal Attempts': default_attribute('Goal Attempts'),
            'Substitutions': default_attribute('Substitutions'),
            'Throwins': default_attribute('Throwins'),
            'Medical Treatment': default_attribute('Medical Treatment')
        }
    }


def map_odds(match, id_fix):
    """
    Mappa le quote dei bookmakers
    :return: nuovo dizionario di quote
    """

    def switch_bet(bet, alternate_bet):
        match bet:
            case 'Match Winner':
                return 'h2h'
            case 'Goals Over/Under':
                if alternate_bet in ('Over 1.5', 'Under 1.5'):
                    return 'under_over_1_5'
                elif alternate_bet in ('Over 2.5', 'Under 2.5'):
                    return 'under_over_2_5'
                elif alternate_bet in ('Over 3.5', 'Under 3.5'):
                    return 'under_over_3_5'
                elif alternate_bet in ('Over 4.5', 'Under 4.5'):
                    return 'under_over_4_5'
            case 'Both Teams Score':
                return 'goal_no_goal'
            case 'Corners Over Under':
                return 'corners'
            case 'Cards Over/Under':
                return 'cards'
            case 'Double Chance':
                return 'dc'
        return None

    fixture_bookmakers = base_api_statistics(path='odds', params={'fixture': id_fix})
    odd_bet = {
        'id_odds_fk': match.odds[0].id_odds_fk if match and len(match.odds) > 0 else None,
        'id_match': match.id_match_fk if match else None,
        'odds_from': 'sports-api'
    }
    if len(fixture_bookmakers) > 0:
        # Inizia a creare il dizionario prima di aggiungere le quote
        bookmakers_filters = [bookmaker for bookmaker in fixture_bookmakers[0]['bookmakers']]

        for bookmaker in bookmakers_filters:
            # Crea il dizionario della fixture aggregando tutti gli eventi con le sue quote
            name_book = bookmaker['name']
            filter_bet = [bet for bet in bookmaker['bets'] if
                          bet['id'] in [id_bet['id'] for id_bet in BET_BOOKMAKERS]]  # SOLO QUOTE DI EVENTI AMMESSI
            for filter_bet_name in filter_bet:
                for value in filter_bet_name['values']:
                    alternate_value = str(value['value']).lower()
                    if alternate_value in ['yes', 'no']:
                        alternate_value = 'goal_' if alternate_value == 'Yes' else 'no_goal_'
                    elif alternate_value in ['home/draw', 'home/away', 'Draw/away']:
                        alternate_value = '1X' if alternate_value == 'home/draw' else '12' if alternate_value == 'home/away' else 'X2'

                    name_bet = switch_bet(bet=filter_bet_name['name'], alternate_bet=value['value'])
                    if name_bet:
                        head_title = f'{alternate_value}_{name_book}'
                        context = {head_title: value['odd']}
                        if context and head_title and value['odd']:
                            odd_bet.get(name_bet).update(context) \
                                if odd_bet.get(name_bet) else odd_bet.update({name_bet: context})

    return odd_bet


def download_import_matches(seasons=None, leagues=None):
    seasons = SEASONS if seasons is None else seasons
    leagues = LEAGUES if leagues is None else leagues

    list_matches = []
    list_dict_matches = []
    try:
        for season in seasons:
            logging.info(f'<<< Start season {season} >>>')

            for league in leagues:
                logging.info(f'<<< Start season {season} for league {league} >>>')

                fixtures = base_api_statistics(
                    path='fixtures',
                    params={'from': from_date, 'to': to_date, 'status': status_list, 'league': league,
                            # 'date': date
                            })
                for fixture in fixtures:
                    id_fix = fixture['fixture']['id']

                    # Cerco in db se esiste già match
                    match = repo_match.filter_by(dict_search={'id_fixture': id_fix})

                    # Mappa la base del match
                    dict_match = map_base_match(match=match, id_fix=id_fix, fixture=fixture, league=league,
                                                season=season)

                    def get_statistics():
                        # Recupero le statistiche dal nodo fixture principale
                        stat = fixture.get('statistics') or []
                        # E se non ci sono chiama l'alternativa
                        if len(stat) == 0:
                            stat = base_api_statistics(path='fixtures/statistics', params={'fixture': id_fix})
                        return stat

                    # Mappa una serie di statistiche
                    statistics = get_statistics()
                    stats_objs = []
                    if len(statistics) > 0:
                        logging.info(f'Statistics match {id_fix} : {statistics}')
                        stats_objs = [
                            Statistics(
                                **map_statistic(
                                    match,
                                    statistic,
                                    'home' if statistic['team']['id'] == fixture['teams']['home']['id'] else 'away',
                                    id_fix,
                                    fixture
                                )
                            )
                            for statistic in statistics
                        ]

                    # Mappa le quote
                    odds_objs = [Odds(**map_odds(match, id_fix))]

                    # AGGIUNGO A LIVELLO DI ORM LE CLASSI
                    dict_match['statistics'] = stats_objs
                    dict_match['odds'] = odds_objs

                    # LE GESTISCO COME DIZIONARI
                    dict_match_json = dict(dict_match)
                    dict_match_json['statistics'] = [s.to_dict() for s in stats_objs]
                    dict_match_json['odds'] = [o.to_dict() for o in odds_objs]

                    if match is None:
                        # LE AGGIUNGO PER PERSISTERE DOPO
                        match = Match(**dict_match)
                        list_matches.append(match)
                        list_dict_matches.append(dict_match_json)
                    else:
                        # Aggiorno man mano se esiste già quel match
                        repo_match.save(match)
    except Exception as e:
        logging.error('Errore durante il download : ', str(e))
    finally:
        try:
            # Salva tutto in maniera massiva SE NON ESISTE
            repo_match.save_all(list_matches)
        except Exception as err:
            logging.error('Errore nel salvataggio a db. File temporaneo salvato')
            # Salvataggio in un file JSON
            if len(list_dict_matches) > 0:
                with open("error_save_dict.json", "w", encoding="utf-8") as f:
                    json.dump(list_dict_matches, f, ensure_ascii=False, indent=4)

    # TODO insert va bene nel primo inserimento ma questo sarà in continuo aggiornamento -> Testare con dati reali
    # TODO media statistiche


def re_processor_error():
    # Lettura da file JSON
    with open("error_save_dict.json", "r", encoding="utf-8") as f:
        dict_error = json.load(f)

    for element in dict_error:
        stat = element['statistics']
        odd = element['odds']
        element.pop('statistics')
        element.pop('odds')
        match = Match(**dict(element))
        match.statistics = [Statistics(**dict(s)) for s in stat]
        match.odds = [Odds(**dict(odd[0]))]
        repo_match.save(match)


# download_import_matches()
# re_processor_error()
