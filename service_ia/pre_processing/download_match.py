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

from repository.match_repository import MatchRepository
from service_ia.model.match import Match
from service_ia.utility.request_api import base_api_statistics

logging.basicConfig(level=logging.DEBUG)

LEAGUES = [135, 136, 140, 78, 39, 94, 203, 2, 3, 848, 492]
SEASONS = [2025]

with open('../json/bet.json', 'r', encoding='utf-8') as file:
    BET_BOOKMAKERS = json.load(file)

# ( - 0 -> Oggi , - 1 -> Ieri , + 1 -> Domani , + 2 DopoDomani)

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
    return {
        'id_fixture': search_value(fixture, 'id_fixture'),
        'name_home': search_value(fixture, 'team_name'),
        'id_team_home': search_value(fixture, 'team_id'),
        'name_away': search_value(fixture, 'team_name'),
        'id_team_away': search_value(fixture, 'team_id'),
        'date_match': search_value(fixture, 'date_fixture'),
        'current_league': search_value(fixture, 'current_league'),
        'league_match': search_value(fixture, 'league_match'),
        'referee': search_value(fixture, 'referee'),
        'round': search_value(fixture, 'round_fixture'),
        'season': search_value(fixture, 'season'),
    }


try:
    for season in SEASONS:
        logging.info(f'<<< Start season {season} >>>')

        for league in LEAGUES:
            logging.info(f'<<< Start season {season} for league {league} >>>')

            fixtures = base_api_statistics(
                path='fixtures',
                params={'from': from_date, 'to': to_date, 'status': status_list, 'league': league,
                        # 'date': date
                        })
            for fixture in fixtures:
                id_fix = fixture['id_fixture']
                dict_match = map_base_match()

                statistics = get_statistics()
                if len(statistics) > 0:
                    logging.info(f'Statistics match {id_fix} : {statistics}')
                    for statistic in statistics:
                        pass

                fixture_bookmakers = base_api_statistics(path='/odds', params={'fixture': id_fix})
                if len(fixture_bookmakers) > 0:
                    # Inizia a creare il dizionario prima di aggiungere le quote
                    bookmakers_filters = [bookmaker for bookmaker in fixture_bookmakers[0]['bookmakers']]

                if len(dict_match) > 0:
                    match = Match(**dict_match)

                    list_dict_match.append(match)


except Exception as e:
    logging.error('Errore durante il download : ', str(e))
finally:
    try:
        # Salva tutto
        repo_match.insert_massive(list_dict_match)
    except Exception as e_db:
        logging.info('Errore durante il salvataggio a db.Salvato in un file temporaneo :', str(e_db))
        # TODO salvare il dizionario come json da poter riprocessare manualmente


def re_processor_error():
    pass


def added_odds():
    dataset_odds = pd.read_csv(name_odds_base)

    ids_bookmakers = [ids_book['id'] for ids_book in BOOKMAKERS_SPORTS]
    ids_bets = [id_bet['id'] for id_bet in BET_BOOKMAKERS]
    for league in leagues:
        fixtures = base_api_statistics(
            path='fixtures',
            params={
                'from': from_date, 'to': to_date,
                'status': status_list,
                'league': league,
                # 'date': date
            })
        odds_bet = []
        for fixture in fixtures:
            id_fixture = fixture['fixture']['id']
            data_fix = fixture['fixture']['date']
            league_id = fixture['league']['id']
            name_league = fixture['league']['name']
            season = fixture['league']['season']
            round_fixture = fixture['league']['round']
            home_id = fixture['teams']['home']['id']
            home_team = fixture['teams']['home']['name']
            away_id = fixture['teams']['away']['id']
            away_team = fixture['teams']['away']['name']
            fixture_bookmakers = base_api_statistics(path='/odds', params={'fixture': id_fixture})

            if len(fixture_bookmakers) > 0:
                # Inizia a creare il dizionario prima di aggiungere le quote
                bookmakers_filters = [bookmaker for bookmaker in fixture_bookmakers[0]['bookmakers'] if
                                      bookmaker['id'] in ids_bookmakers]

                odd_bet = {
                    'id_fixture_from_stat': id_fixture,
                    'api_from': 'sports-api',
                    'sport_key': league_id,
                    'sport_title': name_league,
                    'commence_time': data_fix,
                    'home_id': home_id,
                    'home_team': home_team,
                    'away_id': away_id,
                    'away_team': away_team,
                    'season': season,
                    'round_fixture': round_fixture,
                }

                for bookmaker in bookmakers_filters:
                    # Crea il dizionario della fixture aggregando tutti gli eventi con le sue quote
                    name_book = bookmaker['name']
                    filter_bet = [bet for bet in bookmaker['bets'] if bet['id'] in ids_bets]
                    for filter_bet_name in filter_bet:
                        for value in filter_bet_name['values']:
                            odd_bet.update({
                                f'{name_book}_{filter_bet_name['name']}_{value['value']}': value['odd']
                            })

                odds_bet.append(odd_bet)

            if len(odds_bet) > 0:
                # Inserisci e modifica il dataset attuale solo se c'è almeno un elemento
                odd_bet_dt = pd.DataFrame(odds_bet)
                concat_odds = pd.concat([dataset_odds, odd_bet_dt], axis=0)
                print(concat_odds)
                # TODO concat_odds.to_csv(name_odds_base, index=False)
