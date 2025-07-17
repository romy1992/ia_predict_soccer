import json
import os

import pandas as pd
from dateutil.parser import isoparse

from service_ia.utility.request_api import base_api_odds

dict_matches = []


def switch_league_number_by_string(value):
    """
    Questo switch serve per passare da un codice lega di "API Sport" a una stringa per "The Odds Sport"
    :param value:
    :return:
    """
    match value:
        case '135' | 135:
            return 'soccer_italy_serie_a'
        case _:
            return None


def switch_league_string_by_number(value):
    """
    Questo switch serve per passare da una stringa per "The Odds Sport" a un codice lega di "API Sport"
    :param value:
    :return:
    """
    match value:
        case 'soccer_italy_serie_a':
            return 135
        case _:
            return None


def aggregate_statistics_id():
    """
    Dalle Statistiche del Dataset, in base al match, chiamo le API di ODDS per inserire nel dataset l'id dell'evento
    con data da 2020-06-06T19:45:00Z (AAAA-MM-DD)
    :return: Nuovo dataset con statistiche e id evento ODDS
    """
    date_event_historical = isoparse('2020-06-06T19:45:00Z')
    # Dataset originale delle statistiche
    dataset_statistics = pd.read_csv('../dataset/dataset_statistics_history.csv')
    dataset_statistics['date_fixture'] = pd.to_datetime(dataset_statistics['date_fixture'])
    dataset_statistics['id_events_odds'] = None
    for row in dataset_statistics.itertuples(index=False):
        date_match = row.date_fixture
        if date_match > date_event_historical:
            date_match = date_match.strftime('%Y-%m-%dT%H:%M:%SZ')
            league_api_odds = switch_league_number_by_string(row.league_match)
            events_odds = base_api_odds(type_api='hist', path=f'{league_api_odds}/events', params={'date': date_match})
            for event_id in events_odds['data']:
                name_team_home = row.team_name if row.home_away == 1 else row.opponent_name
                name_team_away = row.opponent_name if row.home_away == 1 else row.team_name
                if name_team_home == event_id['home_team'] and name_team_away == event_id['away_team']:
                    row['id_events_odds'] = event_id['id']


def aggregate_statistics_odds():
    """
    Aggregherà le statistiche storiche con quotazioni storiche
    :return: Nuovo dataset completo
    """
    # Dataset originale delle statistiche
    dataset_statistics = pd.read_csv('../dataset/dataset_statistics_history.csv')

    # Prendo il nome della squadra di casa, ospite e orario del match (unici valori per fare match con il Portale ODDS)
    for row in dataset_statistics.itertuples(index=False):
        name_team_home = row.team_name if row.home_away == 1 else row.opponent_name
        name_team_away = row.opponent_name if row.home_away == 1 else row.team_name
        date_match = row.date_fixture
        league_api_odds = switch_league_number_by_string(row.league_match)

        # Chiamo API odds per recupero Events
        events_odds = base_api_odds(type_api='hist', path=f'{league_api_odds}/events', params={'date': date_match})
        for event_id in events_odds:
            dict_matches.append({
                'id_events_odds': event_id['id']
            })
    dict_matches_file = pd.DataFrame(dict_matches)
    dict_matches_file.drop_duplicates(inplace=True)
    dict_matches_file.sort_values(by=['league_api_sports', 'date'], inplace=True)
    dict_matches_file.to_csv('../dataset/matches_2015_now_2.csv', index=False)


def reorder_and_adapt_csv():
    # Dataset originale delle statistiche
    dataset_statistics = pd.read_excel('../dataset/dataset_statistics_history.xlsx')

    for row in dataset_statistics.itertuples(index=False):
        name_team_home = row.team_name if row.home_away == 1 else row.opponent_name
        name_team_away = row.opponent_name if row.home_away == 1 else row.team_name
        date_match = row.date_fixture
        league_api_odds = switch_league_number_by_string(row.league_match)

        dict_matches.append({
            'name_home': name_team_home,
            'name_away': name_team_away,
            'date_match': date_match,
            'current_league': row.league_match
        })

    dict_matches_file = pd.DataFrame(dict_matches)
    dict_matches_file.drop_duplicates(inplace=True)
    dict_matches_file.to_csv('../dataset/matches_2015_now.csv', index=False)


def csv_sports_odds():
    sports = base_api_odds(type_api='normal')
    sports = {sport['title']: {'key': sport['key'], 'description': sport['description'],
                               'key_number': switch_league_string_by_number(sport['key'])}
              for sport in sports if sport['group'] == 'Soccer' and sport['active']}
    # Salvataggio su file JSON
    with open('../dataset/sports_odds.json', 'w', encoding='utf-8') as file:
        json.dump(sports, file, ensure_ascii=False, indent=4)





# aggregate_statistics_id()
# reorder_and_adapt_csv()


