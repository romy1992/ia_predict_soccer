import logging
import os

import pandas as pd

from src.service_ia.mapper.statistic_mapper import get_attribute_statistics, form_last_5_tot, get_predict
from src.service_ia.utility.request_api import base_api_statistics

# 135, 136, 140,61, 78, 39, 94, 203,492
LEAGUES = [61]
SEASONS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2022, 2023, 2024]
INDEX = 0
JSON_DICT = []
name_history = '../dataset/statistics/dataset_statistics_history_temp.csv'  # TODO temp


def generate_statistics_dataset():
    """
    Genera e accumula tutte le statistiche delle leghe e squadre nel corso degli anni
    """

    def save_dataset():
        if len(JSON_DICT) > 0:
            data_all = pd.concat([pd.read_csv(name_history, low_memory=False), pd.DataFrame(JSON_DICT)],
                                 ignore_index=True) if os.path.exists(name_history) else pd.DataFrame(JSON_DICT)
            data_all.to_csv(name_history, index=False)

    global JSON_DICT
    try:
        # CONSEGUENZA DELL'AGGIORNAMENTO 15/07 FATTO SOTTO : LEGGERò IL DF PER CAPIRE SE ESISTONO LE PARTITE
        df = pd.read_csv(name_history, low_memory=False).to_dict(orient='records')

        for season in SEASONS:
            logging.info(f'<<< Start season {season} >>>')

            for league in LEAGUES:
                logging.info(f'<<< Start season {season} for league {league} >>>')

                # Recupero tutte le squadre del campionato
                teams = base_api_statistics(path='teams', params={'league': league, 'season': season})

                # Prendo solo gli id delle squadre
                id_teams = [team['team']['id'] for team in teams]

                for id_team in id_teams:
                    logging.info(f'<<< Start id team {id_team} >>>')

                    # Chiamo la API delle fixture per farmi restituire tutte le partite disputate finora dalla squadra
                    fixtures = base_api_statistics(path='fixtures',
                                                   params={'season': season, 'team': id_team, 'status': 'FT-AET-PEN'})

                    for fixture in fixtures:
                        logging.info(f'<<< Start Fixture {fixture} >>>')

                        def get_attribute_fixture():

                            def get_team_name_id():
                                teams_home = fixture['teams']['home']
                                teams_away = fixture['teams']['away']

                                if teams_home['id'] == id_team:
                                    team_name = teams_home['name']
                                    home_away = 1
                                    return team_name, home_away, teams_away['id'], teams_away['name']
                                else:
                                    team_name = teams_away['name']
                                    home_away = 0
                                    return team_name, home_away, teams_home['id'], teams_home['name']

                            score_fixture_ht = fixture['score']['halftime']
                            score_fixture_ft = fixture['score']['fulltime']

                            team_name, home_away, opponent_id, opponent_name = get_team_name_id()

                            return {
                                'id_fixture': fixture['fixture']['id'],
                                'season': season,
                                'current_league': league,
                                'league_match': fixture['league']['id'],
                                'date_fixture': fixture['fixture']['date'],
                                'round_fixture': fixture['league']['round'],
                                'referee': fixture['fixture']['referee'],
                                'team_id': id_team,
                                'team_name': team_name,
                                'home_away': home_away,
                                'opponent_id': opponent_id,
                                'opponent_name': opponent_name,
                                'score_home_ht': score_fixture_ht['home'],
                                'score_away_ht': score_fixture_ht['away'],
                                'score_home_ft': score_fixture_ft['home'],
                                'score_away_ft': score_fixture_ft['away']
                            }

                        # Recupera i parametri della partita
                        attribute_fixture = get_attribute_fixture()
                        id_fix = attribute_fixture['id_fixture']

                        # AGGIORNAMENTO 15/07 : AGGIORNARE IL DF DI STATISTICS PRENDENDO LE ULTIME PARTITE
                        # DA MARZO 2025 AD OGGI (CONTROLLERò SE NEL DF HISTORY ESISTE GIà L'ID_FIXTURE)
                        is_duplicate = [id_ for id_ in df if id_['id_fixture'] == id_fix
                                        and id_['team_id'] == attribute_fixture['team_id']
                                        and id_['opponent_id'] == attribute_fixture['opponent_id']]

                        # TODO : togliere questo match_id_from_odds che viene aggiunto al dizionario se non serve
                        match_id_from_odds = is_duplicate[0]['match_id_from_odds'] if len(is_duplicate) > 0 else None
                        attribute_fixture.update({'match_id_from_odds': match_id_from_odds})

                        if len(is_duplicate) == 0:
                            # Chiamo la API delle statistiche di quella partita e di quella squadra
                            # Recupero le statistiche dal nodo fixture principale
                            statistics = fixture.get('statistics') or []
                            # E se non ci sono chiama l'alternativa
                            if len(statistics) == 0:
                                statistics = base_api_statistics(path='fixtures/statistics',
                                                                 params={'fixture': id_fix, 'team': id_team})

                            if len(statistics) > 0:
                                logging.info(f'Statistics match {id_fix} : {statistics}')

                                statistics = statistics[0]['statistics'] \
                                    if statistics[0]['team']['id'] == id_team else statistics[1]

                                attribute_fixture.update(get_attribute_statistics(statistics))

                                # Aggiungo la forma fisica
                                attribute_fixture.update(form_last_5_tot(id_fix, id_team))

                                # Aggiungo al dizionario
                                JSON_DICT.append(attribute_fixture)

                save_dataset()
                JSON_DICT = []
    except Exception as e:
        logging.error(str(e))
    finally:
        save_dataset()


def reload_statistics():
    """
    Questo metodo nasce con l'esigenza di dover aggiungere qualcosa al dataset esistente senza dover riprocessare tutto
    ma passando solo dagli id
    :return: new update dataset statistics
    """
    dataset = pd.read_csv(name_history, low_memory=False)
    dataset_dict = dataset.to_dict(orient='records')
    ids_fixture = set(
        [id_fix['id_fixture'] for id_fix in dataset_dict
         if id_fix['season'] <= 2015
         # and pd.isna(id_fix['match_id_from_odds'])
         and pd.isna(id_fix['predict_win_or_draw'])
         ])
    len_id = len(ids_fixture)
    try:
        for index, id_fix in enumerate(ids_fixture):
            logging.info(f'Process id {id_fix} -> Row {index}/{len_id}')

            # Predict della partita
            fixtures_predict = base_api_statistics(path='/predictions', params={'fixture': id_fix})

            if len(fixtures_predict) > 0:
                predictions = get_predict(fixtures_predict[0])

                # Cerca le righe da modificare
                index_stat_update = [index for index, d in enumerate(dataset_dict) if d['id_fixture'] == id_fix]
                for ind in index_stat_update:
                    dataset.loc[ind, predictions.keys()] = predictions.values()

        print('Finish process')
    except Exception as e:
        print(str(e))
    finally:
        # Aggiorno il dataset
        dataset.to_csv(name_history, index=False)
        print('Aggiornato')


# generate_statistics_dataset()
# reload_statistics()

# convert_csv_to_exel(name_history)

dt = pd.read_csv(name_history, low_memory=False).to_dict(orient='records')
print('Totale : ', len(dt))
print(2014, len([d for d in dt if d['season'] == 2014]))
print(2015, len([d for d in dt if d['season'] == 2015]))
print(2016, len([d for d in dt if d['season'] == 2016]))
print(2017, len([d for d in dt if d['season'] == 2017]))
print(2018, len([d for d in dt if d['season'] == 2018]))
print(2019, len([d for d in dt if d['season'] == 2019]))
print(2020, len([d for d in dt if d['season'] == 2020]))
print(2021, len([d for d in dt if d['season'] == 2021]))
print(2022, len([d for d in dt if d['season'] == 2022]))
print(2023, len([d for d in dt if d['season'] == 2023]))
print(2024, len([d for d in dt if d['season'] == 2024]))
