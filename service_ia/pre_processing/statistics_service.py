import logging
import os

import pandas as pd

from service_ia.utility.request_api import base_api_statistics

# 135, 136, 140, 78, 39, 94, 203,2,3,848,492,144, 
LEAGUES = [94]
# SEASONS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
# todo  League 2 - 2023 deve completarsi
# todo  League 3 - 2023 deve completarsi
# todo  League 848 - 2023 deve completarsi
# todo  League 144 - 2023 deve completarsi
SEASONS = [2023]
INDEX = 0
JSON_DICT = []
name_history = '../dataset/statistics/dataset_statistics_history.csv'


def adapted_percentage(value):
    """
    Rimuovo le percentuali e li adatto al dataset
    """
    return (float(value.replace('%', '')) / 100) if value else 0


def get_predict(predict):
    """
    Recupera le predizioni del match
    :return: dizionario predizioni
    """
    prediction = predict['predictions']
    winner_predict_id = prediction['winner']['id']
    winner_predict_name = prediction['winner']['name']
    win_or_draw = prediction['win_or_draw']
    under_over = prediction['under_over']
    goal_home = prediction['goals']['home']
    goal_away = prediction['goals']['away']
    advice = prediction['advice']
    percent_home = adapted_percentage(prediction['percent']['home'])
    percent_draw = adapted_percentage(prediction['percent']['draw'])
    percent_away = adapted_percentage(prediction['percent']['away'])

    return {
        'predict_winner_predict_id': winner_predict_id,
        'predict_winner_predict_name': winner_predict_name,
        'predict_win_or_draw': win_or_draw,
        'predict_under_over': under_over,
        'predict_goal_home': goal_home,
        'predict_goal_away': goal_away,
        'predict_advice': advice,
        'predict_percent_home': percent_home,
        'predict_percent_draw': percent_draw,
        'predict_percent_away': percent_away
    }


def save():
    path_temp = '../dataset/statistics/dataset_statistics_history_temp.csv'  # TODO : TEMPORANEO perché il 06/08/2025 ho dovuto rielaborare tutto
    if len(JSON_DICT) > 0:
        data_all = pd.concat([pd.read_csv(path_temp, low_memory=False), pd.DataFrame(JSON_DICT)],
                             ignore_index=True) if os.path.exists(path_temp) else pd.DataFrame(JSON_DICT)
        data_all.to_csv(path_temp, index=False)


def generate_statistics():
    """
    Genera e accumula tutte le statistiche delle leghe e squadre nel corso degli anni
    """
    global JSON_DICT
    try:
        # CONSEGUENZA DELL'AGGIORNAMENTO 15/07 FATTO SOTTO : LEGGERò IL DF PER CAPIRE SE ESISTONO LE PARTITE
        df = pd.read_csv(name_history, low_memory=False).to_dict(orient='records')

        # TODO DI SUPPORTO
        ids_fix_excluded = pd.read_csv('../dataset/statistics/dataset_statistics_history_temp.csv',
                                       low_memory=False).to_dict(orient='records')

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

                        # TODO PER ELIMINARE EVENTUALI FIXTURE GIA' INSERITE NEL FILE TEMP
                        is_duplicate_temp = [id_ for id_ in ids_fix_excluded if id_['id_fixture'] == id_fix
                                             and id_['team_id'] == attribute_fixture['team_id']
                                             and id_['opponent_id'] == attribute_fixture['opponent_id']]

                        # AGGIORNAMENTO 15/07 : AGGIORNARE IL DF DI STATISTICS PRENDENDO LE ULTIME PARTITE
                        # DA MARZO 2025 AD OGGI (CONTROLLERò SE NEL DF HISTORY ESISTE GIà L'ID_FIXTURE)
                        is_duplicate = [id_ for id_ in df if id_['id_fixture'] == id_fix
                                        and id_['team_id'] == attribute_fixture['team_id']
                                        and id_['opponent_id'] == attribute_fixture['opponent_id']]

                        # TODO : togliere questo match_id_from_odds che viene aggiunto al dizionario se non serve
                        match_id_from_odds = is_duplicate[0]['match_id_from_odds'] if len(is_duplicate) > 0 else None
                        attribute_fixture.update({'match_id_from_odds': match_id_from_odds})

                        # TODO is_duplicate_temp dovrà poi essere sostituito da is_duplicate che a sua volta dovrà puntare al nuovo dataset aggiornato(ora è TEMP)
                        if len(is_duplicate_temp) == 0:
                            # Chiamo la API delle statistiche di quella partita e di quella squadra
                            # Recupero le statistiche dal nodo fixture principale
                            statistics = fixture.get('statistics') or []
                            # E se non ci sono chiama l'alternativa
                            if len(statistics) == 0:
                                statistics = base_api_statistics(path='fixtures/statistics',
                                                                 params={'fixture': id_fix, 'team': id_team})

                            if len(statistics) > 0:
                                logging.info(f'Statistics match {id_fix} : {statistics}')

                                def get_attribute_statistics():
                                    data_match_stat = {}
                                    for statistic in statistics:
                                        stat_type = statistic['type']
                                        value = statistic['value'] or 0
                                        if isinstance(value, str) and value.endswith('%'):
                                            value = float(value.strip('%'))  # Converti percentuali in numeri
                                        data_match_stat.update({stat_type: value})
                                    return data_match_stat

                                statistics = statistics[0]['statistics'] \
                                    if statistics[0]['team']['id'] == id_team else statistics[1]

                                attribute_fixture.update(get_attribute_statistics())

                                def form_last_5_tot(id_fixture):
                                    """
                                    Recupera la forma fisica della squadra nelle ultime 5 giornate e totali della lega con forma,
                                    forma dell'attacco e forma della difesa
                                    """
                                    # Predict della partita
                                    fixtures_predict = base_api_statistics(path='/predictions',
                                                                           params={'fixture': id_fixture})

                                    if len(fixtures_predict) == 0:
                                        return None

                                    predict = fixtures_predict[0]

                                    # Seleziono il nodo da recuperare
                                    selected_team = 'home' if predict['teams']['home']['id'] == id_team else 'away'

                                    # Ultime 5
                                    teams_form = predict['teams'][selected_team]['last_5']
                                    form = adapted_percentage(teams_form['form'])
                                    form_att = adapted_percentage(teams_form['att'])
                                    form_def = adapted_percentage(teams_form['def'])
                                    goal_for = float(teams_form['goals']['for']['average'])
                                    goal_against = float(teams_form['goals']['against']['average'])

                                    def get_goal_predict(team, prefix):
                                        """Totale statistiche campionato"""
                                        by_team = team[prefix]

                                        # Goal Fatti
                                        total_goals_home = by_team['total']['home']
                                        total_goals_away = by_team['total']['away']
                                        total_goals = by_team['total']['total']

                                        # Media goal fatti
                                        total_average_goals_home = float(by_team['average']['home'])
                                        total_average_goals_away = float(by_team['average']['away'])
                                        total_average_goals = float(by_team['average']['total'])

                                        predict_json = {
                                            f'{prefix}_total_goal_home': total_goals_home,
                                            f'{prefix}_total_goal_away': total_goals_away,
                                            f'{prefix}_total_goal': total_goals,
                                            f'{prefix}_total_average_goals_home': total_average_goals_home,
                                            f'{prefix}_total_average_goals_away': total_average_goals_away,
                                            f'{prefix}_total_average_goals': total_average_goals

                                        }

                                        # Minuti segnati goal
                                        def minute_goal_ext(minute):
                                            return {
                                                f'{prefix}_{minute}_goal_total': by_team['minute'][minute]['total'],
                                                f'{prefix}_{minute}_goal_percentage': by_team['minute'][minute][
                                                    'percentage']
                                            }

                                        predict_json.update(minute_goal_ext('0-15'))
                                        predict_json.update(minute_goal_ext('16-30'))
                                        predict_json.update(minute_goal_ext('31-45'))
                                        predict_json.update(minute_goal_ext('46-60'))
                                        predict_json.update(minute_goal_ext('61-75'))
                                        predict_json.update(minute_goal_ext('76-90'))
                                        predict_json.update(minute_goal_ext('91-105'))
                                        predict_json.update(minute_goal_ext('106-120'))

                                        # Under over
                                        def under_over_ext(value):
                                            return {
                                                f'{prefix}_{value}_over': by_team['under_over'][value]['over'],
                                                f'{prefix}_{value}_under': by_team['under_over'][value]['under']
                                            }

                                        predict_json.update(under_over_ext('0.5'))
                                        predict_json.update(under_over_ext('1.5'))
                                        predict_json.update(under_over_ext('2.5'))
                                        predict_json.update(under_over_ext('3.5'))
                                        predict_json.update(under_over_ext('4.5'))

                                        return predict_json

                                    # Totale campionato
                                    teams_total = predict['teams'][selected_team]['league']['goals']
                                    for_ = get_goal_predict(teams_total, 'for')
                                    against_ = get_goal_predict(teams_total, 'against')

                                    # Fixtures: storico partite disputate con vittoria, pareggio o sconfitta
                                    fix = predict['teams'][selected_team]['league']['fixtures']
                                    wins_home = fix['wins']['home']
                                    wins_away = fix['wins']['away']
                                    draws_home = fix['draws']['home']
                                    draws_away = fix['draws']['away']
                                    loses_home = fix['loses']['home']
                                    loses_away = fix['loses']['away']

                                    # Clean Sheet
                                    clean_sheet = predict['teams'][selected_team]['league']['clean_sheet']
                                    clean_sheet_home = clean_sheet['home']
                                    clean_sheet_away = clean_sheet['away']
                                    clean_sheet_total = clean_sheet['total']

                                    # Form Total preso dal nodo Comparison
                                    comparison = predict['comparison']
                                    comparison_form = adapted_percentage(comparison['form'][selected_team])
                                    comparison_att = adapted_percentage(comparison['att'][selected_team])
                                    comparison_def = adapted_percentage(comparison['def'][selected_team])
                                    comparison_poisson_distribution = adapted_percentage(
                                        comparison['poisson_distribution'][selected_team])
                                    comparison_h2h = adapted_percentage(comparison['h2h'][selected_team])
                                    comparison_goals = adapted_percentage(comparison['goals'][selected_team])
                                    comparison_tot = adapted_percentage(comparison['total'][selected_team])

                                    pre_dict = {
                                        'form_last_5': form,
                                        'form_att_last_5': form_att,
                                        'form_def_last_5': form_def,
                                        'form_goal_for_average_last_5': goal_for,
                                        'form_goal_against_average_last_5': goal_against,
                                    }

                                    pre_dict.update(for_)
                                    pre_dict.update(against_)
                                    pre_dict.update({
                                        'wins_home': wins_home,
                                        'wins_away': wins_away,
                                        'draws_home': draws_home,
                                        'draws_away': draws_away,
                                        'loses_home': loses_home,
                                        'loses_away': loses_away,
                                    })
                                    pre_dict.update(
                                        {
                                            'clean_sheet_home': clean_sheet_home,
                                            'clean_sheet_away': clean_sheet_away,
                                            'clean_sheet_total': clean_sheet_total,
                                        }
                                    )
                                    pre_dict.update(
                                        {
                                            'comparison_form': comparison_form,
                                            'comparison_att': comparison_att,
                                            'comparison_def': comparison_def,
                                            'comparison_poisson_distribution': comparison_poisson_distribution,
                                            'comparison_h2h': comparison_h2h,
                                            'comparison_goals': comparison_goals,
                                            'comparison_tot': comparison_tot
                                        }
                                    )

                                    # Predictions match
                                    pre_dict.update(get_predict(predict))

                                    return pre_dict

                                # Aggiungo la forma fisica
                                attribute_fixture.update(form_last_5_tot(id_fix))

                                # Aggiungo al dizionario
                                JSON_DICT.append(attribute_fixture)

                save()
                JSON_DICT = []
    except Exception as e:
        logging.error(str(e))
    finally:
        save()


generate_statistics()


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

# reload_statistics()
# convert_csv_to_exel(name_history)
# d = pd.read_csv('../dataset/statistics/dataset_statistics_history.csv', low_memory=False)
# d.to_excel('../dataset/statistics/dataset_statistics_history_copy.xlsx', index=False)
