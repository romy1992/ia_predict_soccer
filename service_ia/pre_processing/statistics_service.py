import logging
import os
import time

import pandas as pd

from service_ia.utility.request_api import base_api_statistics

LEAGUES = [136]
SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
INDEX = 0


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


def get_attribute_fixture():
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
            f'{prefix}_{minute}_goal_percentage': by_team['minute'][minute]['percentage']
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


def form_last_5_tot(id_fixture):
    """
    Recupera la forma fisica della squadra nelle ultime 5 giornate e totali della lega con forma,
    forma dell'attacco e forma della difesa
    """
    # Predict della partita
    fixtures_predict = base_api_statistics(path='/predictions', params={'fixture': id_fixture})
    check_sleep()

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
    comparison_poisson_distribution = adapted_percentage(comparison['poisson_distribution'][selected_team])
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

    return pre_dict


def adapted_percentage(value):
    """
    Rimuovo le percentuali e li adatto al dataset
    """
    return (float(value.replace('%', '')) / 100) if value else 0


def get_attribute_statistics():
    data_match_stat = {}
    for statistic in statistics:
        stat_type = statistic['type']
        value = statistic['value'] or 0
        if isinstance(value, str) and value.endswith('%'):
            value = float(value.strip('%'))  # Converti percentuali in numeri
        data_match_stat.update({stat_type: value})
    return data_match_stat


def check_sleep():
    global INDEX
    INDEX += 1
    if INDEX == 100:
        logging.info('Sleep process')
        time.sleep(25)
        INDEX = 0


def save():
    if len(json_dict) > 0:
        file_path = '../dataset/dataset_statistics_history.csv'
        data_all = pd.concat([pd.read_csv(file_path), pd.DataFrame(json_dict)],
                             ignore_index=True) if os.path.exists(file_path) else pd.DataFrame(json_dict)
        data_all.to_csv(file_path, index=False)


try:
    # CONSEGUENZA DELL'AGGIORNAMENTO 15/07 FATTO SOTTO : LEGGERò IL DF PER CAPIRE SE ESISTONO LE PARTITE
    df = pd.read_csv('../dataset/dataset_statistics_history.csv').to_dict(orient='records')

    json_dict = []
    for season in SEASONS:
        logging.info(f'<<< Start season {season} >>>')

        for league in LEAGUES:
            logging.info(f'<<< Start season {season} for league {league} >>>')

            # Recupero tutte le squadre del campionato
            teams = base_api_statistics(path='teams', params={'league': league, 'season': season})
            check_sleep()

            # Prendo solo gli id delle squadre
            id_teams = [team['team']['id'] for team in teams]

            for id_team in id_teams:
                logging.info(f'<<< Start id team {id_team} >>>')

                # Chiamo la API delle fixture per farmi restituire tutte le partite disputate finora dalla squadra
                fixtures = base_api_statistics(path='fixtures',
                                               params={'season': season, 'team': id_team, 'status': 'FT-AET-PEN'})
                check_sleep()

                for fixture in fixtures:
                    logging.info(f'<<< Start Fixture {fixture} >>>')

                    # Recupera i parametri della partita
                    attribute_fixture = get_attribute_fixture()
                    id_fix = attribute_fixture['id_fixture']

                    # AGGIORNAMENTO 15/07 : AGGIORNARE IL DF DI STATISTICS PRENDENDO LE ULTIME PARTITE
                    # DA MARZO 2025 AD OGGI (CONTROLLERò SE NEL DF HISTORY ESISTE GIà L'ID_FIXTURE)
                    if len([id_ for id_ in df if
                            id_['id_fixture'] == id_fix and id_['team_id'] == attribute_fixture['team_id'] and id_[
                                'opponent_id'] == attribute_fixture['opponent_id']]) == 0:
                        # Chiamo la API delle statistiche di quella partita e di quella squadra
                        statistics = base_api_statistics(path='fixtures/statistics', params={'fixture': id_fix})
                        check_sleep()
                        logging.info(f'Statistics match {id_fix} : {statistics}')
                        if len(statistics) > 0:
                            statistics = statistics[0]['statistics']
                            attribute_fixture.update(get_attribute_statistics())

                            # Aggiungo la forma fisica
                            attribute_fixture.update(form_last_5_tot(id_fix))

                            # Aggiungo al dizionario
                            json_dict.append(attribute_fixture)

            save()
            json_dict = []
except Exception as e:
    print(str(e))
    save()
