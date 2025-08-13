from src.service_ia.utility.request_api import base_api_statistics


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


def get_attribute_statistics(statistics):
    """
    Mappa tutte le statistiche nel dizionario
    """
    data_match_stat = {}
    for statistic in statistics:
        stat_type = statistic['type']
        value = statistic['value'] or 0
        if isinstance(value, str) and value.endswith('%'):
            value = float(value.strip('%'))  # Converti percentuali in numeri
        data_match_stat.update({stat_type: value})
    return data_match_stat


def form_last_5_tot(id_fixture, id_team):
    """
    Recupera la forma fisica della squadra nelle ultime 5 giornate e totali della lega con forma,
    forma dell'attacco e forma della difesa più predizioni
    """
    # Predict della partita
    fixtures_predict = base_api_statistics(path='predictions', params={'fixture': id_fixture})

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
