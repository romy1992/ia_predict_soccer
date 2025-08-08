"""
Operazione di migrazione da dataset al db.
Nasce dalla necessità di avere tutto organizzato in un db e non nei csv o excel
"""
import logging

import pandas as pd

from repository.match_repository import MatchRepository
from service_ia.model.match import Match, Statistics, Odds

logging.basicConfig(level=logging.DEBUG)


def migrate_dataset():
    statistics = pd.read_csv('../dataset/statistics/dataset_statistics_history.csv', low_memory=False).to_dict(
        orient='records')
    odds = pd.read_csv('../dataset/odds/odds_dataset_copy.csv', low_memory=False).to_dict(
        orient='records')
    repo_match = MatchRepository()

    list_id_fixture = set([statistic['id_fixture'] for statistic in statistics])

    list_id_odds = set([odd['id'] for odd in odds])

    def search_value(stat, field):
        """
        Cerca il valore nel dizionario singolarmente
        :param stat: statistica corrente di home o away
        :param field: nome della key
        :return: singolo valore
        """
        return stat.get(field)

    def search_value_columns(stat, col_value, not_include=None):
        """
        Cerca i valori nel dizionario con somiglianza nel nome colonna
        :param stat: statistica corrente di home o away
        :param col_value: utilizzato per "la somiglianza" della colonna
        :param not_include: se valorizzato, è una lista che serve a escludere i col_value che hanno somiglianze ma che non servono
        :return: {chiave:valore}
        """
        if not_include:
            list_include = {key: value for key, value in stat.items() if col_value in key}
            return {key: value for key, value in list_include.items()
                    if all(inc not in key for inc in not_include)}
        else:
            return {key: value for key, value in stat.items() if col_value in key}

    def single_statistic(stat, team):
        return {
            'statistics_team_id': search_value(stat, 'team_id'),
            'score_ht': search_value(stat, f'score_{team}_ht'),
            'score_ft': search_value(stat, f'score_{team}_ft'),
            'shots': search_value_columns(stat, 'Shots'),
            'fouls': search_value(stat, 'Fouls'),
            'corners': search_value(stat, 'Corner Kicks'),
            'offside': search_value(stat, 'Offsides'),
            'bass_possession': search_value(stat, 'Ball Possession'),
            'yellow_cards': search_value(stat, 'Yellow Cards'),
            'red_cards': search_value(stat, 'Red Cards'),
            'goal_keeper': search_value(stat, 'Goalkeeper Saves'),
            'passes': search_value_columns(stat, 'asses'),
            'form': search_value_columns(stat, 'form_'),
            'for_': search_value_columns(stat, 'for_'),
            'against': search_value_columns(stat, 'against_'),
            'preview_matches': {
                'wins_home': search_value(stat, 'wins_home'),
                'wins_away': search_value(stat, 'wins_away'),
                'draws_home': search_value(stat, 'draws_home'),
                'draws_away': search_value(stat, 'draws_away'),
                'loses_home': search_value(stat, 'loses_home'),
                'loses_away': search_value(stat, 'loses_away')
            },
            'comparison': search_value_columns(stat, 'comparison_'),
            'predict': search_value_columns(stat, 'predict_'),
            'generic_statistics': {
                'expected_goals': search_value(stat, 'expected_goals'),
                'goals_prevented': search_value(stat, 'goals_prevented'),
                'Assists': search_value(stat, 'Assists'),
                'Counter Attacks': search_value(stat, 'Counter Attacks'),
                'Cross Attacks': search_value(stat, 'Cross Attacks'),
                'Free Kicks': search_value(stat, 'Free Kicks'),
                'Goals': search_value(stat, 'Goals'),
                'Goal Attempts': search_value(stat, 'Goal Attempts'),
                'Substitutions': search_value(stat, 'Substitutions'),
                'Throwins': search_value(stat, 'Throwins'),
                'Medical Treatment': search_value(stat, 'Medical Treatment')
            }
        }

    def map_statistics(stats, multiply=True):
        """
        Mappa tutte le statistiche
        :return: Ritorna il dizionario di statistiche
        """
        if multiply:  # Con array di 2 partite
            stat_home = stats[0] if stats[0]['home_away'] == 1 else stats[1]
            stat_away = stats[0] if stats[0]['home_away'] == 0 else stats[1]

            statistic_home = Statistics(**single_statistic(stat_home, 'home'))
            statistic_away = Statistics(**single_statistic(stat_away, 'away'))

            return {
                'id_fixture': search_value(stat_home, 'id_fixture'),
                'name_home': search_value(stat_home, 'team_name'),
                'id_team_home': search_value(stat_home, 'team_id'),
                'name_away': search_value(stat_away, 'team_name'),
                'id_team_away': search_value(stat_away, 'team_id'),
                'date_match': search_value(stat_away, 'date_fixture'),
                'referee': search_value(stat_away, 'referee'),
                'round': search_value(stat_away, 'round_fixture'),
                'season': search_value(stat_away, 'season'),
                'statistics': [statistic_home, statistic_away]
            }
        else:  # Con una partita
            stat_single = stats[0]
            statistic_ = Statistics(
                **single_statistic(stat_single, 'home' if stat_single['home_away'] == 1 else 'away'))

            return {
                'id_fixture': search_value(stat_single, 'id_fixture'),
                'name_home': search_value(stat_single, 'team_name'),
                'id_team_home': search_value(stat_single, 'team_id'),
                'name_away': search_value(stat_single, 'opponent_name'),
                'id_team_away': search_value(stat_single, 'opponent_id'),
                'date_match': search_value(stat_single, 'date_fixture'),
                'referee': search_value(stat_single, 'referee'),
                'round': search_value(stat_single, 'round_fixture'),
                'season': search_value(stat_single, 'season'),
                'statistics': [statistic_]
            }

    def map_odds(odd):
        return {'odds': [Odds(**{
            'odds_from': search_value(odd, 'api_from'),
            'h2h': search_value_columns(odd, 'home_', ['_team', 'under_', 'over_']),
        })]}

    def match_stat_odd():
        """
        Migra tutte le statistiche che sono accomunate a odds + odds orfani di stat
        :return: Match
        """
        stat_odd_list = []
        for index, id_events in enumerate(list_id_odds):
            logging.info(f'<<< Row {index}/{len(list_id_odds)}')
            odd = [o for o in odds if o['id'] == id_events]
            stat_dict = {}

            # Recupero le statistiche con corrispondenza id_events(odds)
            stats = [statistic for statistic in statistics if statistic['match_id_from_odds'] == id_events]
            id_fixture = None
            if len(stats) == 2:  # ==2 Allora è nella norma
                stat_dict.update(map_statistics(stats))  # Mappa tutte le statistiche
                id_fixture = stat_dict['id_fixture']
            elif len(stats) == 1:  # =1 Potrebbe esserci una partita dove è censita solo una lega o amichevole
                logging.info(f'Trovato un solo elemento in riga {index}')
                stat_dict.update(map_statistics(stats, multiply=False))
                id_fixture = stat_dict['id_fixture']

            # Recupero le quote
            if id_fixture:  # Significa che è passato a mappare i primi dati condivisi della tabella Match
                pass
            else:  # Significa che le statistiche non ci sono e quindi è solo un odds orfano di statistics
                pass

            # Mappa tutte le quote
            stat_dict.update(map_odds(odd[0]))

            if len(stat_dict) > 0:
                match = Match(**stat_dict)
                match.id_events = id_events
                match.id_fixture = id_fixture
                stat_odd_list.append(match)

        return stat_odd_list

    matches = []
    try:
        matches.extend(match_stat_odd())  # Migra tutte le statistiche che sono accomunate a odds + odds orfani di stat

    except Exception as e:
        logging.error(str(e))
    finally:
        # Salva tutto
        repo_match.insert_massive(matches)


migrate_dataset()
# convert_csv_to_exel('../dataset/statistics/dataset_statistics_history_temp_2.csv')
# match = Match()
# match.name_home = d[0]['team_name']
# stat = Statistics()
# stat.score_ft = '{"test":"ciao"}'
# match.statistics.append(stat)
# crud.insert(match)

# matches = crud.search_all()
# for match in matches:
#     print(getattr(match, 'name_home'))
#
# dict_search = {
#     'id_match_fk': 'e3c84abc-5389-4148-b70f-0e9f84e07e8a'
# }
# match = crud.filter_by(dict_search=dict_search)
# print(getattr(match.first(), 'name_home'))

# dict_search = {
#     'id_match_fk': 'e3c84abc-5389-4148-b70f-0e9f84e07e8a',
#     'name_home': 'Lazio233333'
# }
# m = crud.update(dict_search=dict_search, field_change='name_home', value_change='Lazio2', to_dict=True)
# print(m)

# dict_search = {
#     'id_match_fk': "01a8dc4f-19e3-480d-84f2-d80a53e650ea"
# }
# crud.delete(dict_search=dict_search)
