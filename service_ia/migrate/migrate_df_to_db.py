"""
Operazione di migrazione da dataset al db.
Nasce dalla necessità di avere tutto organizzato in un db e non nei csv o excel
"""
import logging

import pandas as pd

from repository.match_repository import MatchRepository
from service_ia.model.match import Match, Statistics
from service_ia.utility.utils import convert_csv_to_exel

logging.basicConfig(level=logging.DEBUG)


def migrate_dataset():
    statistics = pd.read_csv('../dataset/statistics/dataset_statistics_history.csv', low_memory=False).to_dict(
        orient='records')
    odds = pd.read_csv('../dataset/odds/odds_dataset_copy.csv', low_memory=False).to_dict(
        orient='records')
    repo_match = MatchRepository()

    list_id_fixture = set([statistic['id_fixture'] for statistic in statistics])

    list_id_odds = set([odd['id'] for odd in odds])

    matches = []
    try:
        for index, id_events in enumerate(list_id_odds):
            logging.info(f'<<< Row {index}/{len(list_id_odds)}')
            odd = [o for o in odds if o['id'] == id_events]
            if len(odd) == 1:
                stats = [statistic for statistic in statistics if statistic['match_id_from_odds'] == id_events]
                stat_dict = {}
                if len(stats) == 2:
                    def search_value(stat, field):
                        return stat[field]

                    stat_home = stats[0] if stats[0]['home_away'] == 1 else stats[1]
                    stat_away = stats[0] if stats[0]['home_away'] == 0 else stats[1]
                    stat_dict.update({
                        'id_fixture': search_value(stat_home, 'id_fixture'),
                        'name_home': search_value(stat_home, 'team_name'),
                        'id_team_home': search_value(stat_home, 'team_id'),
                        'name_away': search_value(stat_away, 'team_name'),
                        'id_team_away': search_value(stat_away, 'team_id'),
                        'date_match': search_value(stat_away, 'date_fixture'),
                        'referee': search_value(stat_away, 'referee'),
                        'round': search_value(stat_away, 'round_fixture'),
                        'season': search_value(stat_away, 'season'),
                    })

                    statistic = Statistics(**{
                        'score_ht': {
                            'score_home': search_value(stat_home, 'score_home_ht'),
                            'score_away': search_value(stat_away, 'score_away_ht'),
                        },
                        'score_ft': {
                            'score_home': search_value(stat_home, 'score_home_ft'),
                            'score_away': search_value(stat_away, 'score_away_ft'),
                        }
                    })

                    stat_dict.update({
                        'statistics': [statistic]
                    })

                match = Match(**stat_dict)
                match.id_events = id_events

                if match.id_events or match.id_fixture:
                    matches.append(match)
            elif len(odd) > 1:
                print(odd, '> 1 elemento')
            else:  # == 0 : Caso solo statistics
                pass

    except Exception as e:
        logging.error(str(e))
    finally:
        # Salva tutto
        if len(matches) > 0:
            repo_match.insert_massive(matches)


# migrate_dataset()
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
