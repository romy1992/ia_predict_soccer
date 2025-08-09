"""
Operazione di migrazione da dataset al db.
Nasce dalla necessità di avere tutto organizzato in un db e non nei csv o excel
"""
import logging
from datetime import datetime

import pandas as pd

from repository.match_repository import MatchRepository
from service_ia.model.match import Match, Statistics, Odds

logging.basicConfig(level=logging.DEBUG)
pd.set_option('future.no_silent_downcasting', True)


def migrate_dataset():
    statistics = pd.read_csv('../dataset/statistics/dataset_statistics_history.csv', low_memory=False).to_dict(
        orient='records')
    odds = pd.read_csv('../dataset/odds/odds_dataset_copy.csv', low_memory=False).to_dict(
        orient='records')
    repo_match = MatchRepository()

    list_id_odds = set([odd['id'] for odd in odds])

    def search_value(stat, field):
        """
        Cerca il valore nel dizionario singolarmente
        :param stat: statistica corrente di home o away
        :param field: nome della key
        :return: singolo valore
        """
        value = stat.get(field)
        return value if pd.notna(value) else None

    def search_value_columns(stat, col_value, not_include=None):
        """
        Cerca i valori nel dizionario con somiglianza nel nome colonna
        :param stat: statistica corrente di home o away
        :param col_value: utilizzato per "la somiglianza" della colonna
        :param not_include: se valorizzato, è una lista che serve a escludere i col_value che hanno somiglianze ma che non servono
        :return: {chiave:valore}
        """
        if not_include:
            list_include = {key: value for key, value in stat.items() if pd.notna(value) and col_value in key}
            return {key: value for key, value in list_include.items()
                    if all(inc not in key for inc in not_include)}
        else:
            return {key: value for key, value in stat.items() if pd.notna(value) and col_value in key}

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
                'current_league': search_value(stat_away, 'current_league'),
                'league_match': search_value(stat_away, 'league_match'),
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
        dc = search_value_columns(odd, '1X')
        dc.update(search_value_columns(odd, 'X2'))
        dc.update(search_value_columns(odd, '12'))

        under_over_2_5 = search_value_columns(odd, '2.5', ['corner_', 'card_'])
        under_over_2_5.update(search_value_columns(odd, '2_5', ['_home', '_away']))

        under_over_home_away = search_value_columns(odd, '_home')
        under_over_home_away.update(search_value_columns(odd, '_away'))

        return {'odds': [Odds(**{
            'odds_from': search_value(odd, 'api_from'),
            'h2h': search_value_columns(odd, 'home_', ['_team', 'under_', 'over_']),
            'under_over_1_5': search_value_columns(odd, '1_5', ['_home', '_away']),
            'under_over_2_5': under_over_2_5,
            'under_over_3_5': search_value_columns(odd, '3_5', ['_home', '_away']),
            'under_over_4_5': search_value_columns(odd, '4_5', ['_home', '_away']),
            'under_over_home_away': under_over_home_away,
            'goal_no_goal': search_value_columns(odd, 'goal'),
            'corners': search_value_columns(odd, 'corner'),
            'cards': search_value_columns(odd, 'card'),
            'dc': dc
        })]}

    def match_stat_odd():
        """
        Migra tutte le statistiche che sono accomunate a odds + odds orfani di stat
        :return: Match
        """
        stat_odd_list = []
        for index, id_events in enumerate(list_id_odds):
            logging.info(f'<<< Row odds {index}/{len(list_id_odds)}')
            odd = [o for o in odds if o['id'] == id_events][0]
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
            if not id_fixture:  # Significa che le statistiche non ci sono e quindi è solo un odds orfano di statistics
                stat_dict.update({'name_home': search_value(odd, 'home_team'),
                                  'name_away': search_value(odd, 'away_team'),
                                  'date_match': search_value(odd, 'commence_time')})

            # Standard di mapping odds
            id_alternate_events = search_value(odd, 'ids_dates')
            id_alternate = None
            date_alternate = None
            if id_alternate_events:
                id_alternate_events = str(id_alternate_events).replace('(', '').replace(')', '').strip().split(',')
                if len(id_alternate_events) == 2:
                    id_alternate = id_alternate_events[0]
                    date_alternate = id_alternate_events[1]
                elif len(id_alternate_events) > 2:
                    id_alternate = id_alternate_events[len(id_alternate_events) - 3]
                    date_alternate = id_alternate_events[len(id_alternate_events) - 2]

            stat_dict.update({
                'id_alternate_events': id_alternate,
                'date_alternate_match': date_alternate,
                'sport_key': search_value(odd, 'sport_key'),
                'title_league': search_value(odd, 'sport_title'),
            })

            # Mappa tutte le quote
            stat_dict.update(map_odds(odd))

            if len(stat_dict) > 0:
                match = Match(**stat_dict)
                match.id_events = id_events
                match.id_fixture = id_fixture
                stat_odd_list.append(match)

        return stat_odd_list

    def match_only_stat():
        """
        Migra solo le statistiche SENZA odds
        :return: Match
        """
        stat_odd_list = []
        all_matchs = repo_match.search_all()
        ids_db_fixtures = [id_ for id_ in [m.to_dict().get('id_fixture') for m in all_matchs] if id_]
        orphan_stat_id = set([stat['id_fixture'] for stat in statistics if stat['id_fixture'] not in ids_db_fixtures])
        for index, ids_ in enumerate(orphan_stat_id):
            logging.info(f'<<< Row stat {index}/{len(orphan_stat_id)}')
            stat_dict = {}

            # Recupero le statistiche con corrispondenza id_events(odds)
            stats = [statistic for statistic in statistics if statistic['id_fixture'] == ids_]
            id_fixture = None
            if len(stats) == 2:  # ==2 Allora è nella norma
                stat_dict.update(map_statistics(stats))  # Mappa tutte le statistiche
                id_fixture = stat_dict['id_fixture']
            elif len(stats) == 1:  # =1 Potrebbe esserci una partita dove è censita solo una lega o amichevole
                logging.info(f'Trovato un solo elemento in riga {index}')
                stat_dict.update(map_statistics(stats, multiply=False))
                id_fixture = stat_dict['id_fixture']

            if len(stat_dict) > 0:
                match = Match(**stat_dict)
                match.id_fixture = id_fixture
                stat_odd_list.append(match)

        return stat_odd_list

    matches = []
    try:
        # matches.extend(match_stat_odd())  # Migra tutte le statistiche che sono accomunate a odds + odds orfani di stat
        matches.extend(match_only_stat())  # Migra solo le statistiche SENZA odds
    except Exception as e:
        logging.error(str(e))
    finally:
        # Salva tutto
        repo_match.insert_massive(matches)


# migrate_dataset()

columns_mean = ['Shots on Goal', 'Shots off Goal', 'Total Shots', 'Blocked Shots', 'Shots insidebox',
                'Shots outsidebox', 'Fouls',
                'Corner Kicks', 'Offsides', 'Ball Possession', 'Yellow Cards', 'Red Cards',
                'Goalkeeper Saves',
                'Total passes', 'Passes accurate', 'Passes %']


def create_dataset_mean():
    dataset = pd.read_csv('../dataset/statistics/dataset_statistics_history.csv', low_memory=False)
    dataset['date_fixture'] = pd.to_datetime(dataset['date_fixture']).dt.tz_localize(None)
    dataset = dataset.to_dict(orient='records')
    seasons = set([season['season'] for season in dataset])
    id_teams = set([team['team_id'] for team in dataset])
    df_final = pd.DataFrame()
    all_mean = []

    def replace_dataset(df):
        """
         Ripulisco il dizionario eliminando la '%' e calcolandola
         Trova colonne con valori di tipo stringa e con '%'
        """
        for record in df:
            for col, value in record.items():
                try:
                    if isinstance(value, str) and '%' in value:
                        record[col] = float(value.replace('%', '')) / 100
                except Exception:
                    pass  # ignora valori non gestibili
        return df

    for season in seasons:
        for id_team in id_teams:
            # Recupero le partite della squadra della stagione
            team = replace_dataset([df for df in dataset if df['team_id'] == id_team and df['season'] == season])
            if len(team) > 0:
                team = sorted(team, key=lambda x: x["date_fixture"])
                for date_match in team:
                    team_fixture = [t for t in team if t['date_fixture'] < date_match['date_fixture']]
                    mean_team = team_fixture[columns_mean].mean()
                    all_mean.append(mean_team)  # Aggiungo le colonne della media al DataFrame originale
                # Imposto tutte le medie come dataset e ci aggiungo il prefisso
                dataset_mean = pd.DataFrame(all_mean, columns=columns_mean).add_prefix('mean_')
                team = team.reset_index(drop=True)
                dataset_mean = dataset_mean.reset_index(drop=True)
                team = pd.concat([team, dataset_mean], axis=1)  # Le concateno
                team = team.iloc[1:]  # Rimuove la prima riga perché sarà sempre senza statistiche essendo la prima
                all_mean = []  # Reimposto a 0 il dizionario delle medie
                df_final = pd.concat([df_final.fillna(0), team.fillna(0)], axis=0)  # Incolonno uno sotto l'altro

    # Salvo in Excel
    df_final.to_excel("dataset_mean_history.xlsx", index=False)

def create_dataset_mean_2():
    # Legge CSV
    dataset_df = pd.read_csv('../dataset/statistics/dataset_statistics_history.csv', low_memory=False)
    dataset_df['date_fixture'] = pd.to_datetime(dataset_df['date_fixture']).dt.tz_localize(None)

    # Rimuove % e converte in decimali
    for col in dataset_df.columns:
        if dataset_df[col].dtype == object and dataset_df[col].str.contains('%', na=False).any():
            dataset_df[col] = dataset_df[col].str.replace('%', '', regex=False).astype(float) / 100

    # Converte in lista di dizionari
    dataset = dataset_df.to_dict(orient='records')

    seasons = set(season['season'] for season in dataset)
    id_teams = set(team['team_id'] for team in dataset)

    df_final = pd.DataFrame()

    for season in seasons:
        for id_team in id_teams:
            # Filtra partite della squadra in quella stagione
            team = [rec for rec in dataset if rec['team_id'] == id_team and rec['season'] == season]

            if team:
                # Ordina per data
                team = sorted(team, key=lambda x: x["date_fixture"])

                mean_rows = []
                for idx, match in enumerate(team):
                    if idx == 0:
                        mean_rows.append([None] * len(columns_mean))
                    else:
                        prev_matches = team[:idx]
                        prev_df = pd.DataFrame(prev_matches)
                        mean_rows.append(prev_df[columns_mean].mean().tolist())

                # DataFrame delle medie
                dataset_mean = pd.DataFrame(mean_rows, columns=columns_mean).add_prefix('mean_')

                # Converte team in DataFrame per concatenare
                team_df = pd.DataFrame(team).reset_index(drop=True)
                team_df = pd.concat([team_df, dataset_mean], axis=1)

                df_final = pd.concat([df_final, team_df], ignore_index=True)

    df_final.to_excel("dataset_mean_history.xlsx", index=False)
    print("✅ Dataset salvato con medie in 'dataset_mean_history.xlsx'")

create_dataset_mean_2()
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
