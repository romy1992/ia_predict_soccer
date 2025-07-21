"""
FASE FINALE DI AGGREGAZIONE DEI DATASET DI STATISTICHE E ODDS STORICI

Step 3: Aggregherò con il dataset di statistiche -> VEDERE ALTRI STEP IN odds_service.py
"""
import logging
import re
from datetime import datetime

import pandas as pd
from dateutil.parser import isoparse
from rapidfuzz import fuzz

from service_ia.utility.utils import count_row_not_na, count_row_is_na, normalize_, check_name, convert_csv_to_exel

logging.basicConfig(level=logging.DEBUG)

# =============================================== STEP 3 ===============================================
base_dataset = '../dataset'
name_statistics_history = f'{base_dataset}/dataset_statistics_history.csv'  # DT storico di statistiche partite
name_odds_history = f'{base_dataset}/odds/odds_dataset.csv'  # DT storico di odds


def aggregate_ids_into_dataset(partial=False):
    """
       Dopo aver matchato i match tra odds e statistics, nel dataset statistics viene aggiunta la colonna "id"
       di riferimento dell'odds.
       Viceversa, viene inserito l'id_fixture nel dataset statistics
       :param partial :
                False: rielabora tutte le righe (utile SOLO quando il DT di odds è stato aggiornato)
                True: recupera solo i campi con "id_fixture_from_stat" a nan per vedere se c'è una nuova corrispondenza
                            con il DT di statistics (Infatti è utile SOLO se viene aggiornato il DT di statistics)
       :return: nuovi dataset con colonne
       """

    dataset_statistics = pd.read_csv(name_statistics_history, low_memory=False)
    dataset_odds = pd.read_csv(name_odds_history, low_memory=False)

    dict_dataset_statistics = dataset_statistics.to_dict(orient='records')
    dict_data_odds_updated = dataset_odds.to_dict(orient='records')

    def search_statistics():
        """
        Cerca i match passano dai nomi normalizzati e dalla data del dataset ODDS
        :return: Lista di 2 elementi che sono le squadre avversarie con tutte le sue statistiche
        """

        def check_date():
            def check_alternative_date():
                """Controlla se il dizionario ha già 2 elementi con la data principale"""
                return len(
                    [sta for sta in statistics if
                     isoparse(sta[0]['date_fixture']).date() == data_match.date()]) < 2

            date_statistic = isoparse(statistic['date_fixture']).date()
            if date_statistic == data_match.date():  # Se esiste la corrispondenza principale, allora ritorna direttamente
                return True, None
            elif check_alternative_date():
                # Altrimenti recupero l'alternativa e ritorno l'id del match ODDS
                ids_dates = re.findall(r'\(([^,]+),([^)]+)\)', row_odds['ids_dates']) \
                    if isinstance(row_odds['ids_dates'], str) else []
                alternative_ = [id_date for id_date in ids_dates if
                                isoparse(id_date[1].strip()).date() == date_statistic]
                alternative_date = alternative_[len(alternative_) - 1][1] if len(alternative_) > 0 else None
                alternative_id = alternative_[len(alternative_) - 1][0] if len(alternative_) > 0 else None
                return alternative_date is not None and alternative_id is not None, alternative_id

            return False, None

        statistics = []
        for statistic in [d for d in dict_dataset_statistics if
                          d['season'] >= 2019 and check_name(stat=d, home_odds=home_odds, away_odds=away_odds)]:
            check_date_result = check_date()
            if check_date_result[0]:
                statistics.append((statistic, check_date_result[1]))
        return statistics

    container = []

    for index_odds, row_odds in enumerate(dict_data_odds_updated):
        check_id_fix_partial = True if not partial else pd.isna(row_odds['id_fixture_from_stat'])
        if check_id_fix_partial:
            logging.info(f'Row number : {index_odds}')
            home_odds = row_odds['home_team']
            away_odds = row_odds['away_team']
            data_match = isoparse(row_odds['commence_time'])

            match_statistics = search_statistics()

            # Todo : per test
            if len(match_statistics) == 1 or len(match_statistics) > 2:
                [container.append(
                    {
                        "id_fixture": stat[0]["id_fixture"],
                        "date_fixture": stat[0]["date_fixture"],
                        "round_fixture": stat[0]["round_fixture"],
                        "team_id": stat[0]["team_id"],
                        "team_name": stat[0]["team_name"],
                        "opponent_id": stat[0]["opponent_id"],
                        "opponent_name": stat[0]["opponent_name"],
                        'type_error': len(match_statistics)
                    }
                ) for stat in match_statistics]

            # Se è di 2 partite, ci troviamo nel caso base
            # Se è di 3 Solitamente è perchè o i nomi non fanno match bene o c'è un round che è un'eliminazione diretta
            # Se è 4 è perchè può avere 2 partite con le stesse squadre ma di un range annuo ma che fanno riferimento a 2 stagioni differenti
            if 2 <= len(match_statistics) < 5:

                def added_match_into_dataset(match_event):
                    """
                    Per stessi match, inserisce l'd nei dataset
                    """
                    dataset_odds.at[index_odds, 'id_fixture_from_stat'] = match_event[0][0]['id_fixture']
                    # Aggiunta al dataset_statistics
                    for stat in match_event:
                        idx_stat = dataset_statistics[
                            (dataset_statistics['team_name'] == stat[0]['team_name']) &
                            (dataset_statistics['date_fixture'] == stat[0]['date_fixture'])
                            ].index

                        if not idx_stat.empty:
                            if match_event[0][1] is None:
                                dataset_statistics.loc[idx_stat, 'match_id_from_odds'] = row_odds['id']
                            else:
                                dataset_statistics.loc[idx_stat, 'match_id_from_odds'] = match_event[0][1]

                if len(match_statistics) == 4:
                    list_id = set([m[0]['id_fixture'] for m in match_statistics])
                    for id_match in list_id:
                        match = [m for m in match_statistics if m[0]['id_fixture'] == id_match]
                        added_match_into_dataset(match)
                elif len(match_statistics) == 3:
                    list_round = set([m[0]['round_fixture'] for m in match_statistics])
                    for r in list_round:
                        match = [m for m in match_statistics if m[0]['round_fixture'] == r]
                        if len(match) == 2:
                            added_match_into_dataset(list(match))
                elif len(match_statistics) == 2:  # Caso base con soli 2 elementi
                    added_match_into_dataset(match_statistics)

    # Update
    dataset_odds.to_csv(name_odds_history, index=False)
    dataset_statistics.to_csv(name_statistics_history, index=False)

    container = pd.DataFrame(container)  # .sort_values(by=['id_fixture', 'date_fixture', 'team_id'], ascending=True)
    container.to_excel('../dataset/analyze_statistics.xlsx', index=False)

    # Check valori
    count_ids_analyze()


def count_ids_analyze():
    df_odds = pd.read_csv(name_odds_history, low_memory=False)

    # Converte in datetime con timezone (se serve), poi rimuove timezone per sicurezza
    df_odds['commence_time'] = pd.to_datetime(df_odds['commence_time'], errors='coerce').dt.tz_localize(None)
    df_odds_filters = df_odds[df_odds['commence_time'].dt.date < datetime.today().date()].sort_values(
        by='commence_time', ascending=True)
    # Analyze odds Dataset
    analyze_none_id(df_odds_filters)

    df_stat = pd.read_csv(name_statistics_history, low_memory=False)
    print('ODDS - Con valore : ', count_row_not_na(df_odds, 'id_fixture_from_stat'))  # 7421
    print('ODDS - Senza valore alla data odierna : ', count_row_is_na(df_odds_filters, 'id_fixture_from_stat'))  # 1365
    print('STATISTICS - Con valore : ', count_row_not_na(df_stat, 'match_id_from_odds') / 2)  # 7421
    print('STATISTICS - Senza valore : ', count_row_is_na(df_stat, 'match_id_from_odds') / 2)  # 46854


def analyze_none_id(dataset=pd.read_csv(name_odds_history, low_memory=False)):
    dataset_ = dataset[['id', 'commence_time', 'home_team', 'away_team', 'ids_dates', 'id_fixture_from_stat']]
    dataset_ = dataset_.loc[dataset_['id_fixture_from_stat'].isna()].copy()
    dataset_['original_index'] = dataset_.index
    dataset_.to_excel('../dataset/analyze_none_id.xlsx', index=False)


def refused_join_insert():
    dataset_refused = pd.read_excel('../dataset/analyze_none_id.xlsx')
    dataset_history_stat = pd.read_csv(name_statistics_history, low_memory=False)
    dataset_h_dic = dataset_history_stat.to_dict(orient='records')
    dataset_history_odds = pd.read_csv(name_odds_history, low_memory=False)

    for element in dataset_refused.to_dict(orient='records'):
        name_home = element['home_team']
        name_away = element['away_team']
        commence_time = element['commence_time']
        id_event = element['id']

        # Search id_fixtures
        start_date = (commence_time - pd.Timedelta(days=5)).date()
        end_date = (commence_time + pd.Timedelta(days=5)).date()

        dict_id = [
            (index, ele['id_fixture']) for index, ele in enumerate(dataset_h_dic)
            if (
                    check_name(stat=ele, home_odds=name_home, away_odds=name_away)
                    and pd.isna(ele['match_id_from_odds'])
                    and start_date <= isoparse(ele['date_fixture']).date() <= end_date
            )
        ]

        if len(dict_id) == 2 and dict_id[0][1] == dict_id[1][1]:
            rows = [dict_id[0][0], dict_id[1][0]]
            id_fix = dict_id[0][1]
            for index in rows:
                dataset_history_stat.loc[index, 'match_id_from_odds'] = id_event
            dataset_history_odds[element['original_index']] = id_fix

    dataset_history_stat.to_csv(name_statistics_history, index=False)
    dataset_history_odds.to_csv(name_odds_history, index=False)


# TODO 19/07 : CI SONO ANCORA MATCH SENZA CORRISPONDEZA DOVUTI DA ALCUNE PARTITE DEL DF ODDS NON HANNO
#  L'ALTERNATIVA E CORRISPONDEZA PRINCIPALE..
#  DA PROVARE SE I LORO ID SONO STATI DISABILITATI

# aggregate_ids_into_dataset(partial=True)
refused_join_insert()
# count_ids_analyze()

# print(max(fuzz.ratio(normalize_('Elche CF'), normalize_('Elche CF')),
#           fuzz.partial_ratio(normalize_('Elche CF'), normalize_('Elche CF'))))

# convert_excel_to_csv('../dataset/odds/odds_dataset_replace.xlsx')
# convert_excel_to_csv('../dataset/dataset_statistics_history.xlsx')
# convert_csv_to_exel('dataset_test.csv')
