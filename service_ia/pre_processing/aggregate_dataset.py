"""
FASE FINALE DI AGGREGAZIONE DEI DATASET DI STATISTICHE E ODDS STORICI

Step 3: Aggregherò con il dataset di statistiche -> VEDERE ALTRI STEP IN odds_service.py
"""
import logging
import re
from datetime import datetime

import pandas as pd
import unicodedata
from dateutil.parser import isoparse
from rapidfuzz import fuzz

from service_ia.utility.utils import count_row_not_na, count_row_is_na

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

        def check_name():
            """
            Cerca la corrispondenza dei nomi simili tra squadra di fuori casa e casa
            :return: Booleano se il match è stato trovato secondo i nomi delle squadre
            """
            soglia = 90

            def normalize_(name):
                """
                Normalizza il nome della squadra togliendo alcuni path non idonei
                :param name: Nome della squadra
                :return: Nome squadra normalizzata
                """
                name = name.lower()
                # Rimuove gli accenti
                name = ''.join(
                    c for c in unicodedata.normalize('NFD', name)
                    if unicodedata.category(c) != 'Mn'
                )
                for r in ["fc", "bc", "calcio", "football", ".", ",", "-", "ssd", "asd", "club", "1899", "1929"]:
                    name = name.replace(r, "")
                return " ".join(name.split()).strip()

            return (
                    (max
                     (fuzz.ratio(normalize_(statistic['team_name']), normalize_(home_odds)),
                      fuzz.partial_ratio(normalize_(statistic['team_name']), normalize_(home_odds))) >= soglia
                     and statistic['home_away'] == 1)
                    or
                    (max(fuzz.ratio(normalize_(statistic['team_name']), normalize_(away_odds)),
                         fuzz.partial_ratio(normalize_(statistic['team_name']),
                                            normalize_(away_odds))) >= soglia
                     and statistic['home_away'] == 0)
            )

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
        for statistic in dict_dataset_statistics:
            if statistic['season'] >= 2019 and check_name():
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
            if len(match_statistics) == 1:
                id_f = match_statistics[0][0]['id_fixture']
                [container.append(stat) for stat in dict_dataset_statistics if stat['id_fixture'] == id_f]

            if len(match_statistics) == 2:
                dataset_odds.at[index_odds, 'id_fixture_from_stat'] = match_statistics[0][0]['id_fixture']

                # Aggiunta al dataset_statistics
                for stat in match_statistics:
                    idx_stat = dataset_statistics[
                        (dataset_statistics['team_name'] == stat[0]['team_name']) &
                        (dataset_statistics['date_fixture'] == stat[0]['date_fixture'])
                        ].index

                    if not idx_stat.empty:
                        if match_statistics[0][1] is None:
                            dataset_statistics.loc[idx_stat, 'match_id_from_odds'] = row_odds['id']
                        else:
                            dataset_statistics.loc[idx_stat, 'match_id_from_odds'] = match_statistics[0][1]

    # Update
    dataset_odds.to_csv(name_odds_history, index=False)
    dataset_statistics.to_csv(name_statistics_history, index=False)

    container = pd.DataFrame(container)
    container.to_excel('../dataset/analyze_statistics.xlsx')

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
    dataset_ = dataset_.loc[dataset_['id_fixture_from_stat'].isna()]
    dataset_.to_excel('../dataset/analyze_none_id.xlsx', index=False)


# TODO 17/07 : CI SONO ANCORA MATCH SENZA CORRISPONDEZA DOVUTI DA ALCUNE PARTITE DE DF ODDS NON HANNO L'ALTERNATIVA E CORRISPONDEZA PRINCIPALE..
#  DA PROVARE SE I LORO ID SONO STATI DISABILITATI
aggregate_ids_into_dataset(partial=True)
# count_ids_analyze()

# def normalize_(name):
#     """
#     Normalizza il nome della squadra togliendo alcuni path non idonei
#     :param name: Nome della squadra
#     :return: Nome squadra normalizzata
#     """
#     name = name.lower()
#     # Rimuove gli accenti
#     name = ''.join(
#         c for c in unicodedata.normalize('NFD', name)
#         if unicodedata.category(c) != 'Mn'
#     )
#     for r in ["fc", "bc", "calcio", "football", ".", ",", "-", "ssd", "asd", "club", "1899", "1929"]:
#         name = name.replace(r, "")
#     return " ".join(name.split()).strip()
#
#
# print(max(fuzz.ratio(normalize_('US Catanzaro 1929'), normalize_('Catanzaro')),
#           fuzz.partial_ratio(normalize_('US Catanzaro 1929'), normalize_('Catanzaro'))))

# convert_excel_to_csv('../dataset/odds/odds_dataset_replace.xlsx')
# convert_excel_to_csv('../dataset/dataset_statistics_history.xlsx')
# convert_csv_to_exel('../dataset/dataset_statistics_history.csv')
