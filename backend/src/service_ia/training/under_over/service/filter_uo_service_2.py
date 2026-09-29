import statistics

import pandas as pd

from src.repository.match_repository import MatchRepository
from src.service_ia.utility.utils import convert_orm_match_to_dict, adapted_percentage


class FilterUOService:
    """
    Servizio per filtrare le partite con evento under/over
     e generare il dataset per l'addestramento o la predizione
    1. Filtra le partite in base ai filtri passati
    2. Genera il dataset in base alle partite filtrate
    3. Splitta il dataset in X e y
    4. Restituisce X, y e le chiavi delle feature
    5. Se predict è True, genera il dataset per la predizione (senza y)
    6. Se predict è False, genera il dataset per l'addestramento (con y)
    """

    def __init__(self, **kwargs):
        """
        Classe per processare il report delle partite e generare il dataset per l'addestramento
        :param event: 'under_over_1_5', 'under_over_2_5', 'under_over_3_5'
        :param list_stats: lista di nodi dict es: ['form','passes','mean_statistics']
        """
        self.match_repo = MatchRepository()  # Repository per le partite
        self.predict = kwargs.get('predict', False)
        self.list_stats = kwargs.get('list_stats', [])
        self.events = ['under_over_1_5', 'under_over_2_5', 'under_over_3_5']
        self.filters_base = {
            'status': ['FT'],
            'mean_statistics': "not None",
            'odds': "not None",
            'statistics': "not None",
            'season': [2020, 2021, 2022, 2023, 2024, 2025]}

    # --------------------SINGLE EVENT METHODS--------------------
    def get_matches_single_event(self, event: str, filters: dict = None) -> list:
        """
        Genera il dataset in base alle partite filtrate
        :param: predict: se True, genera il dataset per la predizione (senza y)
        :return: dataset
        """
        matches_dict = self.filter_dict_matches(filters)
        matches_dict = [m for m in matches_dict
                        if (m['odds'] and m['odds'][0][event] and len(m['odds'][0][event]) > 0)
                        and len(m['mean_statistics']) > 0]
        dataset = []
        for match in matches_dict:
            statistics_match = match['statistics']
            if len(statistics_match) == 2 or self.predict:
                stat_home = stat_away = None
                if not self.predict:
                    # Statistiche totali con form ,comparison e predict
                    stat_home = statistics_match[0] if statistics_match[0]['statistics_team_id'] == match[
                        'id_team_home'] else \
                        statistics_match[1]
                    stat_away = statistics_match[1] if statistics_match[1]['statistics_team_id'] == match[
                        'id_team_away'] else \
                        statistics_match[0]

                match_records = {'id_fixture': match['id_fixture']}
                if self.predict:  # Se è per predire non metto la y
                    odds = match['odds'][0][event]
                else:  # Se è per addestrare metto la y
                    def switch_under_over():
                        """
                        In base al get_matches.event, switch l'evento under over 1.5, 2.5 o 3.5
                        get_matches.event: 1.5, 2.5 o 3.5
                        :return: odds event + y result
                        """
                        r = None
                        total = stat_home['score_ft'] + stat_away['score_ft']
                        value = 2.5 if event == 'under_over_2_5' else 1.5 if event == 'under_over_1_5' else 3.5
                        match value:
                            case 1.5:
                                r = 1 if total >= 2 else 0
                            case 2.5:
                                r = 1 if total >= 3 else 0
                            case 3.5:
                                r = 1 if total >= 4 else 0
                        return match['odds'][0][event], r

                    odds, result = switch_under_over()
                    match_records.update({'id_fixture': match['id_fixture'], 'y': result})

                match_records.update({
                    'name_home': match['name_home'],
                    'name_away': match['name_away']})
                match_records.update({key: value for key, value in odds.items() if pd.notna(value)})

                if self.list_stats:  # Se bisogna aggiungere le statistiche
                    stats = self.get_specific_stat(match=match, include=['mean_expected_goals'],
                                                   stat_home=stat_home, stat_away=stat_away)

                    # Se ha statistiche con valore !=0
                    if len(stats) > 0:
                        match_records.update(stats)
                        dataset.append(match_records)
                else:
                    dataset.append(match_records)

        return dataset

    def split_dataset_single_event(self, dataset: dict, type_calculation='mean') -> pd.DataFrame:
        """
        Splitta il dataset in X e y
        :param dataset: dataset da splittare
        :param type_calculation: tipo di calcolo 'std' o 'mean'
        :return: X, y, keys
        """

        def generate_feature(element):
            """
            Genera le feature in base al tipo t (std o mean)
            :param element: elemento del dataset
            :return: record con le feature calcolate
            """
            # Statistiche
            stat_itm = {k: float(v) if pd.notna(v) else 0 for k, v in element.items() if '_stat' in k}

            def calculate_label():
                """
                Calcolo under over in base al tipo di calcolo
                1. Recupero tutte le quote under e over
                2. Calcolo la std o la mean
                3. Restituisco i valori calcolati
                :return: calculate_under, calculate_over
                """
                # Recupero tutte le quote tranne il result label e le statistiche
                under_itm = {k: float(v) if pd.notna(v) else 0 for k, v in element.items() if
                             'under' in k and 'stat' not in k}
                over_itm = {k: float(v) if pd.notna(v) else 0 for k, v in element.items() if
                            'over' in k and 'stat' not in k}

                if under_itm and under_itm:
                    if type_calculation == 'std':
                        if len(under_itm) >= 2 or len(over_itm) >= 2:
                            # Calcolo std
                            return statistics.stdev(list(under_itm.values())), statistics.stdev(list(over_itm.values()))
                        else:
                            return None, None
                    elif type_calculation == 'mean':
                        # Calcolo la media
                        return statistics.mean(list(under_itm.values())), statistics.mean(list(over_itm.values()))
                    else:
                        raise Exception
                else:
                    return None, None

            # Calcolo under over
            calculate_under, calculate_over = calculate_label()

            # Costruisco il record solo se ho calcolato under e over
            if calculate_over is not None and calculate_under is not None:
                # Costruisco il record
                if self.predict:
                    return {
                        'id_fixture': element['id_fixture'],
                        'name_home': element['name_home'],
                        'name_away': element['name_away'],
                        f"{type_calculation}_under": calculate_under,
                        f"{type_calculation}_over": calculate_over,
                        **stat_itm,  # aggiunge tutte le statistiche con i loro nomi
                    }

                return {
                    'id_fixture': element['id_fixture'],
                    'name_home': element['name_home'],
                    'name_away': element['name_away'],
                    f"{type_calculation}_under": calculate_under,
                    f"{type_calculation}_over": calculate_over,
                    **stat_itm,  # aggiunge tutte le statistiche con i loro nomi
                    "y": element['y']
                }
            else:
                return {}

        df = [generate_feature(element) for element in dataset]
        df = [d for d in df if len(d) > 0]  # Elimino i record vuoti
        df = pd.DataFrame(df).fillna(0)
        return df

    # --------------------MULTILABELS METHODS--------------------
    def get_match_multilabel_events(self, filters):
        # TODO : controllare bene il giro di multilabel
        matches_dict = self.filter_dict_matches(filters)
        dataset = []
        for match in matches_dict:
            if any(k for k in match['odds'][0].keys() if
                   k in self.events and match['odds'][0][k] and len(match['odds'][0][k]) > 0):
                statistics_match = match['statistics']
                if len(statistics_match) == 2 or self.predict:
                    stat_home = stat_away = None
                    if not self.predict:
                        # Statistiche totali con form ,comparison e predict
                        stat_home = statistics_match[0] if statistics_match[0]['statistics_team_id'] == match[
                            'id_team_home'] else \
                            statistics_match[1]
                        stat_away = statistics_match[1] if statistics_match[1]['statistics_team_id'] == match[
                            'id_team_away'] else \
                            statistics_match[0]

                    match_records = {'id_fixture': match['id_fixture']}
                    odds = {e: match['odds'][0][e] for e in self.events}
                    match_records.update({key: value for key, value in odds.items() if pd.notna(value)})
                    if not self.predict:  # Se è per addestrare metto la y
                        total = stat_home['score_ft'] + stat_away['score_ft']
                        u_o_1_5 = 1 if total >= 2 else 0
                        u_o_2_5 = 1 if total >= 3 else 0
                        u_o_3_5 = 1 if total >= 4 else 0
                        result = [u_o_1_5, u_o_2_5, u_o_3_5]
                        match_records.update({'id_fixture': match['id_fixture'], 'y': result})

                    if self.list_stats:  # Se bisogna aggiungere le statistiche
                        stats = self.get_specific_stat(match=match, include=['mean_expected_goals'],
                                                       stat_home=stat_home, stat_away=stat_away)

                        # Se ha statistiche con valore !=0
                        if len(stats) > 0:
                            match_records.update(stats)
                            dataset.append(match_records)
                    else:
                        dataset.append(match_records)

        return dataset

    def split_dataset_multilabel(self, dataset, type_calculation='mean'):
        """
        Splitta il dataset in X e y
        :param dataset: dataset da splittare
        :param type_calculation: tipo di calcolo 'std' o 'mean'
        :return: X, y, keys
        """

        def generate_feature(element):
            """
            Genera le feature in base al tipo t (std o mean)
            :param element: elemento del dataset
            :return: record con le feature calcolate
            """
            # Statistiche
            stat_itm = {k: float(v) if pd.notna(v) else 0 for k, v in element.items() if '_stat' in k}

            def calculate_label(event):
                """
                Calcolo under over in base al tipo di calcolo
                1. Recupero tutte le quote under e over
                2. Calcolo la std o la mean
                3. Restituisco i valori calcolati
                :return: calculate_under, calculate_over
                """
                under_itm = {k: float(v) if pd.notna(v) else 0 for k, v in element[event].items() if
                             'under' in k and 'stat' not in k}
                over_itm = {k: float(v) if pd.notna(v) else 0 for k, v in element[event].items() if
                            'over' in k and 'stat' not in k}

                if under_itm and under_itm:
                    if type_calculation == 'std':
                        if len(under_itm) >= 2 or len(over_itm) >= 2:
                            # Calcolo std
                            return statistics.stdev(list(under_itm.values())), statistics.stdev(list(over_itm.values()))
                        else:
                            return None, None
                    elif type_calculation == 'mean':
                        # Calcolo la media
                        return statistics.mean(list(under_itm.values())), statistics.mean(list(over_itm.values()))
                    else:
                        raise Exception
                else:
                    return None, None

            records = {}
            for e in self.events:
                # Calcolo under over
                calculate_under, calculate_over = calculate_label(e)
                # Costruisco il record solo se ho calcolato under e over
                if calculate_over is not None and calculate_under is not None:
                    key_event = e.replace('under_over_', '')
                    # Costruisco il record
                    records.update({
                        f"{type_calculation}_under_{key_event}": calculate_under,
                        f"{type_calculation}_over_{key_event}": calculate_over,
                        **stat_itm  # aggiunge tutte le statistiche con i loro nomi
                    })
            if not self.predict:
                records.update({"y": element['y']})
            return records

        df = [generate_feature(element) for element in dataset]
        df = [d for d in df if len(d) > 0]  # Elimino i record vuoti
        df = pd.DataFrame(df).fillna(0)

        if self.predict:
            return df['id_fixture'].values, df.drop(columns=['id_fixture']).values

        return (
            df.drop(columns=['y']).values,  # X
            df['y'].values,  # y
            df.drop(columns=['y']).columns.tolist())  # keys

    # --------------------TOTAL GOALS METHODS--------------------
    def get_match_total_goals(self, filters: dict = None) -> list:
        matches_dict = self.filter_dict_matches(filters)
        dataset = []
        for match in matches_dict:
            # controllo che ci sia almeno un evento under/over con quote
            if (match['odds'] and len(match['odds'][0]) > 0 and
                    any(k for k in match['odds'][0].keys() if k in self.events and match['odds'][0][k] and len(
                        match['odds'][0][k]) > 0)):
                statistics_match = match['statistics']
                if len(statistics_match) == 2 or self.predict:
                    stat_home = stat_away = None
                    if not self.predict:
                        # Statistiche totali con form ,comparison e predict
                        stat_home = statistics_match[0] if statistics_match[0]['statistics_team_id'] == match[
                            'id_team_home'] else statistics_match[1]
                        stat_away = statistics_match[1] if statistics_match[1]['statistics_team_id'] == match[
                            'id_team_away'] else statistics_match[0]

                    match_records = {'id_fixture': match['id_fixture'],
                                     'name_home': match['name_home'],
                                     'name_away': match['name_away']}

                    odds = {e: match['odds'][0][e] for e in self.events}
                    match_records.update({key: value for key, value in odds.items() if pd.notna(value)})
                    if not self.predict:  # Se è per addestrare metto la y
                        result = stat_home['score_ft'] + stat_away['score_ft']
                        match_records.update({'id_fixture': match['id_fixture'], 'y': result})

                    if self.list_stats:  # Se bisogna aggiungere le statistiche
                        stats = self.get_specific_stat(match=match, include=['mean_expected_goals'],
                                                       stat_home=stat_home, stat_away=stat_away)

                        # Se ha statistiche con valore !=0
                        if len(stats) > 0:
                            match_records.update(stats)
                            dataset.append(match_records)
                    else:
                        dataset.append(match_records)

        return dataset

    def split_dataset_total_goals(self, dataset, type_calculation: str = 'mean') -> pd.DataFrame:
        """
        Splitta il dataset in X e y
        :param dataset: dataset da splittare
        :param type_calculation: tipo di calcolo 'std' o 'mean'
        :return: X, y, keys
        """

        def generate_feature(element):
            """
            Genera le feature in base al tipo t (std o mean)
            :param element: elemento del dataset
            :return: record con le feature calcolate
            """
            # Statistiche
            stat_itm = {k: float(v) if pd.notna(v) else 0 for k, v in element.items() if '_stat' in k}

            def calculate_label(event):
                """
                Calcolo under over in base al tipo di calcolo
                1. Recupero tutte le quote under e over
                2. Calcolo la std o la mean
                3. Restituisco i valori calcolati
                :return: calculate_under, calculate_over
                """
                under_itm = {k: float(v) if pd.notna(v) else 0 for k, v in element[event].items() if
                             'under' in k and 'stat' not in k}
                over_itm = {k: float(v) if pd.notna(v) else 0 for k, v in element[event].items() if
                            'over' in k and 'stat' not in k}

                if under_itm and under_itm:
                    if type_calculation == 'std':
                        if len(under_itm) >= 2 or len(over_itm) >= 2:
                            # Calcolo std
                            return statistics.stdev(list(under_itm.values())), statistics.stdev(list(over_itm.values()))
                        else:
                            return None, None
                    elif type_calculation == 'mean':
                        # Calcolo la media
                        return statistics.mean(list(under_itm.values())), statistics.mean(list(over_itm.values()))
                    else:
                        raise Exception
                else:
                    return None, None

            records = {}
            for e in self.events:
                # Calcolo under over
                calculate_under, calculate_over = calculate_label(e)
                # Costruisco il record solo se ho calcolato under e over
                if calculate_over is not None and calculate_under is not None:
                    key_event = e.replace('under_over_', '')
                    # Costruisco il record
                    records.update({
                        'id_fixture': element['id_fixture'],
                        'name_home': element['name_home'],
                        'name_away': element['name_away'],
                        f"{type_calculation}_under_{key_event}": calculate_under,
                        f"{type_calculation}_over_{key_event}": calculate_over,
                        **stat_itm  # aggiunge tutte le statistiche con i loro nomi
                    })
            if not self.predict:
                records.update({"y": element['y']})
            return records

        df = [generate_feature(element) for element in dataset]
        df = [d for d in df if len(d) > 0]  # Elimino i record vuoti
        df = pd.DataFrame(df).fillna(0)
        return df

    # --------------------UTILITY METHODS--------------------
    def filter_dict_matches(self, filters: dict = None):
        matches = self.match_repo.search_filter(filters=filters if filters else self.filters_base)
        return convert_orm_match_to_dict(matches)

    def get_specific_stat(self, match, **kwargs):
        """
        Aggrega tutte le statistiche che mi servono per addestrare
        :param match: dict della partita
        :param kwargs:
            :exclude: casi in cui serve la lista di elementi dict che non servono es: ['Total passes']->
                exclude=['id_team','mean_Shots on Goal', 'mean_Shots off Goal', 'mean_Blocked Shots', 'mean_Shots insidebox',
                        'mean_Shots outsidebox', 'mean_Corner Kicks','mean_Fouls','mean_Offsides','mean_Yellow Cards',
                         'mean_Red Cards','mean_Goalkeeper Saves','mean_Total passes', 'mean_Passes %', 'mean_Passes accurate']
            :include: casi in cui serve la lista di elementi dict che servono es: ['expected_goal']
            :stat_home: dict statistiche della squadra di casa
            :stat_away: dict statistiche della squadra ospite
        :return: dict di statistiche di home e away
        """
        stat_home = kwargs.get('stat_home', {})
        stat_away = kwargs.get('stat_away', {})
        exclude = kwargs.get('exclude', None)
        include = kwargs.get('include', None)

        # Solo media dei alcune statistiche principali
        mean_stat = match['mean_statistics']
        mean_stat_home = None
        mean_stat_away = None
        if mean_stat and len(mean_stat) == 2:
            mean_stat_home = mean_stat[0] if mean_stat[0]['id_team'] == match['id_team_home'] else \
                mean_stat[1]
            mean_stat_away = mean_stat[1] if mean_stat[1]['id_team'] == match['id_team_away'] else \
                mean_stat[0]

        match_stat = {}
        for stat in self.list_stats:  # Ciclo tutti dict delle feature che mi servono
            if 'mean_statistics' == stat:
                if mean_stat_home and mean_stat_away:
                    if exclude:  # Elimino features che non mi servono
                        mean_stat_home = {key: value for key, value in mean_stat_home.items() if
                                          key not in exclude}
                        mean_stat_away = {key: value for key, value in mean_stat_away.items() if
                                          key not in exclude}
                    elif include:  # Prendo solo le feature che mi servono
                        mean_stat_home = {key: value for key, value in mean_stat_home.items() if
                                          key in include}
                        mean_stat_away = {key: value for key, value in mean_stat_away.items() if
                                          key in include}

                    # Non voglio che siano tutti zero, basta che ce ne sia almeno uno diverso da 0
                    all_values_mean = list(mean_stat_home.values()) + list(mean_stat_away.values())
                    if any(v != 0 for v in all_values_mean):
                        match_stat.update(
                            {f'{key}_home_stat': adapted_percentage(value) for key, value in
                             mean_stat_home.items()})
                        match_stat.update(
                            {f'{key}_away_stat': adapted_percentage(value) for key, value in
                             mean_stat_away.items()})
            else:
                if not self.predict:
                    s_home = stat_home[stat]
                    s_away = stat_away[stat]
                    if exclude:  # Elimino features che non mi servono
                        s_home = {key: value for key, value in s_home.items() if key not in exclude}
                        s_away = {key: value for key, value in s_away.items() if key not in exclude}

                    # Non voglio che siano tutti zero, basta che ce ne sia almeno uno diverso da 0
                    all_values = list(s_home.values()) + list(s_away.values())
                    if any(v != 0 for v in all_values):
                        match_stat.update(
                            {f'{key}_home_stat': adapted_percentage(value) for key, value in
                             s_home.items()})
                        match_stat.update(
                            {f'{key}_away_stat': adapted_percentage(value) for key, value in
                             s_away.items()})

        return match_stat
