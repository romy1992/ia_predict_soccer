import pandas as pd

from src.repository.match_repository import MatchRepository
from src.service_ia.utility.utils import convert_orm_match_to_dict, adapted_percentage


class FilterService:
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

    def __init__(self, event, predict=False, filters=None, **kwargs):
        """
        Classe per processare il report delle partite e generare il dataset per l'addestramento
        :param event: 'under_over_1_5', 'under_over_2_5', 'under_over_3_5'
        :param list_stats: lista di nodi dict es: ['form','passes','mean_statistics']
        """
        self.event = event
        self.predict = predict
        self.list_stats = kwargs.get('list_stats', [])
        # Repository per le partite
        self.match_repo = MatchRepository()
        # Filtro per prendere solo le partite finite con statistiche e quote
        self.filters = {
            'status': ['FT'],
            'mean_statistics': "not None",
            'odds': "not None",
            'statistics': "not None",
            'season': [2020, 2021, 2022, 2023, 2024, 2025]} if filters is None else filters
        # Recupero tutte le partite che rispettano il filtro
        self.matches_dict = self.filter_matches(filters=self.filters)

    def filter_matches(self, filters=None):
        """
        Filtra le partite in base ai filtri passati
        :param filters: filtri da applicare
        :return: lista di partite filtrate
        """
        matches = self.match_repo.search_filter(filters=filters)
        matches_dict = convert_orm_match_to_dict(matches)
        return [m for m in matches_dict
                if (m['odds'] and m['odds'][0][self.event] and len(m['odds'][0][self.event]) > 0)
                and len(m['mean_statistics']) > 0]

    def get_matches(self):
        """
        Genera il dataset in base alle partite filtrate
        :param: predict: se True, genera il dataset per la predizione (senza y)
        :return: dataset
        """
        dataset = []
        for match in self.matches_dict:
            statistics = match['statistics']
            if len(statistics) == 2 or self.predict:
                if not self.predict:
                    # Statistiche totali con form ,comparison e predict
                    stat_home = statistics[0] if statistics[0]['statistics_team_id'] == match['id_team_home'] else \
                        statistics[1]
                    stat_away = statistics[1] if statistics[1]['statistics_team_id'] == match['id_team_away'] else \
                        statistics[0]

                def switch_under_over():
                    """
                    In base al get_matches.event, switch l'evento under over 1.5, 2.5 o 3.5
                    get_matches.event: 1.5, 2.5 o 3.5
                    :return: odds event + y result
                    """
                    r = None
                    value = 2.5 if self.event == 'under_over_2_5' else 1.5 if self.event == 'under_over_1_5' else 3.5
                    total = stat_home['score_ft'] + stat_away['score_ft']
                    match value:
                        case 1.5:
                            r = 1 if total >= 2 else 0
                        case 2.5:
                            r = 1 if total >= 3 else 0
                        case 3.5:
                            r = 1 if total >= 4 else 0
                    return match['odds'][0][self.event], r

                def get_specific_stat(exclude=None, include=None):
                    """
                    Aggrega tutte le statistiche che mi servono per addestrare
                    :param list_stats: lista di nodi dict es: ['form','passes','mean_statistics']
                    :param exclude: casi in cui serve la lista di elementi dict che non servono es: ['Total passes']
                    :param include: casi in cui serve la lista di elementi dict che servono es: ['expected_goal']
                    :return: dict di statistiche di home e away
                    """
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

                match_records = {'id_fixture': match['id_fixture']}
                if self.predict:  # Se è per predire non metto la y
                    odds = match['odds'][0][self.event]
                else:  # Se è per addestrare metto la y
                    odds, result = switch_under_over()
                    match_records.update({'id_fixture': match['id_fixture'], 'y': result})

                match_records.update({key: value for key, value in odds.items() if pd.notna(value)})

                if self.list_stats:  # Se bisogna aggiungere le statistiche
                    stats = get_specific_stat(
                        # exclude=['id_team',
                        #          'mean_Shots on Goal', 'mean_Shots off Goal', 'mean_Blocked Shots', 'mean_Shots insidebox',
                        #          'mean_Shots outsidebox', 'mean_Corner Kicks',
                        #          'mean_Fouls',
                        #          'mean_Offsides',
                        #          'mean_Yellow Cards', 'mean_Red Cards',
                        #          'mean_Goalkeeper Saves',
                        #          'mean_Total passes', 'mean_Passes %', 'mean_Passes accurate'],
                        include=['mean_expected_goals']

                    )
                    # Se ha statistiche con valore !=0
                    if len(stats) > 0:
                        match_records.update(stats)
                        dataset.append(match_records)
                else:
                    dataset.append(match_records)

        return dataset

    def split_dataset(self, type_calculation='mean'):
        """
        Splitta il dataset in X e y
        :param type_calculation: tipo di calcolo 'std' o 'mean'
        :return: X, y, keys
        """
        import statistics
        dataset = self.get_matches()

        def generate_feature(element):
            """
            Genera le feature in base al tipo t (std o mean)
            :param element: elemento del dataset
            :return: record con le feature calcolate
            """
            # Recupero tutte le quote tranne il result label e le statistiche
            under_itm = {k: float(v) if pd.notna(v) else 0 for k, v in element.items() if
                         'under' in k and 'stat' not in k}
            over_itm = {k: float(v) if pd.notna(v) else 0 for k, v in element.items() if
                        'over' in k and 'stat' not in k}
            # Statistiche
            stat_itm = {k: float(v) if pd.notna(v) else 0 for k, v in element.items() if '_stat' in k}

            if type_calculation == 'std':
                if len(under_itm) >= 2 or len(over_itm) >= 2:
                    # Calcolo std
                    calculate_under = statistics.stdev(list(under_itm.values()))
                    calculate_over = statistics.stdev(list(over_itm.values()))
                else:
                    # Skip element if not enough data
                    calculate_under = None
                    calculate_over = None

            elif type_calculation == 'mean':
                # Calcolo la media
                calculate_under = statistics.mean(list(under_itm.values()))
                calculate_over = statistics.mean(list(over_itm.values()))
            else:
                raise Exception

            # Costruisco il record

            if self.predict:
                return {
                    'id_fixture': element['id_fixture'],
                    f"{type_calculation}_under": calculate_under,
                    f"{type_calculation}_over": calculate_over,
                    **stat_itm,  # aggiunge tutte le statistiche con i loro nomi
                } if calculate_over and calculate_under is not None else {}

            return {
                f"{type_calculation}_under": calculate_under,
                f"{type_calculation}_over": calculate_over,
                **stat_itm,  # aggiunge tutte le statistiche con i loro nomi
                "y": element['y']
            } if calculate_over and calculate_under is not None else {}

        df = [generate_feature(element) for element in dataset]
        df = [d for d in df if len(d) > 0]  # Elimino i record vuoti
        df = pd.DataFrame(df).fillna(0)

        if self.predict:
            return df['id_fixture'].values, df.drop(columns=['id_fixture']).values

        return (
            df.drop(columns=['y']).values,  # X
            df['y'].values,  # y
            df.drop(columns=['y']).columns.tolist())  # keys

    def get_match_teams(self):
        """
        :return: lista di tuple (id_fixture, name_home, name_away)
        """
        return [(m['id_fixture'], m['name_home'], m['name_away']) for m in self.matches_dict]
