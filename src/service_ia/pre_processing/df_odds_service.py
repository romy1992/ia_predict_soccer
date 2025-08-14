"""
Essendoci un problema di date nelle API di ODDS(non c'è molta organizzazione come quelle di API Sports),
bisogna creare un dataset di soli ODDS+EVENTI e DOPO aggregare con quelle delle statistics.
Leghe che servono:
Serie A
Serie B
Premier
Bundesliga
Liga spagnola
Liga one Francese
Champions
Europa League
Conference
Turchia
Come funzionerà ?
Step 1:
    Per ogni campionato, chiamerò ogni mese dell'anno 3 volte : 1 - 11 - 21 di ogni mese(per il momento) dal 2020 al 2025
    Mi restituirà le partite di quel mese e controllerò se già esistono nel dataset:
    se non ci sono aggiungo altrimenti passa avanti -> Recupero quindi le partite, gli id e le quote basilari ("h2h","totals" = 20 token(10 per evento))
        - 07/07/2025 -> API finite, devo riprendere il mese prossimo dal campionato FRANCESE con data > 2021-08-15T15:00:00Z (Controllare l'ultima riga del file @name_odds_base)

Step 2: Dopo aver recuperato le partite, gli id e le quote basilari, inizierò a prendermi le quote più avanzate
        (Oltre a h2h e totals recuperate già nello step 1, inseriremo anche :
            1 - "alternate_totals": simile a totals ma contiene più under over (es 3.5 ma non vedo 1.5)
            2 - "btts": goal o no goal
            3 - team_totals: under/over per singola squadra
            https://the-odds-api.com/sports-odds-data/betting-markets.html#featured-betting-markets -> qui ci sono tutte:
                    leggere anche "Soccer Player Props API" e "Other soccer betting markets" per cartellini e angoli)

Step 3: Aggregherò con il dataset di statistiche (classe totalmente dedicate per l'unione di 2 dataset di servizi diversi: @aggregate_dataset.py)

LAVORARE QUINDI IN MANIERA INDIPEDENTE
"""
# =============================================== STEP 1 ===============================================
import ast
import json
import logging
import os
from datetime import datetime, timezone, timedelta

import pandas as pd
from dateutil.relativedelta import relativedelta

from src.service_ia.utility.request_api import base_api_odds, base_api_statistics

logging.basicConfig(level=logging.DEBUG)
base_dataset = '../dataset/odds'
name_id_odds_h2h_totals = f'{base_dataset}/id_odds_h2h_totals.csv'  # Dataset ancora "grezzo" dove chiama solo API Odds per recuperare principalmente gli id e poi i primi odds h2h e totals
name_odds_bookmakers = f'{base_dataset}/odds_h2h_totals_bookmakers.csv'  # Dataset conseguenza del precedente dove prende i bookmakers(colonna) che è solo json e crea un dataset solo con quote
name_odds_base = f'{base_dataset}/odds_dataset.csv'  # Dataset che unisce il primo e il secondo precedenti ripuliti per duplicati

BASE_DIR = os.path.dirname(__file__)

with open(os.path.abspath(os.path.join(BASE_DIR, '..', 'json', 'bookmakers.json')), 'r', encoding='utf-8') as file:
    BOOKMAKERS_SPORTS = json.load(file)

with open(os.path.abspath(os.path.join(BASE_DIR, '..', 'json', 'bet.json')), 'r', encoding='utf-8') as file:
    BET_BOOKMAKERS = json.load(file)

regions = 'us,eu'
markets_base = 'h2h,totals'
# TODO ci saranno da inserire gli altri non appena scatterà il nuovo abbonamento :  https://the-odds-api.com/sports-odds-data/betting-markets.html#featured-betting-markets -> qui ci sono tutte:leggere anche "Soccer Player Props API" e "Other soccer betting markets" per cartellini e angoli
markets_events = 'alternate_totals,btts,double_chance,alternate_team_totals,team_totals'  # TODO : alternate_totals_corners,alternate_totals_cards' -> Per poter elaborare una grossa quantità di eventi,meglio tenere da parte angoli e cartellini
# Data iniziale
start_date = datetime(2020, 6, 6, 10, 5, tzinfo=timezone.utc)

# Data finale (oggi, in UTC)
end_date = datetime.now(timezone.utc)

# Giorni del mese desiderati
days = [1, 11, 21]

# Lista per salvare le date
dates = []

# Partiamo dal primo mese che include almeno un giorno >= start_date
current = start_date.replace(day=1)

while current <= end_date:
    for day in days:
        try:
            candidate = current.replace(day=day, hour=10, minute=5)
            if start_date <= candidate <= end_date:
                dates.append(candidate.strftime('%Y-%m-%dT%H:%M:%SZ'))
        except ValueError:
            # Salta giorni non validi, es. 31 aprile
            continue
    # Vai al mese successivo
    current += relativedelta(months=1)

dat_match = [
    # {
    #     'key_sports': 'soccer_italy_serie_a',
    #     'date_match': dates
    # },
    # {
    #     'key_sports': 'soccer_italy_serie_b',
    #     'date_match': dates
    # },
    # {
    #     'key_sports': 'soccer_epl',
    #     'date_match': dates
    # },
    # {
    #     'key_sports': 'soccer_germany_bundesliga',
    #     'date_match': dates
    # },
    # {
    #     'key_sports': 'soccer_spain_la_liga',
    #     'date_match': dates
    # },
    # {
    #     'key_sports': 'soccer_france_ligue_one',
    #     'date_match': dates
    # },
    # {
    #     'key_sports': 'soccer_uefa_champions_league',
    #     'date_match': dates
    # },
    # {
    #     'key_sports': 'soccer_uefa_europa_league',
    #     'date_match': dates
    # },
    # {
    #     'key_sports': 'soccer_uefa_europa_conference_league',
    #     'date_match': dates
    # },
    # {
    #     'key_sports': 'soccer_turkey_super_league',
    #     'date_match': dates
    # },
    # {
    #     'key_sports': 'soccer_netherlands_eredivisie',  # attuale
    #     'date_match': dates
    # },
    # {
    #     'key_sports': 'soccer_dutch_eredivisie',  # vecchia
    #     'date_match': dates
    # },
    # TODO : i prossimi saranno loro
    {
        'key_sports': 'soccer_belgium_first_div',  # attuale e storica
        'date_match': dates
    },
    {
        'key_sports': 'soccer_portugal_primeira_liga',  # attuale e storica
        'date_match': dates
    }
]


def get_outcomes_event(value, bookmaker):
    markets = [market for market in bookmaker['markets'] if market.get('key') == value]
    return markets[0]['outcomes'] if len(markets) > 0 else []


def search_price_name(outcomes, value):
    """
    Cerca e torna il prezzo dell'evento dove si ha la certezza che c'è solo 1 elemento con quel value specificato
    :param outcomes: prezzi
    :param value: nome squadra o evento
    :return: prezzo evento
    """
    if len(outcomes) > 0:
        price = [outcome['price'] for outcome in outcomes if outcome['name'] == value]
        return price[0] if len(price) > 0 else None


def search_price_name_point(outcomes, value, point, des=None):
    """
    Cerca e torna i prezzi dell'evento dove ci sono più value con lo stesso nome e quindi bisogna filtrare con il point
    :param des: se valorizzato, significa che c'è un ulteriore parametro da filtrare es nome squadra
    :param outcomes: prezzi
    :param value: nome squadra o evento
    :param point: ese: 1.5 -> Over o Under (value)
    :return: prezzi evento
    """
    if len(outcomes) > 0:
        if des is None:
            price = [outcome.get('price') for outcome in outcomes if
                     outcome.get('name') == value and outcome.get('point') == point]
            return price[0] if len(price) > 0 else None
        else:
            price = [outcome.get('price') for outcome in outcomes if
                     outcome.get('name') == value and outcome.get('point') == point and (
                             outcome.get('description') and outcome.get('description') == des)]
            return price[0] if len(price) > 0 else None


def create_odds_dataset():
    """
    Crea il primo dataset con le partite storiche con id e il json dei bookmakers
    :return: id_odds_h2h_totals.csv con le partite storiche dal 2020 a oggi
    """
    try:
        for m in dat_match:
            for d in m['date_match']:
                odds_hist = base_api_odds(type_api='hist', path=f'{m['key_sports']}/odds',
                                          params={'regions': regions, 'markets': markets_base,
                                                  'date': d})  # TODO : fino alla data 03/08/2025 , ho utilizzato solo 'eu' come regions.Sarebbe meglio provare anche us per tutti?
                if len(odds_hist) > 0 and len(odds_hist['data']) > 0:
                    df = pd.DataFrame(odds_hist['data'])
                    df.to_csv(name_id_odds_h2h_totals, mode='a',
                              header=not pd.io.common.file_exists(name_id_odds_h2h_totals),
                              index=False)
    except Exception as e:
        logging.error(str(e))
    # dataset = pd.read_csv(name_id_odds_h2h_totals)
    # dataset.drop_duplicates(subset='id', keep='first', inplace=True)
    # dataset.to_csv(name_id_odds_h2h_totals, index=False)


def remap_json_bookmakers():
    """
    Il json dei Bookmakers precedentemente prima creati, verranno "puliti" e inseriti in un altro dataset
    :return: odds_h2h_totals_bookmakers.csv
    """

    df = pd.read_csv(name_id_odds_h2h_totals)

    # Elimina duplicati
    df = df.drop_duplicates(subset='id', keep='first')

    # Rimuove le righe in cui la colonna 'bookmakers' contiene una lista vuota
    df = df[df['bookmakers'].apply(lambda x: str(x).strip() != '[]')]

    flat_rows = []

    names_book = [name['name'] for name in BOOKMAKERS_SPORTS]

    for _, row in df.iterrows():
        raw_json = row['bookmakers']

        home_team, away_team = row['home_team'], row['away_team']

        try:
            bookmakers = ast.literal_eval(raw_json)
        except Exception as e:
            print(f"Errore nel parsing JSON: {e}")
            continue

        r = {f'id': row['id']}

        for bookmaker in bookmakers:
            title = bookmaker.get('title')
            if title in names_book:
                out_h2h = get_outcomes_event('h2h', bookmaker)
                out_totals = get_outcomes_event('totals', bookmaker)

                # Aggiunta quote H2H
                r[f'home_{title}'] = search_price_name(out_h2h, home_team)
                r[f'away_{title}'] = search_price_name(out_h2h, away_team)
                r[f'draw_{title}'] = search_price_name(out_h2h, 'Draw')

                # Aggiunta quote Totals
                r[f'over_2.5_{title}'] = search_price_name(out_totals, 'Over')
                r[f'under_2.5_{title}'] = search_price_name(out_totals, 'Under')

        if len(r) > 1:
            flat_rows.append(r)

    if len(flat_rows) > 0:
        flat_df = pd.DataFrame(flat_rows)
        flat_df.to_csv(name_odds_bookmakers, index=False)


def aggregate_odds_bookmakers_base():
    """
    Aggrega i 2 dataset in un unico completo controllando se quello finale ha già quelle partite
    :return: @name_odds_base
    """
    odds_h2h_totals = pd.read_csv(name_id_odds_h2h_totals).drop(columns=['bookmakers'], axis=1)
    bookmakers_dataset = pd.read_csv(name_odds_bookmakers)
    merged_dataset = pd.merge(odds_h2h_totals, bookmakers_dataset, on='id', how='inner')
    odds_dataset = pd.read_csv(name_odds_base)
    merged_dataset = merged_dataset[~merged_dataset['id'].isin(odds_dataset['id'])]
    odds_dataset = pd.concat([odds_dataset, merged_dataset], axis=0)
    pd.DataFrame(odds_dataset).to_csv(f'{base_dataset}/odds_dataset_copy.csv', index=False)

    # merged_dataset.to_csv(name_odds_base, index=False) TODO : commentato perchè altrimenti sovrascrive quelli già matchati con le statistiche
    # TODO : creare quindi un secondo dataset ,uguali per le prime partite ma diverse per le ultime dove poi aggiungerò a quello originale le partite senza match :
    #       la colonna "id_fixture_from_stat" dell'originale odds_dataset.csv ha tutte le righe valorizzate quindi basta aggiungere sotto quelle che non hanno la colonna
    #       Tutto questo però DOPO aver eseguito anche il metodo successivo "remove_duplicate_match_by_names"


def remove_duplicate_match_by_names():
    """
    Questo metodo nasce con l'esigenza del fatto che nel dataset @name_odds_base ci sono
    match duplicati NON per id del match ma per match completo.
    Quello che bisogna fare per eliminarli è:
        - Per ogni match tra squadra A e squadra B che hanno un duplicato entro una data di 5 giorni(quindi se il match è ripetuto entro un range di 5 giorni)
            bisogna prendere l'ultima data (la più recente) è confrontarla con l'altra.Se ci sono bookmakers mancanti, allora prendo le quote della data più vecchia,
            altrimenti mi tengo la corrente (la più recente)
    :return: @name_odds_base ripulito senza duplicati di MATCH
    """
    # TODO STESSO MOTIVO DEL METODO aggregate_odds_bookmakers_base
    # dataset = pd.read_csv(name_odds_base)
    dataset = pd.read_csv(f'{base_dataset}/odds_dataset_copy.csv')

    dict_dat = []
    for row in dataset.itertuples(index=False):
        name_home = row.home_team
        name_away = row.away_team
        sport_key = row.sport_key

        def check_date():
            target_time = pd.to_datetime(row.commence_time)
            # Trova l'anno di riferimento della stagione
            year = target_time.year if target_time.month >= 7 else target_time.year - 1

            # Definisci l'intervallo della stagione: da agosto a maggio
            start_season = pd.Timestamp(year=year, month=7, day=1, tz='UTC')
            end_season = pd.Timestamp(year=year + 1, month=6, day=30, hour=23, minute=59, second=59, tz='UTC')

            return (pd.to_datetime(dataset['commence_time']) >= start_season) & (
                    pd.to_datetime(dataset['commence_time']) <= end_season)

        row_matches = dataset[(
                (dataset['sport_key'] == sport_key) &
                (dataset['home_team'] == name_home) &
                (dataset['away_team'] == name_away) &
                (check_date())
        )].sort_values(by='commence_time', ascending=True)

        if not row_matches.empty:
            match = row_matches.iloc[- 1].to_dict()
            aggregate_date = ''
            if len(row_matches) > 1:
                row_matches = row_matches.iloc[:-1]
                for row_match in row_matches.itertuples(index=False):
                    comma = ','
                    aggregate_date += f'({row_match.id}{comma}{row_match.commence_time}){comma if len(row_matches) > 1 else ''}'

            match['ids_dates'] = aggregate_date
            dict_dat.append(match)

    final_dataset = pd.DataFrame(dict_dat)
    final_dataset.drop_duplicates(subset='id', keep='last', inplace=True)

    # TODO STESSO MOTIVO DEL METODO aggregate_odds_bookmakers_base
    # final_dataset.to_csv(name_odds_base, index=False)
    final_dataset.to_csv(f'{base_dataset}/odds_dataset_copy.csv', index=False)


def added_odds():
    """
    Aggiunge le quote NON dalla piattaforma ODDS ma da API_SPORT
    Aggiunge al DT altre righe con nomi colonne simili
    :return: new dt odds
    """
    leagues = [135, 136, 140, 78, 39, 94, 203, 2, 3, 848]

    # ( - 0 -> Oggi , - 1 -> Ieri , + 1 -> Domani , + 2 DopoDomani)

    # Formato richiesto è esempio:"2025-02-12"
    format_data = '%Y-%m-%d'
    current_data = datetime.now()

    # Scegliere da che giorno indietro si vuole andare per recuperare le partite
    from_date = (current_data - timedelta(days=3)).strftime(format_data)
    # Fino a ...
    to_date = (current_data - timedelta(days=0)).strftime(format_data)

    # Scegliere i giorni indietro che si vuole andare per recuperare le partite (ESEGUIRA' solo un giorno)
    date = (current_data - timedelta(days=0)).strftime(format_data)
    date_manual = '2025-03-07'

    # FT è partita finita
    # AET è per partita finita ai supplementari (QUINDI PER COPPE)
    # PEN è per partita finita ai rigori (QUINDI PER COPPE)
    status_list = ['FT', 'AET', 'PEN']

    dataset_odds = pd.read_csv(name_odds_base)

    ids_bookmakers = [ids_book['id'] for ids_book in BOOKMAKERS_SPORTS]
    ids_bets = [id_bet['id'] for id_bet in BET_BOOKMAKERS]
    for league in leagues:
        fixtures = base_api_statistics(
            path='fixtures',
            params={
                'from': from_date, 'to': to_date,
                'status': status_list,
                'league': league,
                # 'date': date
            })
        odds_bet = []
        for fixture in fixtures:
            id_fixture = fixture['fixture']['id']
            data_fix = fixture['fixture']['date']
            league_id = fixture['league']['id']
            name_league = fixture['league']['name']
            season = fixture['league']['season']
            round_fixture = fixture['league']['round']
            home_id = fixture['teams']['home']['id']
            home_team = fixture['teams']['home']['name']
            away_id = fixture['teams']['away']['id']
            away_team = fixture['teams']['away']['name']
            fixture_bookmakers = base_api_statistics(path='/odds', params={'fixture': id_fixture})

            if len(fixture_bookmakers) > 0:
                # Inizia a creare il dizionario prima di aggiungere le quote
                bookmakers_filters = [bookmaker for bookmaker in fixture_bookmakers[0]['bookmakers'] if
                                      bookmaker['id'] in ids_bookmakers]

                odd_bet = {
                    'id_fixture_from_stat': id_fixture,
                    'api_from': 'sports-api',
                    'sport_key': league_id,
                    'sport_title': name_league,
                    'commence_time': data_fix,
                    'home_id': home_id,
                    'home_team': home_team,
                    'away_id': away_id,
                    'away_team': away_team,
                    'season': season,
                    'round_fixture': round_fixture,
                }

                for bookmaker in bookmakers_filters:
                    # Crea il dizionario della fixture aggregando tutti gli eventi con le sue quote
                    name_book = bookmaker['name']
                    filter_bet = [bet for bet in bookmaker['bets'] if bet['id'] in ids_bets]
                    for filter_bet_name in filter_bet:
                        for value in filter_bet_name['values']:
                            odd_bet.update({
                                f'{name_book}_{filter_bet_name['name']}_{value['value']}': value['odd']
                            })

                odds_bet.append(odd_bet)

            if len(odds_bet) > 0:
                # Inserisci e modifica il dataset attuale solo se c'è almeno un elemento
                odd_bet_dt = pd.DataFrame(odds_bet)
                concat_odds = pd.concat([dataset_odds, odd_bet_dt], axis=0)
                print(concat_odds)
                # TODO concat_odds.to_csv(name_odds_base, index=False)


def clean_dataset_odds():
    """
    Ripulisce il dataset da colonne di bookmakers in eccesso
    :return:
    """
    list_columns = ['id', 'ids_dates', 'id_fixture_from_stat', 'api_from', 'sport_key', 'sport_title', 'commence_time',
                    'home_team', 'away_team']
    names_book = [name['name'] for name in BOOKMAKERS_SPORTS]
    dataset = pd.read_csv(name_odds_base)
    dataset = dataset[[c for c in dataset.columns if c in list_columns or [el for el in names_book if el in c]]]
    dataset.to_csv(name_odds_base, index=False)


# create_odds_dataset()
# remap_json_bookmakers()
# aggregate_odds_bookmakers_base()
# remove_duplicate_match_by_names()
# convert_csv_to_exel(f'{base_dataset}/odds_dataset_copy.csv')
# added_odds()
# clean_dataset_odds()


# =============================================== STEP 1 ===============================================

# =============================================== STEP 2 ===============================================

def get_alternate_totals(bookmaker, title):
    """
    Alternate totals: Under e Over (inteso come match) di tutte le tipologie
    :param bookmaker: il bookmaker corrente
    :param title: titolo bookmaker
    :return: dict alternat totals
    """
    out_alternate_totals = get_outcomes_event('alternate_totals', bookmaker)

    # 1.5
    over_1_5 = search_price_name_point(out_alternate_totals, 'Over', 1.5)
    under_1_5 = search_price_name_point(out_alternate_totals, 'Under', 1.5)
    # 2.5
    over_2_5 = search_price_name_point(out_alternate_totals, 'Over', 2.5)
    under_2_5 = search_price_name_point(out_alternate_totals, 'Under', 2.5)
    # 3.5
    over_3_5 = search_price_name_point(out_alternate_totals, 'Over', 3.5)
    under_3_5 = search_price_name_point(out_alternate_totals, 'Under', 3.5)
    # 4.5
    over_4_5 = search_price_name_point(out_alternate_totals, 'Over', 4.5)
    under_4_5 = search_price_name_point(out_alternate_totals, 'Under', 4.5)

    return {
        f'alternate_under_1_5_{title}': under_1_5,
        f'alternate_over_1_5_{title}': over_1_5,
        f'alternate_under_2_5_{title}': under_2_5,
        f'alternate_over_2_5_{title}': over_2_5,
        f'alternate_under_3_5_{title}': under_3_5,
        f'alternate_over_3_5_{title}': over_3_5,
        f'alternate_under_4_5_{title}': under_4_5,
        f'alternate_over_4_5_{title}': over_4_5
    }


def get_teams_total(bookmaker, title, name_home, name_away):
    """
    teams_totals e alternate_team_totals : under e over per singola squadra
    :param bookmaker: il bookmaker corrente
    :param title: titolo bookmaker
    :param name_home: nome squadra casa
    :param name_away: nome squadra ospite
    :return: dict di under over
    """
    out_teams_totals = get_outcomes_event('teams_totals', bookmaker)
    out_alternate_team_totals = get_outcomes_event('alternate_team_totals', bookmaker)

    dict_teams_totals = out_teams_totals + out_alternate_team_totals

    under_1_5_home = search_price_name_point(dict_teams_totals, 'Under', 1.5, des=name_home)
    over_1_5_home = search_price_name_point(dict_teams_totals, 'Over', 1.5, des=name_home)
    under_1_5_away = search_price_name_point(dict_teams_totals, 'Under', 1.5, des=name_away)
    over_1_5_away = search_price_name_point(dict_teams_totals, 'Over', 1.5, des=name_away)

    under_2_5_home = search_price_name_point(dict_teams_totals, 'Under', 2.5, des=name_home)
    over_2_5_home = search_price_name_point(dict_teams_totals, 'Over', 2.5, des=name_home)
    under_2_5_away = search_price_name_point(dict_teams_totals, 'Under', 2.5, des=name_away)
    over_2_5_away = search_price_name_point(dict_teams_totals, 'Over', 2.5, des=name_away)

    under_3_5_home = search_price_name_point(dict_teams_totals, 'Under', 3.5, des=name_home)
    over_3_5_home = search_price_name_point(dict_teams_totals, 'Over', 3.5, des=name_home)
    under_3_5_away = search_price_name_point(dict_teams_totals, 'Under', 3.5, des=name_away)
    over_3_5_away = search_price_name_point(dict_teams_totals, 'Over', 3.5, des=name_away)

    under_4_5_home = search_price_name_point(dict_teams_totals, 'Under', 4.5, des=name_home)
    over_4_5_home = search_price_name_point(dict_teams_totals, 'Over', 4.5, des=name_home)
    under_4_5_away = search_price_name_point(dict_teams_totals, 'Under', 4.5, des=name_away)
    over_4_5_away = search_price_name_point(dict_teams_totals, 'Over', 4.5, des=name_away)

    return {
        f'under_1_5_home_{title}': under_1_5_home,
        f'over_1_5_home_{title}': over_1_5_home,
        f'under_1_5_away_{title}': under_1_5_away,
        f'over_1_5_away_{title}': over_1_5_away,

        f'under_2_5_home_{title}': under_2_5_home,
        f'over_2_5_home_{title}': over_2_5_home,
        f'under_2_5_away_{title}': under_2_5_away,
        f'over_2_5_away_{title}': over_2_5_away,

        f'under_3_5_home_{title}': under_3_5_home,
        f'over_3_5_home_{title}': over_3_5_home,
        f'under_3_5_away_{title}': under_3_5_away,
        f'over_3_5_away_{title}': over_3_5_away,

        f'under_4_5_home_{title}': under_4_5_home,
        f'over_4_5_home_{title}': over_4_5_home,
        f'under_4_5_away_{title}': under_4_5_away,
        f'over_4_5_away_{title}': over_4_5_away
    }


def get_btts(bookmaker, title):
    """
    btts : Goal e NoGoal
    :param bookmaker: il bookmaker corrente
    :param title: titolo bookmaker
    :return: dict bbts
    """
    out_alternate_totals = get_outcomes_event('btts', bookmaker)
    goal = search_price_name(out_alternate_totals, 'Yes')
    no_goal = search_price_name(out_alternate_totals, 'No')
    return {
        f'goal_{title}': goal,
        f'no_goal_{title}': no_goal
    }


def get_corners(bookmaker, title):
    """
    Cerca e ritorna direttamente la lista quota dei corners non differenziando il tipo di under/over
    :param bookmaker: il bookmaker corrente
    :param title: titolo bookmaker
    :return: dict totale dei corners
    """
    corners = get_outcomes_event('alternate_totals_corners', bookmaker)
    return {f'corner_{corner.get('name')}_{corner.get('point')}_{title}': corner.get('price') for corner in corners}


def get_cards(bookmaker, title):
    """
    Cerca e ritorna direttamente la lista quota dei cartellini non differenziando il tipo di under/over
    :param bookmaker: il bookmaker corrente
    :param title: titolo bookmaker
    :return: dict totale dei cartellini
    """
    cards = get_outcomes_event('alternate_totals_cards', bookmaker)
    return {f'card_{card.get('name')}_{card.get('point')}_{title}': card.get('price') for card in cards}


def get_double_chance(bookmaker, title, name_home, name_away):
    """
    Recupera le doppie chance
    :param bookmaker: il bookmaker corrente
    :param title: titolo bookmaker
    :param name_home: nome squadra casa
    :param name_away: nome squadra ospite
    :return: dict di doppia chance
    """
    dc_dict = get_outcomes_event('double_chance', bookmaker)
    n_dc_dict = {}
    for dc in dc_dict:
        if name_home in dc['name'] and 'or Draw' in dc['name']:  # 1X
            n_dc_dict.update({f'1X_{title}': dc['price']})
        elif name_away in dc['name'] and 'or Draw' in dc['name']:  # X2
            n_dc_dict.update({f'X2_{title}': dc['price']})
        elif name_home in dc['name'] and name_away in dc['name']:  # 12
            n_dc_dict.update({f'12_{title}': dc['price']})
    return n_dc_dict


def aggregate_events_into_dataset():
    """
    https://the-odds-api.com/liveapi/guides/v4/#usage-quota-costs-8 :
    Recupera eventi più avanzati e sono disponibili dopo il 03/05/2023 alle 05:30:00
    :return: odds_dataset con più quote
    https://the-odds-api.com/sports-odds-data/betting-markets-examples.html

    Mercati comuni
    | `market_key`            | Descrizione                                                                       |
    | ----------------------- | --------------------------------------------------------------------------------- |
    | `h2h`                   | Head‑to‑Head (Moneyline o 1X2) — esito finale (include il pareggio per il calcio) |
    | `spreads`               | Handicap sul punteggio (es. +1.5, –1.5)                                           |
    | `totals`                | Over/Under sul totale dei goal dell’incontro                                      |
    | `outrights`             | Vincente del torneo/competizione (es. Mondiali, campionati, ecc.)                 |
    | `alternate_totals`      | Tutte le varianti Over/Under per il totale goal                                   |
    | `btts`                  | Both Teams To Score — scommessa sì/no che entrambe le squadre segnino             |
    | `team_totals`           | Over/Under sul totale di goal segnati da una singola squadra                      |
    | `alternate_team_totals` | Tutte le varianti di Over/Under per i goal di una squadra                         |

    Mercati solo calcio
    | `market_key`                 | Descrizione                                       |
    | ---------------------------- | ------------------------------------------------- |
    | `player_goal_scorer_anytime` | Giocatore che segna in qualsiasi momento (Yes/No) |
    | `player_first_goal_scorer`   | Giocatore che segna il primo gol                  |
    | `player_last_goal_scorer`    | Giocatore che segna l'ultimo gol                  |
    | `player_to_receive_card`     | Giocatore che riceve un cartellino (Yes/No)       |
    | `player_to_receive_red_card` | Giocatore che riceve un cartellino rosso (Yes/No) |
    | `player_shots_on_target`     | Over/Under sui tiri in porta del giocatore        |
    | `player_shots`               | Over/Under sui tiri totali del giocatore          |
    | `player_assists`             | Over/Under sugli assist del giocatore             |
    | `alternate_spreads_corners`  | Handicap corner (es. +2.5 o –3.5 corner)          |
    | `alternate_totals_corners`   | Totale corner Over/Under                          |
    | `alternate_spreads_cards`    | Handicap cartellini (bookings)                    |
    | `alternate_totals_cards`     | Totale cartellini Over/Under                      |
    | `double_chance`              | Doppia chance (1X, 12, X2)                        |


    | `market_key`                | Regioni disponibili | Note                                                  | Tu: Bookmakers che lo gestiscono      | Chi lo gestisce se i tuoi non lo fanno                                                                                                                 |
    | --------------------------- | ------------------- | ----------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
    | `h2h`                       | us, eu, uk, au      | Esito partita (1X2/moneyline)                         | Pinnacle, William Hill, 1xBet, Unibet | —                                                                                                                                                      |
    | `spreads`                   | us, eu, uk, au      | Handicap sul punteggio                                | Pinnacle, William Hill, 1xBet, Unibet | —                                                                                                                                                      |
    | `totals`                    | us, eu, uk, au      | Over/Under sul totale goal                            | Pinnacle, William Hill, 1xBet, Unibet | —                                                                                                                                                      |
    | `outrights`                 | us, eu, uk, au      | Vincente torneo (futures)                             | Pinnacle, William Hill, 1xBet, Unibet | —                                                                                                                                                      |
    | `alternate_totals`          | us                  | Varianti Over/Under (es. O/U 2.25, 2.75)              | (non confermato) Pinnacle, 1xBet      | principalmente bookmaker USA come DraftKings, BetMGM, FanDuel ([the-odds-api.com][1], [the-odds-api.com][2], [the-odds-api.com][3], [SportsDataIO][4]) |
    | `btts`                      | us                  | “Entrambe le squadre segnano”                         | (possibile) 1xBet                     | bookmaker USA selezionati (es. DraftKings…) ([the-odds-api.com][1])                                                                                    |
    | `team_totals`               | us                  | Over/Under su goal squadra                            | (possibile) Pinnacle, 1xBet           | Solo USA (DraftKings, BetMGM…) ([the-odds-api.com][1])                                                                                                 |
    | `alternate_team_totals`     | us                  | Varianti team O/U (es. +2.5 goal squadra)             | ❌ Nessuno                             | Solo USA (bookmaker USA selezionati) ([the-odds-api.com][1])                                                                                           |
    | Player props (`player_*`)   | us                  | Props sui giocatori (goal, assist, tiri, cartellini…) | ❌ Nessuno                             | Solo bookmaker USA top campionati (EPL, Serie A…) ([the-odds-api.com][1], [the-odds-api.com][5])                                                       |
    | `alternate_spreads_corners` | us                  | Handicap corner                                       | ❌ Nessuno                             | Solo USA bookmaker selezionati ([the-odds-api.com][1])                                                                                                 |
    | `alternate_totals_corners`  | us                  | Totale corner Over/Under                              | ❌ Nessuno                             | Solo USA bookmaker selezionati ([the-odds-api.com][1])                                                                                                 |
    | `alternate_spreads_cards`   | us                  | Handicap cartellini                                   | ❌ Nessuno                             | Solo USA bookmaker selezionati ([the-odds-api.com][1])                                                                                                 |
    | `alternate_totals_cards`    | us                  | Totale cartellini Over/Under                          | ❌ Nessuno                             | Solo USA bookmaker selezionati ([the-odds-api.com][1])                                                                                                 |
    | `double_chance`             | us, eu              | Doppie chance (1X, 12, X2)                            | Pinnacle, William Hill, 1xBet, Unibet | —                                                                                                                                                      |

    [1]: https://the-odds-api.com/sports-odds-data/betting-markets.html?utm_source=chatgpt.com "List of API Betting Markets"
    [2]: https://the-odds-api.com/?utm_source=chatgpt.com "The Odds API: Sports Odds API"
    [3]: https://the-odds-api.com/sports-odds-data/football-odds.html?utm_source=chatgpt.com "Football Odds Data"
    [4]: https://sportsdata.io/live-odds-api?utm_source=chatgpt.com "Odds API | Sports Betting API"
    [5]: https://the-odds-api.com/sports-odds-data/betting-markets-examples.html?utm_source=chatgpt.com "Betting Markets - Examples"


    """
    path_csv = os.path.abspath(os.path.join(BASE_DIR, base_dataset, 'odds_dataset.csv'))
    dataset_odds = pd.read_csv(path_csv, low_memory=False)
    dict_ods = dataset_odds.to_dict(orient='records')
    # TODO : Check per controllare quante ne ha fatte fino ad ora -> 04/08/2025 = 853
    #   dict_ods = [d for d in dataset_odds.to_dict(orient='records') if
    #             pd.to_datetime(d['commence_time']).tz_convert(None) >= pd.Timestamp('2023-05-03') and pd.notna(
    #                 d['event_used'])]
    try:
        for index, row in enumerate(dict_ods):
            logging.info(f'Row {index}/{len(dict_ods)}')
            data_row = row['commence_time']
            if pd.to_datetime(data_row).tz_convert(None) >= pd.Timestamp('2023-05-03') and pd.isna(row['event_used']):
                logging.info(f'Row {index}')
                home_team, away_team = row['home_team'], row['away_team']  # Nomi delle squadre

                """
                    L'id di ricerca può variare a seconda se il match è stato rinviato o meno, quindi:
                    Se l'id principale 'id' non ha la colonna 'ids_dates' valorizzato, allora torna 'id'
                    altrimenti la colonna tuple(ids_dates), userà l'id alternativo per quel match
                    :return: id match
                """
                id_match = row['id'] if pd.isna(row['ids_dates']) else row['ids_dates'][0]

                # Recupera odds storiche
                events_odds_hist = base_api_odds(type_api='hist', path=f'{row['sport_key']}/events/{id_match}/odds',
                                                 params={'regions': regions, 'markets': markets_events,
                                                         'date': data_row})
                # Potenzialmente ci sono 5834 partite * 2 regioni * 7 eventi = 408380 token
                if len(events_odds_hist) > 0 and events_odds_hist['data']:
                    final_row = {}
                    for bookmaker in events_odds_hist['data']['bookmakers']:
                        title = bookmaker.get('title')
                        # Alternate totals: Under e Over (inteso come match) di tutte le tipologie
                        final_row.update(get_alternate_totals(bookmaker, title))
                        # btts : Goal e NoGoal
                        final_row.update(get_btts(bookmaker, title))
                        # teams_totals e alternate_team_totals : under e over per singola squadra
                        final_row.update(get_teams_total(bookmaker, title, home_team, away_team))
                        # alternate_totals_cards: Under e over dei cartellini
                        final_row.update(get_cards(bookmaker, title))
                        # alternate_totals_corners: Under e over dei corners
                        final_row.update(get_corners(bookmaker, title))
                        # double_chance : Doppia Chance 1x x2 12
                        final_row.update(get_double_chance(bookmaker, title, home_team, away_team))
                    # Applica final_row alla riga corrente
                    filter_final_row = {}
                    for col, val in final_row.items():
                        if val:  # Solo se la colonna ha un valore altrimenti lascia stare
                            filter_final_row.update({col: val})
                    # Per eventi futuri, marchio la riga come UTILIZZATA per questa api
                    filter_final_row.update({'event_used': 1})
                    for col, val in filter_final_row.items():
                        dataset_odds.at[index, col] = val
    except Exception as e:
        logging.warning(str(e))
    finally:
        # Salva il DataFrame aggiornato
        dataset_odds.to_csv(f'{base_dataset}/odds_dataset_copy.csv', index=False)
        logging.info(f'{name_odds_base} Aggiornato')

# aggregate_events_into_dataset()
# =============================================== STEP 2 ===============================================
# TODO test per eliminare lo sbaglio con corner e cards e salvare in un dataset di test
# d = pd.read_csv(f'{base_dataset}/odds_dataset.csv')
# columns = [a for a in d.columns if 'corner_' in a and 'or ' in a]
# d.drop(columns=columns, inplace=True)
# d.to_excel(f'{base_dataset}/dataset_events.xlsx', index=False)

# TODO questo qui sotto era per un test, ma magari facciamo uno unit
# with open('../../../tests/service/json_test/odds_api_test.json', 'r', encoding='utf-8') as file:
#     test = json.load(file)
# for b in test['data']['bookmakers']:
#     title = b.get('title')
#     print(get_corners(b, title))
#     print(get_cards(b, title))
#     print(get_double_chance(b, title, test['data']['home_team'], test['data']['away_team']))
