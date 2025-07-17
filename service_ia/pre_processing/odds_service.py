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
                    leggere anche "Soccer Player Props API" e "Other soccer betting markets" per cartellini e angoli
        )

Step 3: Aggregherò con il dataset di statistiche (classe totalmente dedicate per l'unione di 2 dataset di servizi diversi: @aggregate_dataset.py)

LAVORARE QUINDI IN MANIERA INDIPEDENTE
"""
# =============================================== STEP 1 ===============================================
import ast
from datetime import datetime, timezone

import pandas as pd
from dateutil.relativedelta import relativedelta

from service_ia.utility.request_api import base_api_odds
from service_ia.utility.utils import convert_csv_to_exel

base_dataset = '../dataset/odds'
name_id_odds_h2h_totals = f'{base_dataset}/id_odds_h2h_totals.csv'  # Dataset ancora "grezzo" dove chiama solo API Odds per recuperare principalmente gli id e poi i primi odds h2h e totals
name_odds_bookmakers = f'{base_dataset}/odds_h2h_totals_bookmakers.csv'  # Dataset conseguenza del precedente dove prende i bookmakers(colonna) che è solo json e crea un dataset solo con quote
name_odds_base = f'{base_dataset}/odds_dataset.csv'  # Dataset che unisce il primo e il secondo precedenti
name_odds_replace = f'{base_dataset}/odds_dataset_replace.csv'  # Dataset che ripulisce il precedente perché ci sono dei duplicati per match e NON per id
markets_base = 'h2h,totals'
markets_events = 'alternate_totals,btts,team_totals'  # TODO ci saranno da inserire gli altri non appena scatterà il nuovo abbonamento :  https://the-odds-api.com/sports-odds-data/betting-markets.html#featured-betting-markets -> qui ci sono tutte:leggere anche "Soccer Player Props API" e "Other soccer betting markets" per cartellini e angoli
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
    {
        'key_sports': 'soccer_italy_serie_a',
        'date_match': dates
    },
    {
        'key_sports': 'soccer_italy_serie_b',
        'date_match': dates
    },
    {
        'key_sports': 'soccer_epl',
        'date_match': dates
    },
    {
        'key_sports': 'soccer_germany_bundesliga',
        'date_match': dates
    },
    {
        'key_sports': 'soccer_spain_la_liga',
        'date_match': dates
    },
    {
        'key_sports': 'soccer_france_ligue_one',
        'date_match': dates
    },
    {
        'key_sports': 'soccer_uefa_champions_league',
        'date_match': dates
    },
    {
        'key_sports': 'soccer_uefa_europa_league',
        'date_match': dates
    },
    {
        'key_sports': 'soccer_uefa_europa_conference_league',
        'date_match': dates
    },
    {
        'key_sports': 'soccer_turkey_super_league',
        'date_match': dates
    }
]


def get_outcomes_event(value, bookmaker):
    mark = bookmaker['markets'][0] if bookmaker['markets'][0]['key'] == value else bookmaker['markets'][
        1] if len(bookmaker['markets']) > 1 else None
    return mark['outcomes'] if mark else []


def search_price_name(outcomes, value):
    if len(outcomes) > 0:
        price = [outcome['price'] for outcome in outcomes if outcome['name'] == value]
        return price[0] if len(price) > 0 else None


def create_odds_dataset():
    """
    Crea il primo dataset con le partite storiche con id e il json dei bookmakers
    :return: id_odds_h2h_totals.csv con le partite storiche dal 2020 a oggi
    """
    for m in dat_match:
        for d in m['date_match']:
            odds_hist = base_api_odds(type_api='hist', path=f'{m['key_sports']}/odds',
                                      params={'regions': 'eu', 'markets': markets_base, 'date': d})
            if odds_hist and len(odds_hist['data']) > 0:
                df = pd.DataFrame(odds_hist['data'])
                df.to_csv(name_id_odds_h2h_totals, mode='a',
                          header=not pd.io.common.file_exists(name_id_odds_h2h_totals),
                          index=False)

    dataset = pd.read_csv(name_id_odds_h2h_totals)
    dataset.drop_duplicates(subset='id', keep='first', inplace=True)
    dataset.to_csv(name_id_odds_h2h_totals)


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
            out_h2h = get_outcomes_event('h2h', bookmaker)
            out_totals = get_outcomes_event('totals', bookmaker)

            # Aggiunta quote H2H
            r[f'home_{title}'] = search_price_name(out_h2h, home_team)
            r[f'away_{title}'] = search_price_name(out_h2h, away_team)
            r[f'draw_{title}'] = search_price_name(out_h2h, 'Draw')

            # Aggiunta quote Totals
            r[f'over_2.5_{title}'] = search_price_name(out_totals, 'Over')
            r[f'under_2.5_{title}'] = search_price_name(out_totals, 'Under')

        flat_rows.append(r)

    flat_df = pd.DataFrame(flat_rows)
    flat_df.to_csv(name_odds_bookmakers, index=False)


def aggregate_odds_bookmakers_base():
    """
    Aggrega i 2 dataset in un unico completo
    :return: @name_odds_base
    """
    odds_dataset = pd.read_csv(name_id_odds_h2h_totals).drop(columns=['bookmakers'], axis=1)
    bookmakers_dataset = pd.read_csv(name_odds_bookmakers)
    merged_dataset = pd.merge(odds_dataset, bookmakers_dataset, on='id', how='inner')
    merged_dataset.to_csv(name_odds_base, index=False)


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
    dataset = pd.read_csv(name_odds_base)

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
    final_dataset.to_csv(name_odds_replace, index=False)
    convert_csv_to_exel(name_odds_replace)


# create_odds_dataset()
# remap_json_bookmakers()
# aggregate_odds_bookmakers_base()
remove_duplicate_match_by_names()


# =============================================== STEP 1 ===============================================

# =============================================== STEP 2 ===============================================
def aggregate_events_into_dataset():
    dataset_odds = pd.read_csv(name_odds_replace)

    for index, row in dataset_odds.iterrows():
        home_team, away_team = row['home_team'], row['away_team']

        events_odds_hist = base_api_odds(type_api='hist', path=f'{row['sport_key']}/events/{row['id']}',
                                         params={'regions': 'eu', 'markets': markets_events,
                                                 'date': row['commence_time']})
        for bookmaker in events_odds_hist['data']['bookmakers']:
            title = bookmaker.get('title')

            # TODO : aggiungere gli altri markets
            out_alternate_totals = get_outcomes_event('alternate_totals', bookmaker)
            dataset_odds.at[index, f'home_{title}'] = search_price_name(out_alternate_totals, home_team)
            dataset_odds.at[index, f'away_{title}'] = search_price_name(out_alternate_totals, away_team)

    # Salva il DataFrame aggiornato
    dataset_odds.to_csv(name_odds_replace, index=False)

# aggregate_events_into_dataset()

# =============================================== STEP 2 ===============================================
