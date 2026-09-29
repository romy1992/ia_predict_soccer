import os
import json
import pandas as pd
import unicodedata
from rapidfuzz import fuzz

with open('../json/black_name.json', 'r', encoding='utf-8') as file:
    black_names = json.load(file)


def convert_excel_to_csv(path_file):
    """
    Converte file excel in csv
    :param path_file: percorso file
    :return: nuovo csv
    """
    dataset = pd.read_excel(path_file)
    csv_file = os.path.splitext(path_file)[0] + '.csv'
    dataset.to_csv(csv_file, index=False)


def convert_csv_to_exel(path_file):
    """
    Converte file csv in excel
    :param path_file: percorso file
    :return: nuovo excel
    """
    dataset = pd.read_csv(path_file, low_memory=False)
    excel_file = os.path.splitext(path_file)[0] + '.xlsx'
    dataset.to_excel(excel_file, index=False)


def count_row_not_na(df, value):
    return df[value].notna().sum()


def count_row_is_na(df, value):
    return df[value].isna().sum()


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
        name = name.replace(r, "").strip()

    black_name = [n for n in black_names if n['original'] == name]
    if len(black_name) > 0:
        name = black_name[0]['replace']

    return " ".join(name.split()).strip()


# print(max(fuzz.ratio(normalize_('Pordenone'), normalize_('Frosinone')),
#           fuzz.partial_ratio(normalize_('Pordenone'), normalize_('Frosinone'))))


def check_name(**kwargs):
    """
    Cerca la corrispondenza dei nomi simili tra squadra di fuori casa e casa
    :return: Booleano se il match è stato trovato secondo i nomi delle squadre
    """
    stat = kwargs.get('stat')
    home_odds = kwargs.get('home_odds')
    away_odds = kwargs.get('away_odds')
    soglia = 87

    def fuzzy_score(a, b):
        a_norm, b_norm = normalize_(a), normalize_(b)
        return max(fuzz.ratio(a_norm, b_norm), fuzz.partial_ratio(a_norm, b_norm))

    score_home_andata = fuzzy_score(stat['team_name'], home_odds)
    score_away_andata = fuzzy_score(stat['opponent_name'], away_odds)
    score_home_ritorno = fuzzy_score(stat['team_name'], away_odds)
    score_away_ritorno = fuzzy_score(stat['opponent_name'], home_odds)

    return ((score_home_andata >= soglia) and (score_away_andata >= soglia)) or (
                (score_home_ritorno >= soglia) and (score_away_ritorno >= soglia))
