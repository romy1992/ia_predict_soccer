import os

import pandas as pd


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
