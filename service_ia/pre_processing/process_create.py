import pandas as pd

# Alcuni modelli come XGBoost, LightGBM, CatBoost gestiscono automaticamente i NaN in input senza bisogno di riempirli.

odds_dataset = pd.read_csv('../dataset/odds/odds_dataset.csv')
odds_dataset = odds_dataset[[c for c in odds_dataset.columns if 'under' in c or 'over' in c]]
odds_dataset = odds_dataset.dropna(axis=1, thresh=6000)

# Over
# col_over = [c for c in odds_dataset.columns if 'over' in c]
# odds_dataset['mean_over_2.5'] = odds_dataset[col_over].mean(axis=1, skipna=True)
#
# # Under
# col_under = [c for c in odds_dataset.columns if 'under' in c]
# odds_dataset['mean_under_2.5'] = odds_dataset[col_under].mean(axis=1, skipna=True)
#
# odds_dataset.drop(columns=col_over + col_under, axis=1, inplace=True)

print(odds_dataset.head())
print(odds_dataset.info())
print(odds_dataset.describe())
# print(odds_dataset.hist())
