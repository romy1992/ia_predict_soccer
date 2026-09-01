import logging

import pandas as pd

logging.basicConfig(level=logging.DEBUG)

# Alcuni modelli come XGBoost, LightGBM, CatBoost gestiscono automaticamente i NaN in input senza bisogno di riempirli.
stat_dataset = pd.read_csv('../../../dataset/statistics/dataset_statistics_history.csv', low_memory=False)
stat_dataset = stat_dataset[stat_dataset['season'] >= 2019]
stat_dataset = stat_dataset.to_dict(orient='records')

odds_dataset = pd.read_csv('../../../dataset/odds/odds_dataset.csv', low_memory=False)
odds_dataset_u_o = odds_dataset[['id'] + [c for c in odds_dataset.columns if 'under' in c or 'over' in c]]
odds_dataset_u_o = odds_dataset_u_o.dropna(axis=1, thresh=6000)

# Over
col_over = [c for c in odds_dataset_u_o.columns if 'over' in c]
odds_dataset_u_o['mean_over_2.5'] = odds_dataset_u_o[col_over].mean(axis=1, skipna=True)
odds_dataset_u_o['std_over_2.5'] = odds_dataset_u_o[col_over].std(axis=1, skipna=True)

# Under
col_under = [c for c in odds_dataset_u_o.columns if 'under' in c]
odds_dataset_u_o['mean_under_2.5'] = odds_dataset_u_o[col_under].mean(axis=1, skipna=True)
odds_dataset_u_o['std_under_2.5'] = odds_dataset_u_o[col_under].std(axis=1, skipna=True)

# odds_dataset_u_o.drop(columns=col_over + col_under, axis=1, inplace=True)
odds_dataset = odds_dataset[['home_team', 'away_team']]
odds_dataset_u_o = pd.concat([odds_dataset, odds_dataset_u_o], axis=1)

for index, odd in enumerate(odds_dataset_u_o.to_dict(orient='records')):
    result = [res for res in stat_dataset if res['match_id_from_odds'] == odd['id']]
    if len(result) > 0:
        logging.info(f'Row {index}')
        result = result[0]
        total_goals = result['score_home_ft'] + result['score_away_ft']
        predict_event = result['predict_under_over']
        odds_dataset_u_o.loc[index, 'predict_2.5'] = 1 if float(predict_event) >= 2.5 else 0.5 if pd.isna(
            predict_event) else 0
        odds_dataset_u_o.loc[index, '2.5'] = 1 if total_goals >= 3 else 0

print(odds_dataset_u_o.head())
print(odds_dataset_u_o.info())
print(odds_dataset_u_o.describe())
# print(odds_dataset.hist())
odds_dataset_u_o = odds_dataset_u_o[odds_dataset_u_o['2.5'].notna()]
odds_dataset_u_o.to_csv("../dataset/u_o_2.5.csv", index=False)
