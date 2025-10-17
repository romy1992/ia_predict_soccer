import pandas as pd
from sklearn.utils.validation import check_is_fitted

from src.service_ia.training.utility_training.save_load import SaveLoad
from src.service_ia.training.version_1.fit.filter_service import FilterService


class PredictUO:
    def __init__(self, model, X):
        self.model = model
        check_is_fitted(model)
        self.X = X

    def predict(self):
        return self.model.predict(self.X)

    def predict_proba(self):
        return self.model.predict_proba(self.X)


m_name = 'mean_under_over_1_5_random_calibrated_isotonic'
model_load = SaveLoad(filename=f"{m_name}.pkl")
model_load = model_load.load_model()
filters = {
    'status': ['NS'],
    'mean_statistics': "not None",
    'season': [2025]}
process_report_uo = FilterService(event='under_over_1_5', predict=True, filters=filters, list_stats=['mean_statistics'])
id_fixtures, features = process_report_uo.split_dataset(type_calculation='mean')

uo = PredictUO(model=model_load, X=features)
match_teams_df = pd.DataFrame(process_report_uo.get_match_teams(), columns=['id_fixture', 'home_team', 'away_team'])
predictions_df = pd.DataFrame({
    'id_fixture': id_fixtures,
    'predictions': uo.predict()
})

# Unisci i DataFrame sull'id_fixture
result_df = pd.merge(match_teams_df, predictions_df, on='id_fixture')

# Se vuoi aggiungere anche le probabilità
proba_df = pd.DataFrame(uo.predict_proba(), columns=['proba_0', 'proba_1'])
proba_df['id_fixture'] = id_fixtures
result_df = pd.merge(result_df, proba_df, on='id_fixture')
result_df.to_excel(f'results_{m_name}.xlsx', index=False)
