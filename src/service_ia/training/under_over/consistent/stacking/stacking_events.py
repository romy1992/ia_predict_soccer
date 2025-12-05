import ast
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.service_ia.training.under_over.consistent.stacking.hier_classifier.HierOverUnderClassifier import \
    HierOverUnderClassifier, FlatOverUnderClassifier
from src.service_ia.training.under_over.service.filter_uo_service import FilterUOService
from src.service_ia.training.utility_training.fit_search_best_model import FitUtilitySearchBestModel

warnings.filterwarnings('ignore', category=FutureWarning)


class StackingEvents:
    df_odds_multilabel = None
    df_mean_stat_multilabel = None

    @staticmethod
    def generate_datasets():
        """
        Genera i dataset multilabel per odds e mean statistics, li salva in file Excel.
        :return: DataFrame odds multilabel, DataFrame mean statistics multilabel
        """
        service = FilterUOService()
        dataset = service.get_match_multilabel_events()
        df_odds_multilabel = service.split_dataset_multilabel(dataset=dataset)
        cols = ['mean_under_1_5', 'mean_over_1_5',
                'mean_under_3_5', 'mean_over_3_5']
        df_odds_multilabel = df_odds_multilabel[~(df_odds_multilabel[cols] == 0).any(axis=1)]
        df_odds_multilabel.to_excel('odds_multilabel.xlsx', index=False)

        service = FilterUOService(list_stats=['mean_statistics'])
        dataset = service.get_matches_only_mean_stat(multilabel=True)
        df_mean_stat_multilabel = pd.DataFrame(dataset)
        cols = ['mean_Total Shots_home_stat', 'mean_Blocked Shots_home_stat',
                'mean_Shots insidebox_home_stat', 'mean_Shots outsidebox_home_stat',
                # 'mean_expected_goals_home_stat', 'mean_goals_prevented_home_stat'
                ]
        df_mean_stat_multilabel = df_mean_stat_multilabel[~(df_mean_stat_multilabel[cols] == 0).any(axis=1)]
        df_mean_stat_multilabel.to_excel('mean_stat_multilabel.xlsx',
                                         index=False)

        return df_odds_multilabel, df_mean_stat_multilabel

    ############################ STACKING ODDS ############################
    @staticmethod
    def get_dataset_odds():
        df_odds = pd.read_excel('odds_multilabel.xlsx')
        odds_15 = df_odds[['mean_over_1_5', 'mean_under_1_5']].values
        odds_25 = df_odds[['mean_over_2_5', 'mean_under_2_5']].values
        odds_35 = df_odds[['mean_over_3_5', 'mean_under_3_5']].values
        y_odds = df_odds[['y']].values

        # ora è 1D: array(['[0, 0, 0]', ...], dtype=object)
        strings = np.ravel(y_odds)
        # 2) converti ogni stringa in lista di int
        rows = [ast.literal_eval(s) for s in strings]  # qui 's' è una stringa, non un array
        # 3) impacchetta in una matrice numerica
        y_odds = np.array(rows, dtype=int)

        return [odds_15, odds_25, odds_35], y_odds

    def stacking_odds(self):
        """
        Esegue lo stacking dei modelli HierOverUnderClassifier sui dataset multilabel.
        :return:
        """
        Xs, y_odds = self.get_dataset_odds()

        method = 'sigmoid'

        base1 = Pipeline(steps=[('smote', SMOTE(random_state=42)),
                                ('model', LGBMClassifier(learning_rate=0.01, max_depth=5,
                                                         n_estimators=200, n_jobs=-1,
                                                         num_leaves=100,
                                                         objective='binary', random_state=42,
                                                         scale_pos_weight=2))])

        base2 = Pipeline(steps=[('smote', SMOTE(random_state=42)),
                                ('model',
                                 LGBMClassifier(colsample_bytree=0.5, learning_rate=0.01,
                                                max_depth=5, n_estimators=200, n_jobs=-1,
                                                num_leaves=100, objective='binary',
                                                random_state=42, scale_pos_weight=1,
                                                subsample=0.7))])

        base3 = Pipeline(steps=[('scaler', StandardScaler()), ('smote', SMOTE(random_state=42)),
                                ('model', SVC(C=np.float64(0.01), gamma=0.01, probability=True,
                                              random_state=42))])

        clf = HierOverUnderClassifier(
            base_1=base1, base_2=base2, base_3=base3,
            calibrator="sigmoid",
            thresholds=(0.54, 0.52, 0.45)
        )
        clf.cross_hierarchical(clf, y=y_odds, X=[Xs[0], Xs[1], Xs[2]], stratify_on=1)

        # clf.fit(X=[Xs[0], Xs[1], Xs[2]], y=y_odds)
        # clf.save('artifacts/mean_odds_stacking.joblib')

    def search_best_models_odds(self):
        Xs, y_odds = self.get_dataset_odds()
        keys_col = ['1_5', '2_5', '3_5']
        names = [  # 'random', 'svc',
            'lg', 'decision', 'nei',
            # 'lgbn', 'xgb', 'gb', 'ada'
        ]

        for i in range(len(Xs)):
            for name in names:
                print(f"Dataset Xs[{i}] shape: {Xs[i].shape} with estimator: {name}")
                search = FitUtilitySearchBestModel(X=Xs[i], y=y_odds[:, i], name=name,
                                                   best_cross_save='best_cross_save',
                                                   with_smote=True, with_scaler=True,
                                                   filename=f'best_grid_mean_odds_{keys_col[i]}_{name}',
                                                   keys=keys_col[i])
                search.fit_halving_random_search(des=f'mean_odds_smote_scaler_rf_{keys_col[i]}_{name}')

    ############################ STACKING ODDS ############################

    ############################# STACKING MEAN STAT ############################
    @staticmethod
    def get_dataset_mean_stat():
        df_mean_stat = pd.read_excel('mean_stat_multilabel.xlsx')
        df_mean_stat = df_mean_stat.drop(
            columns=['id_fixture', 'name_home', 'name_away', 'id_team_home_stat', 'id_team_away_stat'])
        # df mean stat
        selected_features = [
            "mean_Total Shots_home_stat",
            "mean_Corner Kicks_home_stat",
            "mean_Goalkeeper Saves_home_stat",
            "mean_Passes accurate_home_stat",
            "mean_Total Shots_away_stat",
            "mean_Corner Kicks_away_stat",
            "mean_Goalkeeper Saves_away_stat",
            "mean_Passes accurate_away_stat"
        ]
        df_mean = df_mean_stat[selected_features].values
        y_mean = np.ravel(df_mean_stat[['y']].values)
        y_mean = np.array([ast.literal_eval(s) for s in y_mean], dtype=int)

        return df_mean, y_mean

    def stacking_mean_stat(self):
        df_mean, y_mean = self.get_dataset_mean_stat()
        # SMOTE meno aggressivo: non porta la minoranza al 50%, ma al 60–70% della maggioranza
        _SMOTE_LIGHT = SMOTE(
            # sampling_strategy=0.6,   # <--- prova 0.5 ~ 0.8
            # k_neighbors=3,           # <--- meno vicini = meno “cloni” rischiosi
            random_state=42
        )

        method = 'sigmoid'

        base1 = Pipeline(steps=[('smote', SMOTE(random_state=42)),
                                ('model',
                                 XGBClassifier(**{'subsample': 1.0, 'scale_pos_weight': 2,
                                                  'n_estimators': 200, 'max_depth': 3,
                                                  'learning_rate': 0.1,
                                                  'colsample_bytree': 0.7}))])

        base2 = Pipeline(steps=[('scaler', StandardScaler()), ('smote', SMOTE(random_state=42)),
                                ('model',
                                 CalibratedClassifierCV(SVC(C=np.float64(0.01), probability=True, random_state=42),
                                                        method=method))])

        base3 = Pipeline(steps=[('smote', SMOTE(random_state=42)),
                                ('model',
                                 CalibratedClassifierCV(RandomForestClassifier(bootstrap=False, max_features='log2',
                                                                               min_samples_leaf=2, min_samples_split=5,
                                                                               n_estimators=400, n_jobs=-1,
                                                                               random_state=42), method=method))])

        flat = FlatOverUnderClassifier(
            base_1=base1, base_2=base2, base_3=base3,
            calibrator="sigmoid",
            thresholds=(0.57, 0.55, 0.40)
        )

        flat.cross_hierarchical(flat, X=df_mean, y=y_mean, stratify_on=1)

        flat.fit(X=df_mean, y=y_mean)
        flat.save('artifacts/mean_stat_stacking.joblib')


def search_best_models_stat(self):
    df_mean, y_mean = self.get_dataset_mean_stat()
    keys_col = ['1_5', '2_5', '3_5']
    names = [  # 'random', 'svc',
        # 'lg', 'decision', 'nei',
        'lgbn', 'xgb', 'gb', 'ada'
    ]

    for i in range(y_mean.shape[1]):
        for name in names:
            print(f"Dataset Xs[{i}] shape: {df_mean.shape} with estimator: {name}")
            search = FitUtilitySearchBestModel(X=df_mean, y=y_mean[:, i], name=name,
                                               best_cross_save='best_cross_save',
                                               with_smote=True, with_scaler=True,
                                               filename=f'best_grid_mean_stat_{keys_col[i]}_{name}',
                                               keys=keys_col[i])
            search.fit_halving_random_search(des=f'mean_stat_smote_scaler_{keys_col[i]}_{name}')


############################# STACKING MEAN STAT ############################


stacking_event = StackingEvents()
# mean_odds, mean_stat = stacking_event.generate_datasets()
# stacking_event.stacking_mean_stat()
stacking_event.stacking_odds()

# SMOTE meno aggressivo: non porta la minoranza al 50%, ma al 60–70% della maggioranza
# _SMOTE_LIGHT = SMOTE(
# sampling_strategy=0.6,   # <--- prova 0.5 ~ 0.8
# k_neighbors=3,           # <--- meno vicini = meno “cloni” rischiosi
#     random_state=42
# )
# clf = HierOverUnderClassifier(
#     base_1=base1, base_2=base2, base_3=base1,
#     # calibrator="sigmoid",
#     # thresholds=(0.52, 0.50, 0.48)  # da ottimizzare
# )
#
# idx = np.arange(len(y))
# idx_tr, idx_te = train_test_split(idx, test_size=0.2, stratify=y[:, 1], random_state=42)
#
# X_train = [X[0][idx_tr], X[1][idx_tr], X[2][idx_tr]]
# X_test = [X[0][idx_te], X[1][idx_te], X[2][idx_te]]
# y_train, y_test = y[idx_tr], y[idx_te]
#
# clf.fit_and_select_thresholds(X_train, y_train)
# results = clf.evaluate(X_test, y_true=y_test)
#
# print("=== METRICHE MACRO ===")
# for k, v in results["macro"].items():
#     print(f"{k}: {v:.3f}")
#
# print("\n=== METRICHE PER TARGET ===")
# for target, vals in results["per_target"].items():
#     print(target, vals)
#
# print("\n=== METRICHE PER CONFUSION MATRIX ===")
# for target, vals in results["confusion_matrices"].items():
#     print(target, vals)
#
# print("\n=== METRICHE PER CLASSIFICATION REPORTS ===")
# for target, vals in results["classification_reports"].items():
#     print(target, vals)
#
# # salva tutto (modelli + soglie) in un unico file.joblib
# clf.save("artifacts/hier_ou_v1.joblib")
