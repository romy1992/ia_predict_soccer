"""
In questa classe addestrerò per ensemble :
VotingClassifier
BaggingClassifier
OOB RandomForest
AdaBoostClassifier
GradientBoostingClassifier
XGBoost
StackingClassifier

"""
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.service_ia.training.utility_training.fit_ensemble_classifier import FitUtilityEnsembleClassifier
from src.service_ia.training.utility_training.fit_search_best_model import FitUtilitySearchBestModel
from src.service_ia.training.under_over.inconsistent.fit.filter_service import FilterService


def fit_process_voting():
    """
    Con Voting che accumula i modelli e restituisce il migliore
    """
    random_forest = RandomForestClassifier(random_state=42, n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
    logistic = LogisticRegression(random_state=42, solver='lbfgs')
    scv = SVC(gamma='scale', random_state=42, probability=True)
    decision = DecisionTreeClassifier()

    estimators = [('rf', random_forest), ('lg', logistic), ('svc', scv), ('decision', decision)]

    fit_ensemble_hard = FitUtilityEnsembleClassifier(X=X, y=y, cross_save='cross_save',
                                                     with_smote=with_smote, with_scaler=with_scaler,
                                                     filename=f'{t}_{e}_hard_voting')
    fit_ensemble_hard.fit_voting(estimators=estimators, voting_hs='hard', threshold=threshold, des=des)

    fit_ensemble_soft = FitUtilityEnsembleClassifier(X=X, y=y, cross_save='cross_save',
                                                     with_smote=with_smote, with_scaler=with_scaler,
                                                     filename=f'{t}_{e}_soft_voting')
    fit_ensemble_soft.fit_voting(estimators=estimators, voting_hs='soft', threshold=threshold, des=des)


def fit_process_bagging_pasting():
    """
    | Caratteristica     | Descrizione                                                                                                       |
    | ------------------ | ----------------------------------------------------------------------------------------------------------------- |
    | Base Estimator | Di default usa `DecisionTreeClassifier`, ma puoi usare qualsiasi altro classificatore (es. SVC, KNN, ecc.).       |
    | Bootstrap      | Se `True`, i campioni sono selezionati con ripetizione (tipico del bagging).                                      |
    | n_estimators  | Numero di modelli (cloni) da addestrare. Più modelli = maggiore stabilità.                                        |
    | max_samples   | Percentuale o numero assoluto di campioni da prelevare per ciascun modello.                                       |
    | max_features  | Simile a `max_samples`, ma per le feature (utile per diversificare i modelli).                                    |
    | n_jobs        | Per parallelizzare l’addestramento su più core CPU.                                                               |
    | oobscore     | Se `True`, usa i dati "fuori dal sacco" (out-of-bag) per validazione, utile per valutare senza usare set di test. |

    | Tecnica     | Bootstrap | Descrizione                                                                                               |
    | ----------- | --------- | --------------------------------------------------------------------------------------------------------- |
    | Bagging | `True`    | Estrae **con ripetizione** i campioni (bootstrap sampling). È la forma classica usata nei Random Forest.  |
    | Pasting | `False`   | Estrae **senza ripetizione** i campioni (cioè ogni riga può apparire al massimo una volta in ogni clone). |
    """

    estimator = DecisionTreeClassifier(criterion='entropy', max_depth=5,
                                       max_features='sqrt', min_samples_split=5)

    fit_bagging = FitUtilityEnsembleClassifier(X, y, cross_save='cross_save',
                                               with_smote=with_smote, save_pkl=save_pkl,
                                               filename=f'{t}_{e}_bagging_bagpast',
                                               with_scaler=False)  # Poco senso se l'estimator è un albero
    fit_bagging.fit_bagging_pasting(threshold=threshold, estimator=estimator, bagging=True,
                                    des=f'{des}/stat={list_stats}/bagging=True')

    fit_bagging = FitUtilityEnsembleClassifier(X, y, cross_save='cross_save',
                                               with_smote=with_smote, save_pkl=save_pkl, filename=f'{t}_{e}_bagpast',
                                               with_scaler=False)  # Poco senso se l'estimator è un albero
    fit_bagging.fit_bagging_pasting(threshold=threshold, estimator=estimator, bagging=False,
                                    des=f'{des}/stat={list_stats}/bagging=False')


def fit_random_():
    """
    L'iper parametro oob, indica che il RandomForest può essere addestrato direttamente con X e y senza dover splittare
    in test. Di fatti per poter poi vedere il risultato finale, non c'è bisogno di fare il predict ma di richiamare una
    funzione .oob_score_

    | Parametro           | Controlla...                            |
    | ------------------- | --------------------------------------- |
    | `max_depth`         | La profondità massima dell'albero       |
    | `max_leaf_nodes`    | Il numero massimo di foglie             |
    | `min_samples_leaf`  | Il numero minimo di campioni per foglia |
    | `min_samples_split` | Campioni minimi per dividere un nodo    |

    NON HA SENSO APPLICARE LO SCALING CON RANDOM FOREST PERCHè LUI NON USA LA DISTANZA
    QUINDI NON E' SENSIBILE ALLA SCALA DEI DATI
    """
    if e == 'under_over_1_5':
        params = {'bootstrap': False, 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1,
                  'min_samples_split': 5}
    elif e == 'under_over_2_5':
        params = {'bootstrap': True, 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 4,
                  'min_samples_split': 2}
    else:  # under_over_3_5
        params = None

    if params and 'bootstrap' not in params:
        random_oob = FitUtilityEnsembleClassifier(X, y, cross_save='cross_save',
                                                  save_pkl=save_pkl, filename=f'{t}_{e}_oob_random',
                                                  with_smote=False, with_scaler=False)
        random_oob.fit_random_(oob=True, des=f'{des}/stat={list_stats}/oob=True', override_params=params,
                               calibrated=True)

    random = FitUtilityEnsembleClassifier(X, y, cross_save='cross_save',
                                          save_pkl=save_pkl, filename=f'{t}_{e}_random',
                                          with_smote=with_smote, with_scaler=False)
    random.fit_random_(oob=False, des=f'{des}/stat={list_stats}/oob=False', override_params=params, calibrated=True)


def fit_adaboost():
    """
     Sulla base di un Algoritmo, proverà ad addestrare e controllare le stime.
     Se alcune stime saranno errate, si riaddestrerà sulla base di quelle stime cercando di migliorare sui suoi errori.
     - Estimator e n_estimator che sono i numeri di cloni dell'algoritmo.
     - Algorithm con SAMME.R (che è il default) userà le probabilità il che lo rende più robusto a SAMME che usa solo 1 o 0
     Quindi in base a estimator e al n_estimator, lui man mano si riaddestrerà sulla base degli errori precedenti
    """
    if e == 'under_over_1_5':
        params = None  # Scarsi modelli
    elif e == 'under_over_2_5':
        params = {'n_estimators': 800, 'learning_rate': 0.5, 'algorithm': 'SAMME'}
    else:  # under_over_3_5
        params = None

    if params is None:
        estimator = DecisionTreeClassifier(criterion='log_loss', max_depth=5, max_features='log2',
                                           min_samples_leaf=2, min_samples_split=5)  # TODO per 2.5
        ada_samme = FitUtilityEnsembleClassifier(X=X, y=y, cross_save='cross_save', with_smote=with_smote,
                                                 with_scaler=False, save_pkl=save_pkl,
                                                 filename=f'{t}_{e}_ada_samme')  # Poco senso se l'estimator è un albero
        ada_samme.fit_adaboost(estimator=estimator, alg_def=False, threshold=threshold,
                               des=f'{des}/samme/stat={list_stats}')

        ada_samme_r = FitUtilityEnsembleClassifier(X=X, y=y, cross_save='cross_save', save_pkl=save_pkl,
                                                   with_smote=with_smote, filename=f'{t}_{e}_ada_samme_r',
                                                   with_scaler=False)  # Poco senso se l'estimator è un albero
        ada_samme_r.fit_adaboost(estimator=estimator, threshold=threshold,
                                 des=f'{des}/samme_r/stat={list_stats}')
    else:
        ada = FitUtilityEnsembleClassifier(X=X, y=y, cross_save='cross_save', with_smote=with_smote, save_pkl=save_pkl,
                                           with_scaler=with_scaler, filename=f'{t}_{e}_ada')
        ada.fit_adaboost(estimator=None, threshold=threshold, des=f'{des}/stat={list_stats}',
                         override_params=params)


def fit_gradient_boosting():
    """
    Gradient Boosting è simile ad ADABOOST ma invece che correggere le stime del precedente, lui le riadatta.
    Anche qui troviamo n_estimators che indica quante volte deve ripetere l'operazione per il riadattamento che di
    default usa l'algoritmo di DecisionTreeRegressor(max_depth=3)(anche se è per classificazione poi lui lo riadatta a 0 o 1)
    Quindi se il mio n_estimators ha 200, significa che userà 200 cloni dove ognuno dei quali riadatterà/correggerà
    il precedente.
    """
    gb_fit = FitUtilityEnsembleClassifier(X=X, y=y, cross_save='cross_save', with_smote=with_smote,
                                          save_pkl=save_pkl,
                                          with_scaler=False, filename=f'{t}_{e}_gb')
    if e == 'under_over_1_5':
        params = {'subsample': 0.7, 'n_estimators': 400, 'max_depth': 5, 'learning_rate': 0.05}
    elif e == 'under_over_2_5':
        params = {'subsample': 1.0, 'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05}
    else:  # under_over_3_5
        params = None
    gb_fit.fit_gradient_boosting(des=des, override_params=params)


def fit_xgb():
    """
    XGBClassifier è un'evoluzione del GradientBoosting

    Percentuale di righe del dataset usate per addestrare ciascun albero (stile bagging).
    Di solito tra 0.5 e 1.0 → valore più basso = più regularizzazione.
    subsample=0.8

    Percentuale di colonne (feature) usate da ciascun albero.
    Aiuta a diversificare gli alberi e migliorare la generalizzazione (simile a Random Forest).
    colsample_bytree=0.8

💡 Ci sono varianti:
    colsample_bylevel: per ogni livello dell’albero
    colsample_bynode: per ogni split

    | Parametro          | Descrizione                                     | Tipico Range              |
    | ------------------ | ----------------------------------------------- | ------------------------- |
    | `n_estimators`     | Numero di alberi                                | 100–1000                  |
    | `learning_rate`    | Quanto ogni albero contribuisce alla predizione | 0.01–0.2                  |
    | `max_depth`        | Profondità massima degli alberi                 | 3–10                      |
    | `subsample`        | % di righe usate per ogni albero                | 0.5–1.0                   |
    | `colsample_bytree` | % di colonne usate per ogni albero              | 0.5–1.0                   |
    | `eval_metric`      | Funzione per valutare la performance            | 'logloss', 'error', 'auc' |

    """
    if e == 'under_over_1_5':
        params = {'subsample': 0.5, 'scale_pos_weight': 10, 'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.2,
                  'colsample_bytree': 0.5}
    elif e == 'under_over_2_5':
        params = {'subsample': 0.7, 'scale_pos_weight': 1, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.2,
                  'colsample_bytree': 1.0}
    else:  # under_over_3_5
        params = None

    gb_fit = FitUtilityEnsembleClassifier(X=X, y=y, cross_save='cross_save', with_smote=with_smote, save_pkl=save_pkl,
                                          with_scaler=with_scaler, filename=f'{t}_{e}_xgb')
    gb_fit.fit_xgb(des=des, override_params=params)


def fit_stacking_classifier(passthrough=True, stack_method='predict_proba'):
    """
    Lo stacking classifier combina più algoritmi(level-0) dove le loro previsioni vengono poi passate al modello finale
    (meta-model level-1) che restituisce la predizione finale.
    Input Dati
        ↓
       +--------------+
       | Base Models  |  → esempio: Random Forest, SVM, Logistic
       +--------------+
         ↓     ↓     ↓
    pred1  pred2  pred3   → queste predizioni diventano feature per il livello successivo
            ↓↓↓
       +-----------------+
       | Meta Classifier |  → esempio: Logistic Regression
       +-----------------+
            ↓
    Output Finale


    | Parametro          | Descrizione                                                                    |
    | ------------------ | ------------------------------------------------------------------------------ |
    | `estimators`       | Lista di modelli base con nome                                                 |
    | `final_estimator`  | Modello di livello superiore                                                   |
    | `cv`               | Metodo di cross-validation per generare predizioni affidabili dai modelli base |
    | `stack_method`     | `'auto'`, `'predict_proba'`, `'decision_function'` o `'predict'`               |
    | `passthrough=True` | Se vuoi passare anche le feature originali al meta-model oltre alle predizioni |

    passthrough=Di default: passthrough=False
        Il meta-modello (final_estimator) riceve solo le predizioni dei modelli base (es. predict_proba o predict).
        Non vede le feature originali del dataset.

        Se metti passthrough=True:
        Le feature originali X vengono aggiunte alle predizioni dei modelli base come input per il modello finale.
        Quindi il final_estimator lavora su:
        [predizioni dei base model | feature originali]

    QUINDI:
     - FALSE DA' AL META MODEL FINALE SOLO LE PREVISIONI
     - TRUE DA' AL META MODEL PREVISIONI DEI MODELLI PRECEDENTI + LE ALTRE COLONNE(FEATURE ORIGINALI)

    Attenzione
    Ogni modello base deve avere .fit() e .predict(), e se usi predict_proba anche .predict_proba().
    Richiede più tempo computazionale, soprattutto se cv è alto.
    È facile overfittare se hai pochi dati → regolarizza bene il final_estimator.
    """
    method = 'isotonic'  # 'sigmoid'  # 'isotonic'
    if e == 'under_over_1_5':
        estimators = [
            ('rf', CalibratedClassifierCV(Pipeline(steps=[('smote', SMOTE(random_state=42)),
                                                          ('model',
                                                           RandomForestClassifier(bootstrap=False, max_features='log2',
                                                                                  min_samples_split=5, n_estimators=400,
                                                                                  n_jobs=-1, random_state=42))]),
                                          method=method)),
            ('svc',
             CalibratedClassifierCV(Pipeline(steps=[('scaler', StandardScaler()), ('smote', SMOTE(random_state=42)),
                                                    ('model',
                                                     SVC(C=np.float64(0.01), gamma=1.0, #probability=True,
                                                         random_state=42),)]), method=method)),
            ('xgb', CalibratedClassifierCV(Pipeline(steps=[('smote', SMOTE(random_state=42)),
                                                           ('model', XGBClassifier(**
                                                                                   {'subsample': 0.5,
                                                                                    'scale_pos_weight': 10,
                                                                                    'n_estimators': 200,
                                                                                    'max_depth': 3,
                                                                                    'learning_rate': 0.2,
                                                                                    'colsample_bytree': 0.5}
                                                                                   ))]), method=method)),
            # ('lgbmc', CalibratedClassifierCV(Pipeline(steps=[('smote', SMOTE(random_state=42)),
            #                                                  ('model',
            #                                                   LGBMClassifier(n_estimators=50, n_jobs=-1, num_leaves=100,
            #                                                                  objective='binary', random_state=42,
            #                                                                  scale_pos_weight=10, subsample=0.7),
            #                                                   )]), method=method)),
            # ('gbc', Pipeline(steps=[('smote', SMOTE(random_state=42)),
            #                         ('model',
            #                          GradientBoostingClassifier(learning_rate=0.05, max_depth=5,
            #                                                     n_estimators=400, random_state=42,
            #                                                     subsample=0.7))]))

        ]
        final_estimator = LogisticRegression(C=0.75, class_weight='balanced', max_iter=500)
        # final_estimator = CalibratedClassifierCV(RandomForestClassifier(bootstrap=False, max_features='log2',
        #                                                                 min_samples_split=5, n_estimators=400,
        #                                                                 n_jobs=-1, random_state=42), method=method)
    elif e == 'under_over_2_5':
        estimators = [
            ('rf', Pipeline(steps=[('smote', SMOTE(random_state=42)),
                                   ('model', RandomForestClassifier(max_depth=10, max_features='log2',
                                                                    min_samples_leaf=4, n_estimators=400,
                                                                    n_jobs=-1, random_state=42))])),
            ('svc', Pipeline(steps=[('scaler', StandardScaler()), ('smote', SMOTE(random_state=42)),
                                    ('model', SVC(C=np.float64(0.21544346900318834), gamma=1.0,
                                                  probability=True, random_state=42))])),
            ('ada', Pipeline(steps=[('scaler', StandardScaler()), ('smote', SMOTE(random_state=42)),
                                    ('model', AdaBoostClassifier(algorithm='SAMME', learning_rate=0.5,
                                                                 n_estimators=800, random_state=42))])),
            ('gbc',
             Pipeline(steps=[('smote', SMOTE(random_state=42)), ('model', GradientBoostingClassifier(learning_rate=0.05,
                                                                                                     n_estimators=200,
                                                                                                     random_state=42))]))
        ]
        final_estimator = LogisticRegression(C=np.float64(0.1), max_iter=2000, n_jobs=-1, random_state=42)
    else:
        estimators = None
        final_estimator = None

    f_name = f'{t}_{e}_stacking_{stack_method}_pass={passthrough}' if passthrough else f'{t}_{e}_stacking_{stack_method}'
    staking_fit = FitUtilityEnsembleClassifier(X=X, y=y,
                                               cross_save='cross_save',
                                               # with_smote=with_smote, with_scaler=with_scaler,
                                               save_pkl=save_pkl, filename=f'{f_name}_calibrated_{method}')

    staking_fit.fit_stacking_classifier(stack_method=stack_method, estimators=estimators,
                                        final_estimator=final_estimator, passthrough=passthrough,
                                        des=f'{des}/stack_method={stack_method}/passthrough={passthrough}')


def search_best_model(estimator):
    search = FitUtilitySearchBestModel(X=X, y=y, name=estimator,
                                       best_cross_save='best_cross_save',
                                       with_smote=with_smote, with_scaler=with_scaler,
                                       save_pkl=save_pkl, filename=filename,
                                       keys=keys)

    # search.fit_grid_search(des=des)
    # search.fit_random_search(des=des)
    # search.fit_halving_search(des=des)
    search.fit_halving_random_search(des=des)


# Params base
events = ['under_over_1_5']
ts = ['mean']
list_stats = ['mean_statistics']
threshold = None
with_smote = True
with_scaler = True
save_pkl = True

is_grid = False
estimators_grid = [
    # 'random','svc','lg','decision','nei',
    # 'lgbn', 'xgb', 'gb', 'ada'
]

for e in events:
    for t in ts:
        process_uo = FilterService(event=e, predict=False, list_stats=list_stats)
        X, y, keys = process_uo.split_dataset(type_calculation=t)
        des = f't={t}/event={e}/stat={list_stats}/smote={with_smote}/scaler={with_scaler}'
        if is_grid:
            for est in estimators_grid:
                filename = f'best_grid_{t}_{e}_{est}'
                search_best_model(estimator=est)
        else:
            fit_stacking_classifier(passthrough=False)
            # fit_stacking_classifier()
            # fit_xgb()
            # fit_gradient_boosting()
            # fit_adaboost()
            # fit_random_()
            # fit_process_bagging_pasting()
            # fit_process_voting()

# save_load = SaveLoad(filename='best_mean_under_over_2_5.pkl')
# estimator = save_load.load_model()
# print(estimator)
