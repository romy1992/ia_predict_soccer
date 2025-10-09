"""
In questa classe addestrerò per ensemble :
VotingClassifier
BaggingClassifier
OOB RandomForest
AdaBoostClassifier
GradientBoostingClassifier
XGBoost
StackingClassifier

| Framework                    | Estimatore configurabile? | Tipo di stimatore di default          |
| ---------------------------- | ------------------------- | ------------------------------------- |
| `AdaBoostClassifier`         | ✅ `base_estimator`        | `DecisionTreeClassifier(max_depth=1)` |
| `BaggingClassifier`          | ✅ `base_estimator`        | `DecisionTreeClassifier()`            |
| `GradientBoostingClassifier` | ❌                         | `DecisionTreeRegressor(max_depth=3)`  |


| Modello                    | Usa Bagging? | Usa Pasting? | Bootstrap configurabile?        |
    | -------------------------- | ------------ | ------------ | ------------------------------- |
    | BaggingClassifier      | ✅            | ✅            | ✅ (`bootstrap=True/False`)      |
    | BaggingRegressor       | ✅            | ✅            | ✅                               |
    | RandomForestClassifier | ✅ (fisso)    | ❌            | ✅ (`bootstrap=True` di default) |
    | VotingClassifier       | ❌            | ❌            | ❌                               |
    | GradientBoosting       | ❌            | ❌            | ❌                               |
    | StackingClassifier     | ❌            | ❌            | ❌                               |


Learning_rate : XGBoost, LightGBM e AdaBoost -> per tutti
Il learning_rate controlla quanto ogni nuovo modello contribuisce alla previsione finale.
È un moltiplicatore per l’output del nuovo albero.

predizione_finale += learning_rate * nuovo_albero(x)

    | Valore `learning_rate`   | Effetto                                                  |
    | ------------------------ | -------------------------------------------------------- |
    | **Alto** (`0.5 - 1.0`)   | 💥 Veloce apprendimento ma rischio **overfitting**       |
    | **Basso** (`0.01 - 0.2`) | 🛡️ Apprendimento più lento ma **più preciso e robusto** |
    | **Molto basso**          | 🐢 Richiede molti più alberi (`n_estimators` alto)       |

Regola d’oro
Basso learning_rate + Alto n_estimators = Generalizzazione migliore



REGOLA SCALING :

Attenzione: anche se non serve, lo scaling non danneggia questi modelli. È solo ridondante e fa perdere tempo.

| Modello                                                                               | Motivo                                         |
| ------------------------------------------------------------------------------------- | ---------------------------------------------- |
| **Decision Tree**                                                                     | Lavora per split, non su distanza              |
| **Random Forest**                                                                     | È un insieme di alberi, quindi stesso discorso |
| **Gradient Boosting** (es. `HistGradientBoosting`, `XGBoost`, `LightGBM`, `CatBoost`) | Basato su alberi ⇒ no scaling necessario       |
| **Naive Bayes** (in particolare `CategoricalNB`, `MultinomialNB`)                     | Lavora con frequenze e conteggi, non distanze  |
| **Rule-based Models** (es. RuleFit, Explainable Boosting Machine)                     | Basati su logiche e split                      |



Modelli che richiedono scaling (scaling è fortemente consigliato o obbligatorio) :
| Modello                                   | Motivo                                                                                         |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **SVM (LinearSVC, SVC)**                  | Basato su distanze e margini                                                                   |
| **K-Nearest Neighbors (KNN)**             | Basato su distanza euclidea o simili                                                           |
| **Logistic Regression**                   | Ottimizzazione numerica ⇒ convergenza più rapida e precisa con scaling                         |
| **Linear Regression (OLS, Ridge, Lasso)** | Stessa cosa, scala influenza i coefficienti                                                    |
| **Perceptron / MLP / Reti Neurali**       | Convergenza durante l’ottimizzazione (gradient descent) migliora molto con dati standardizzati |
| **PCA, LDA**                              | Si basano su varianza, correlazione ⇒ le scale influiscono fortemente                          |
"""
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.repository.match_repository import MatchRepository
from src.service_ia.training.fit_classifier import FitEnsembleClassifier
from src.service_ia.training.fit_search_best_model import FitSearchBestModel
from src.service_ia.utility.utils import convert_orm_match_to_dict, adapted_percentage


def get_matches():
    match_repo = MatchRepository()
    filters = {
        'status': ['FT'],
        'mean_statistics': "not None",
        'odds': "not None",
        'statistics': "not None",
        'season': [2020, 2021, 2022, 2023, 2024, 2025]}
    matches = match_repo.search_filter(filters=filters)
    matches_dict = convert_orm_match_to_dict(matches)
    matches_dict = [m for m in matches_dict
                    if (m['odds'] and m['odds'][0][event] and len(m['odds'][0][event]) > 0)
                    and len(m['mean_statistics']) > 0]
    dataset = []
    for match in matches_dict:
        statistics = match['statistics']
        if len(statistics) == 2:
            # Statistiche totali con form ,comparison e predict
            stat_home = statistics[0] if statistics[0]['statistics_team_id'] == match['id_team_home'] else statistics[1]
            stat_away = statistics[1] if statistics[1]['statistics_team_id'] == match['id_team_away'] else statistics[0]

            def switch_under_over():
                """
                In base al get_matches.event, switch l'evento under over 1.5, 2.5 o 3.5
                get_matches.event: 1.5, 2.5 o 3.5
                :return: odds event + y result
                """
                r = None
                value = 2.5 if event == 'under_over_2_5' else 1.5 if event == 'under_over_1_5' else 3.5
                total = stat_home['score_ft'] + stat_away['score_ft']
                match value:
                    case 1.5:
                        r = 1 if total >= 2 else 0
                    case 2.5:
                        r = 1 if total >= 3 else 0
                    case 3.5:
                        r = 1 if total >= 4 else 0
                return match['odds'][0][event], r

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
                    mean_stat_home = mean_stat[0] if mean_stat[0]['id_team'] == match['id_team_home'] else mean_stat[1]
                    mean_stat_away = mean_stat[1] if mean_stat[1]['id_team'] == match['id_team_away'] else mean_stat[0]

                match_stat = {}
                for stat in list_stats:  # Ciclo tutti dict delle feature che mi servono
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
                        s_home = stat_home[stat]
                        s_away = stat_away[stat]
                        if exclude:  # Elimino features che non mi servono
                            s_home = {key: value for key, value in s_home.items() if key not in exclude}
                            s_away = {key: value for key, value in s_away.items() if key not in exclude}

                        # Non voglio che siano tutti zero, basta che ce ne sia almeno uno diverso da 0
                        all_values = list(s_home.values()) + list(s_away.values())
                        if any(v != 0 for v in all_values):
                            match_stat.update(
                                {f'{key}_home_stat': adapted_percentage(value) for key, value in s_home.items()})
                            match_stat.update(
                                {f'{key}_away_stat': adapted_percentage(value) for key, value in s_away.items()})

                return match_stat

            match_records = {}
            odds, result = switch_under_over()
            match_records.update({key: value for key, value in odds.items() if pd.notna(value)})
            match_records.update({'id_fixture': match['id_fixture'], 'y': result})

            if list_stats:  # Se bisogna aggiungere le statistiche
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


def split_dataset():
    """
    Splitta il dataset in X e y
    :param t: tipo di calcolo 'std' o 'mean'
    :return: X, y, keys
    """
    import statistics
    dataset = get_matches()

    def generate_feature(element):
        """
        Genera le feature in base al tipo t (std o mean)
        :param t: tipo di calcolo 'std' o 'mean'
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

        if t == 'std':
            if len(under_itm) >= 2 or len(over_itm) >= 2:
                # Calcolo std
                calculate_under = statistics.stdev(list(under_itm.values()))
                calculate_over = statistics.stdev(list(over_itm.values()))
            else:
                # Skip element if not enough data
                calculate_under = None
                calculate_over = None

        elif t == 'mean':
            # Calcolo la media
            calculate_under = statistics.mean(list(under_itm.values()))
            calculate_over = statistics.mean(list(over_itm.values()))
        else:
            raise Exception

        # Costruisco il record
        return {
            f"{t}_under": calculate_under,
            f"{t}_over": calculate_over,
            **stat_itm,  # aggiunge tutte le statistiche con i loro nomi
            "y": element['y']
        } if calculate_over and calculate_under is not None else {}

    df = [generate_feature(element) for element in dataset]
    df = [d for d in df if len(d) > 0]  # Elimino i record vuoti
    df = pd.DataFrame(df).fillna(0)

    return (
        df.drop(columns=['y']).values,  # X
        df['y'].values,  # y
        df.drop(columns=['y']).columns.tolist())  # keys


def fit_process_voting():
    """
    Con Voting che accumula i modelli e restituisce il migliore
    """
    # X, y, keys = split_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    random_forest = RandomForestClassifier(random_state=42, n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
    logistic = LogisticRegression(random_state=42, solver='lbfgs')
    scv = SVC(gamma='scale', random_state=42, probability=True)
    decision = DecisionTreeClassifier()

    estimators = [('rf', random_forest), ('lg', logistic), ('svc', scv), ('decision', decision)]

    fit_ensemble_hard = FitEnsembleClassifier(X=X_train, y=y_train, cross_save='cross_save', with_smote=with_smote,
                                              with_scaler=with_scaler)
    fit_ensemble_hard.fit_voting(estimators=estimators,
                                 voting_hs='hard',
                                 threshold=threshold, des=des)

    fit_ensemble_soft = FitEnsembleClassifier(X=X_train, y=y_train, cross_save='cross_save', with_smote=with_smote,
                                              with_scaler=with_scaler)
    fit_ensemble_soft.fit_voting(estimators=estimators,
                                 voting_hs='soft',
                                 threshold=threshold,
                                 des=des)


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
    # X, y, keys = split_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

    # estimator = DecisionTreeClassifier(max_leaf_nodes=16, random_state=42)
    estimator = DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features='sqrt', min_samples_split=5)

    fit_bagging = FitEnsembleClassifier(X_train, y_train, cross_save='cross_save', with_smote=with_smote,
                                        with_scaler=with_scaler)
    fit_bagging.fit_bagging_pasting(threshold=threshold, estimator=estimator, bagging=True,
                                    des=f'{des}/stat={list_stats}/bagging=True')

    fit_bagging = FitEnsembleClassifier(X_train, y_train, cross_save='cross_save', with_smote=with_smote,
                                        with_scaler=with_scaler)
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

    """
    # X, y, keys = split_dataset()

    random_oob = FitEnsembleClassifier(X, y, cross_save='cross_save', with_smote=False, with_scaler=with_scaler)
    random_oob.fit_random_(oob=True, des=f'{des}/stat={list_stats}/oob=True')

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    random = FitEnsembleClassifier(X_train, y_train, cross_save='cross_save', with_smote=with_smote,
                                   with_scaler=with_scaler)
    random.fit_random_(oob=False, des=f'{des}/stat={list_stats}/oob=False')


def fit_adaboost():
    """
     Sulla base di un Algoritmo, proverà ad addestrare e controllare le stime.
     Se alcune stime saranno errate, si riaddestrerà sulla base di quelle stime cercando di migliorare sui suoi errori.
     - Estimator e n_estimator che sono i numeri di cloni dell'algoritmo.
     - Algorithm con SAMME.R (che è il default) userà le probabilità il che lo rende più robusto a SAMME che usa solo 1 o 0
     Quindi in base a estimator e al n_estimator, lui man mano si riaddestrerà sulla base degli errori precedenti
    """
    # X, y, keys = split_dataset()

    # estimator = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=16, random_state=42)
    # estimator = DecisionTreeClassifier(criterion='entropy', max_depth=5,
    #                                    # max_features='sqrt', 1.5
    #                                    max_features='log2',  # 3.5
    #                                    min_samples_split=5)
    estimator = DecisionTreeClassifier(criterion='log_loss', max_depth=5, max_features='log2',
                                       min_samples_leaf=2, min_samples_split=5)  # 2.5

    ada_samme = FitEnsembleClassifier(X=X, y=y, cross_save='cross_save', with_smote=with_smote, with_scaler=with_scaler)
    ada_samme.fit_adaboost(estimator=estimator, alg_def=False, threshold=threshold,
                           des=f'{des}/samme/stat={list_stats}')

    ada_samme_r = FitEnsembleClassifier(X=X, y=y, cross_save='cross_save', with_smote=with_smote,
                                        with_scaler=with_scaler)
    ada_samme_r.fit_adaboost(estimator=estimator, threshold=threshold,
                             des=f'{des}/samme_r/stat={list_stats}')


def fit_gradient_boosting():
    """
    Gradient Boosting è simile ad ADABOOST ma invece che correggere le stime del precedente, lui le riadatta.
    Anche qui troviamo n_estimators che indica quante volte deve ripetere l'operazione per il riadattamento che di
    default usa l'algoritmo di DecisionTreeRegressor(max_depth=3)(anche se è per classificazione poi lui lo riadatta a 0 o 1)
    Quindi se il mio n_estimators ha 200, significa che userà 200 cloni dove ognuno dei quali riadatterà/correggerà
    il precedente.
    """
    # X, y, keys = split_dataset()
    gb_fit = FitEnsembleClassifier(X=X, y=y, cross_save='cross_save', with_smote=with_smote, with_scaler=with_scaler)
    gb_fit.fit_gradient_boosting(des=des)


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
    # X, y, keys = split_dataset()
    gb_fit = FitEnsembleClassifier(X=X, y=y, cross_save='cross_save', with_smote=with_smote, with_scaler=with_scaler)
    gb_fit.fit_xgb(des=des)


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
    # X, y, keys = split_dataset()

    # if '1.5' in event:
    # estimators = [
    #     ('dt', DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features='sqrt', min_samples_split=5)),
    #     ('svc', SVC(probability=True)),
    #     ('rf', RandomForestClassifier(bootstrap=False, class_weight='balanced',
    #                                   max_features='log2', n_estimators=4272, warm_start=True)),
    #     ('knn', KNeighborsClassifier(metric='euclidean', n_neighbors=3, p=1, weights='distance')),
    #     ('xgb', xgb.XGBClassifier(
    #         **{'colsample_bytree': 1.0, 'learning_rate': 0.2, 'max_depth': 10, 'n_estimators': 200,
    #            'scale_pos_weight': 2, 'subsample': 0.7}))
    # ]
    # final_estimator = LogisticRegression()

    # --- estimators (come li hai già) ---
    estimators = [
        ('svc_cal', CalibratedClassifierCV(
            estimator=SVC(kernel='rbf', probability=True, C=1.0, gamma='scale'),
            method='sigmoid', cv=3)),
        ('rf', RandomForestClassifier(
            n_estimators=800, max_features='sqrt', bootstrap=True,
            # con SMOTE non serve class_weight, puoi toglierlo:
            # class_weight='balanced',
            n_jobs=-1, random_state=42)),
        ('knn', make_pipeline(StandardScaler(with_mean=False),
                              KNeighborsClassifier(n_neighbors=21, weights='distance', metric='euclidean'))),
        ('xgb', CalibratedClassifierCV(
            estimator=xgb.XGBClassifier(
                n_estimators=400, max_depth=7, learning_rate=0.07, subsample=0.8,
                colsample_bytree=0.9, reg_lambda=1.0, reg_alpha=0.0,
                # con SMOTE rimuovi scale_pos_weight:
                # scale_pos_weight=2,
                tree_method='hist', n_jobs=-1, random_state=42),
            method='isotonic', cv=3)),
        ('sgd_lin', make_pipeline(StandardScaler(with_mean=False),
                                  SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=2000, tol=1e-3, random_state=42)))
    ]

    final_estimator = LogisticRegression(C=0.5, max_iter=2000)  # senza class_weight col SMOTE

    staking_fit = FitEnsembleClassifier(X=X, y=y, cross_save='cross_save', with_smote=with_smote,
                                        with_scaler=with_scaler)
    staking_fit.fit_stacking_classifier(stack_method=stack_method, estimators=estimators,
                                        final_estimator=final_estimator, passthrough=passthrough,
                                        des=f'{des}/stack_method={stack_method}/passthrough={passthrough}')


def search_best_model(estimator):
    # X, y, keys = split_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    search = FitSearchBestModel(X=X_train, y=y_train, name=estimator,
                                best_cross_save='best_cross_save',
                                with_smote=with_smote, with_scaler=with_scaler,
                                keys=keys)

    # search.fit_grid_search(des=des)
    # search.fit_random_search(des=des)
    search.fit_halving_search(des=des)
    # search.fit_halving_random_search(des=des)


# Params base
events = ['under_over_1_5', 'under_over_2_5', 'under_over_3_5']
ts = ['mean', 'std']
is_grid = True
with_smote = True
with_scaler = True
estimators_grid = ['decision']
list_stats = ['mean_statistics']
threshold = 0

for event in events:
    for t in ts:
        X, y, keys = split_dataset()
        des = f't={t}/event={event}/stat={list_stats}/smote={with_smote}/scaler={with_scaler}'
        if is_grid:
            for est in estimators_grid:
                search_best_model(estimator=est)
        else:
            fit_stacking_classifier(passthrough=False)
            fit_stacking_classifier()
            fit_xgb()
            fit_gradient_boosting()
            fit_adaboost()
            fit_random_()
            fit_process_bagging_pasting()
            # fit_process_voting()
