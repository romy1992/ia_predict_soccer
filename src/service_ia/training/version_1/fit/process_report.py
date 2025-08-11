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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.service_ia.training.fit_classifier import FitEnsembleClassifier
from src.service_ia.training.fit_search_best_model import FitSearchBestModel


def split_dataset(t='mean', predict_feature=False):
    dataset = pd.read_csv('../dataset/u_o_2.5.csv')
    base_events = ['over_2.5_William Hill', 'under_2.5_William Hill', 'over_2.5_1xBet', 'under_2.5_1xBet',
                   'over_2.5_Unibet', 'under_2.5_Unibet', 'over_2.5_Pinnacle', 'under_2.5_Pinnacle']

    if t == 'mean':
        for value in ['mean_under_2.5', 'mean_over_2.5', '2.5']:
            dataset = dataset[dataset[value].notna()]
        dataset.drop(columns=['id', 'home_team', 'away_team', 'std_over_2.5', 'std_under_2.5'] + base_events,
                     inplace=True)
    elif t == 'std':
        for value in ['std_over_2.5', 'std_under_2.5', '2.5']:
            dataset = dataset[dataset[value].notna()]
        dataset.drop(columns=['id', 'home_team', 'away_team', 'mean_under_2.5', 'mean_over_2.5'] + base_events,
                     inplace=True)
    elif t == 'base':
        for value in ['2.5'] + base_events:
            dataset = dataset[dataset[value].notna()]
        dataset.drop(columns=['id', 'home_team', 'away_team', 'mean_under_2.5', 'mean_over_2.5', 'std_over_2.5',
                              'std_under_2.5'],
                     inplace=True)

    if not predict_feature:
        dataset.drop(columns=['predict_2.5'], inplace=True)

    # View Correlation
    for col in dataset.columns:
        print(dataset.corr()[col].sort_values(ascending=False), '\n')

    X = dataset.drop(columns=['2.5'])
    y = dataset['2.5']

    return X, y


def fit_process_voting(t='mean', predict_feature=False):
    """
    Con Voting che accumula i modelli e restituisce il migliore
    """
    X, y = split_dataset(t, predict_feature)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    random_forest = RandomForestClassifier(random_state=42, n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
    logistic = LogisticRegression(random_state=42, solver='lbfgs')
    scv = SVC(gamma='scale', random_state=42, probability=True)
    decision = DecisionTreeClassifier()

    # st = StandardScaler()
    # X_train = st.fit_transform(X_train)
    # X_test = st.transform(X_test)

    estimators = [('rf', random_forest), ('lg', logistic), ('svc', scv), ('decision', decision)]
    des = f't={t}/predict_feature={predict_feature}'

    fit_ensemble_hard = FitEnsembleClassifier(X=X_train, y=y_train, cross_save='cross_save')
    fit_ensemble_hard.fit_voting(estimators=estimators, voting_hs='hard', threshold=0.0, des=des)

    fit_ensemble_soft = FitEnsembleClassifier(X=X_train, y=y_train, cross_save='cross_save')
    fit_ensemble_soft.fit_voting(estimators=estimators, voting_hs='soft', threshold=0.0, des=des)


def fit_process_bagging_pasting(t='mean', predict_feature=False):
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
    X, y = split_dataset(t)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

    estimator = DecisionTreeClassifier(max_leaf_nodes=16, random_state=42)

    fit_bagging = FitEnsembleClassifier(X_train, y_train, cross_save='cross_save')
    fit_bagging.fit_bagging_pasting(estimator=estimator, bagging=True,
                                    des=f't={t}/predict_feature={predict_feature}/bagging=True')

    fit_bagging = FitEnsembleClassifier(X_train, y_train, cross_save='cross_save')
    fit_bagging.fit_bagging_pasting(estimator=estimator, bagging=False,
                                    des=f't={t}/predict_feature={predict_feature}/bagging=False')


def fit_random_(t='mean', predict_feature=False):
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
    X, y = split_dataset(t, predict_feature)

    random_oob = FitEnsembleClassifier(X, y, cross_save='cross_save')
    random_oob.fit_random_(oob=True, des=f't={t}/predict_feature={predict_feature}/oob=True')

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    random = FitEnsembleClassifier(X_train, y_train, cross_save='cross_save')
    random.fit_random_(oob=False, des=f't={t}/predict_feature={predict_feature}/oob=False')


def fit_adaboost(t='mean', predict_feature=False):
    """
     Sulla base di un Algoritmo, proverà ad addestrare e controllare le stime.
     Se alcune stime saranno errate, si riaddestrerà sulla base di quelle stime cercando di migliorare sui suoi errori.
     - Estimator e n_estimator che sono i numeri di cloni dell'algoritmo.
     - Algorithm con SAMME.R (che è il default) userà le probabilità il che lo rende più robusto a SAMME che usa solo 1 o 0
     Quindi in base a estimator e al n_estimator, lui man mano si riaddestrerà sulla base degli errori precedenti
    """
    X, y = split_dataset(t)
    estimator = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=16, random_state=42)

    ada_samme = FitEnsembleClassifier(X=X, y=y, cross_save='cross_save')
    ada_samme.fit_adaboost(estimator=estimator, alg_def=False, des=f't={t}/predict_feature={predict_feature}/samme')

    ada_samme_r = FitEnsembleClassifier(X=X, y=y, cross_save='cross_save')
    ada_samme_r.fit_adaboost(estimator=estimator, des=f't={t}/predict_feature={predict_feature}/samme_r')


def fit_gradient_boosting(t='mean', predict_feature=False):
    """
    Gradient Boosting è simile ad ADABOOST ma invece che correggere le stime del precedente, lui le riadatta.
    Anche qui troviamo n_estimators che indica quante volte deve ripetere l'operazione per il riadattamento che di
    default usa l'algoritmo di DecisionTreeRegressor(max_depth=3)(anche se è per classificazione poi lui lo riadatta a 0 o 1)
    Quindi se il mio n_estimators ha 200, significa che userà 200 cloni dove ognuno dei quali riadatterà/correggerà
    il precedente.
    """
    X, y = split_dataset(t)
    gb_fit = FitEnsembleClassifier(X=X, y=y, cross_save='cross_save')
    gb_fit.fit_gradient_boosting(des=f't={t}/predict_feature={predict_feature}')


def fit_xgb(t='mean', predict_feature=False):
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
    X, y = split_dataset(t)
    gb_fit = FitEnsembleClassifier(X=X, y=y, cross_save='cross_save')
    gb_fit.fit_xgb(des=f't={t}/predict_feature={predict_feature}')


def fit_stacking_classifier(t='mean', passthrough=True, stack_method='predict_proba', predict_feature=False):
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
    X, y = split_dataset(t)

    estimators = [
        ('dt', DecisionTreeClassifier(max_depth=5)),
        ('svc', SVC(probability=True)),
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('knn', KNeighborsClassifier()),
        ('xgb', xgb.XGBClassifier(eval_metric='logloss'))
    ]

    final_estimator = LogisticRegression()

    staking_fit = FitEnsembleClassifier(X=X, y=y, cross_save='cross_save')
    staking_fit.fit_stacking_classifier(stack_method=stack_method, estimators=estimators,
                                        final_estimator=final_estimator, passthrough=passthrough,
                                        des=f't={t}/predict_feature={predict_feature}/stack_method={stack_method}, passthrough={passthrough}')


def search_best_model(estimator, t='base', predict_feature=False):
    X, y = split_dataset(t)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    search = FitSearchBestModel(X=X_train, y=y_train, name=estimator, best_cross_save='best_cross_save')

    des = f't={t}/predict_feature={predict_feature}'
    # search.fit_grid_search(des=des)
    # search.fit_random_search(des=des)
    search.fit_halving_search(des=des)
    # search.fit_halving_random_search(des=des)


search_best_model(estimator='gb', t='base', predict_feature=False)

# fit_stacking_classifier()
# fit_xgb()
# fit_gradient_boosting()
# fit_adaboost()
# fit_random_()
fit_process_bagging_pasting()

# fit_process_voting()
