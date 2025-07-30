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
import datetime
import os

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, \
    precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, cross_validate, \
    cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from service_ia.training.grid_search_create import create_gs


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


def fit_process_voting(t='mean', hard=True):
    """
    Con Voting che accumula i modelli e restituisce il migliore
    """
    X, y = split_dataset(t)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    random_forest = RandomForestClassifier(random_state=42, n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
    logistic = LogisticRegression(random_state=42, solver='lbfgs')
    scv = SVC(gamma='scale', random_state=42, probability=True)
    decision = DecisionTreeClassifier()

    # st = StandardScaler()
    # X_train = st.fit_transform(X_train)
    # X_test = st.transform(X_test)

    voting = VotingClassifier(
        estimators=[('rf', random_forest), ('lg', logistic), ('svc', scv), ('decision', decision)],
        voting='hard' if hard else 'soft', verbose=0, n_jobs=-1)
    voting.fit(X_train, y_train)
    y_predict = voting.predict(X_test)

    return f'Voting {t} {'hard' if hard else 'soft'} -> {accuracy_score(y_test, y_predict)}'


def fit_process_bagging_pasting(t='mean', bagging=True):
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
    model = BaggingClassifier(
        estimator=RandomForestClassifier(max_leaf_nodes=16, n_estimators=500, n_jobs=-1, verbose=0, random_state=42),
        n_estimators=500, n_jobs=-1, verbose=1, max_samples=100, bootstrap=bagging, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    return f'BaggingClassifier(RandomForest) {t} {'bagging' if bagging else 'pasting'} -> {accuracy_score(y_test, y_predict)}'


def fit_random_(oob=True, t='mean', predict_feature=False):
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
    if oob:
        model = RandomForestClassifier(max_leaf_nodes=16, n_estimators=500, random_state=42, n_jobs=-1, oob_score=True)
        model.fit(X, y)
        return f'RandomForest {t} oob {model.oob_score_:.3f}'
    else:
        model = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42, n_jobs=-1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_predict)
        return f'RandomForest {t} accuracy : {accuracy}'


def fit_adaboost(t='mean', alg_def=True):
    """
     Sulla base di un Algoritmo, proverà ad addestrare e controllare le stime.
     Se alcune stime saranno errate, si riaddestrerà sulla base di quelle stime cercando di migliorare sui suoi errori.
     - Estimator e n_estimator che sono i numeri di cloni dell'algoritmo.
     - Algorithm con SAMME.R (che è il default) userà le probabilità il che lo rende più robusto a SAMME che usa solo 1 o 0
     Quindi in base a estimator e al n_estimator, lui man mano si riaddestrerà sulla base degli errori precedenti
    """
    X, y = split_dataset(t)
    estimator = RandomForestClassifier(max_depth=1, max_leaf_nodes=16, n_estimators=100, random_state=42, n_jobs=-1)
    algorithm = 'SAMME.R' if alg_def else 'SAMME'  # SAMME.R (default, usa probabilità), SAMME (senza probabilità)
    if algorithm == 'SAMME':
        ada_model = AdaBoostClassifier(estimator=estimator, n_estimators=500, algorithm=algorithm, learning_rate=0.5,
                                       random_state=42)
    else:
        ada_model = AdaBoostClassifier(estimator=estimator, n_estimators=500, learning_rate=0.5, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    ada_model.fit(X_train, y_train)
    y_predict = ada_model.predict(X_test)
    return f'AdaBoostClassifier {t} {algorithm}: {accuracy_score(y_test, y_predict)}'


def fit_gradient_boosting(t='mean'):
    """
    Gradient Boosting è simile ad ADABOOST ma invece che correggere le stime del precedente, lui le riadatta.
    Anche qui troviamo n_estimators che indica quante volte deve ripetere l'operazione per il riadattamento che di
    default usa l'algoritmo di DecisionTreeRegressor(max_depth=3)(anche se è per classificazione poi lui lo riadatta a 0 o 1)
    Quindi se il mio n_estimators ha 200, significa che userà 200 cloni dove ognuno dei quali riadatterà/correggerà
    il precedente.
    """
    X, y = split_dataset(t)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    model = GradientBoostingClassifier(max_depth=2, n_estimators=200, learning_rate=0.5, random_state=42)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return f'Gradient Boosting {t} {accuracy_score(y_test, y_predict)}'


def fit_xgb(t='mean'):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        # colsample_bytree=0.8,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return f'XGB {t} {accuracy_score(y_test, y_predict)}'


def fit_stacking_classifier(t='mean', passthrough=True, stack_method='predict_proba'):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    estimators = [
        ('dt', DecisionTreeClassifier(max_depth=5)),
        ('svc', SVC(probability=True)),
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('knn', KNeighborsClassifier()),
        ('xgb', xgb.XGBClassifier(eval_metric='logloss'))
    ]

    final_estimator = LogisticRegression()

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,  # K-fold è il cross validation
        # verbose=1,
        n_jobs=-1,
        passthrough=passthrough,
        stack_method=stack_method
    )

    stacking.fit(X_train, y_train)
    y_predict = stacking.predict(X_test)
    return f'Stacking Classifier {t} passthrough={passthrough} {stack_method}: {accuracy_score(y_test, y_predict)}'


def cross(estimator, X, y, **kwargs):
    """
    Unisce metodi di cross per valutare gli algoritmi
    | Metodo                | Tipo di output              | Supportato da                        | Uso tipico                 |
    | --------------------- | --------------------------- | ------------------------------------ | -------------------------- |
    | `predict_proba()`     | Probabilità per ogni classe | RandomForest, GradientBoosting, ecc. | ROC, Precision/Recall      |
    | `decision_function()` | Score continuo grezzo       | SVC, LogisticRegression              | ROC, SVM margine           |
    | `predict()`           | Classe predetta             | Tutti                                | Accuracy, Confusion Matrix |

    """
    cv = kwargs.get('cv') or StratifiedKFold(n_splits=5, shuffle=True,
                                             random_state=42)  # Metodo di split per evitare il test_train_split
    threshold = kwargs.get('threshold') or 0.0

    def predict_score():
        """
        Come cross_val_score/cross_validate ma restituisce il predict (in questo caso con metodo 'decision_function')
        Il method però deve essere idoneo con estimator, esempio: RandomForest non ha il 'decision_function' e quindi
        va inserito il 'predict_proba' se si vuole la probabilità ma in questo caso NON utilizzare il method.
        Entrambi, se inseriamo il 'method', restituisce le probabilità che vengono poi gestite dalla soglia
        """
        try:
            return cross_val_predict(estimator, X, y, cv=cv, method='decision_function') > threshold
        except Exception as e:
            # print(str(e))
            return cross_val_predict(estimator, X, y, cv=cv) > threshold

    scoring = [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'roc_auc',
        'average_precision',
        'neg_log_loss',
        'neg_brier_score',
        'f1_macro',
        'f1_weighted',
        'balanced_accuracy'
    ]

    cross_score = cross_val_score(estimator, X, y, cv=cv)  # Restituisce array accuracy_score in base agli split
    cross_val = cross_validate(estimator, X, y, cv=cv,  # return_estimator=True,
                               scoring=scoring)  # Come val_score con return di estimatori
    predict_score = predict_score().astype(int)
    accuracy = accuracy_score(y, predict_score)
    precision = precision_score(y, predict_score)  # Falsi positivi
    recall = recall_score(y, predict_score)  # Veri positivi
    f1 = f1_score(y, predict_score)  # Unisce il recall e la precisione
    cm = confusion_matrix(y, predict_score)  # Un array che va vedere i falsi/veri positivi/negativi per le classi
    precisions, recalls, thresholds = precision_recall_curve(y, predict_score)  # Return stesse descritte prima

    report = classification_report(y, predict_score, output_dict=True)
    results = {
        'estimator': estimator,
        'data': datetime.date.today(),
        'description': kwargs.get('des'),
        # 'fit_process': 'GridSearch',
        'cross_val_score': cross_score,
        'cross_val': cross_val,
        # 'cross_validate_estimator': cross_val['estimator'],
        # 'cross_validate_test_score': cross_val['test_score'],  # restituisce gli stessi valori di cross_val_score
        'cross_val_predict': predict_score,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'precisions': precisions,
        'recall': recall,
        'recalls': recalls,
        'threshold': threshold,
        'thresholds': thresholds,
        'confusion matrix': cm}
    results.update(report)
    save_report(results)
    # TODO
    # sns.heatmap(cm, annot=True)
    # plt.savefig('confusion_matrix.png')
    # RocCurveDisplay.from_estimator(cross_val['estimator'][0], X, y)
    # plt.show()


def search_best_model(estimator, t='base', predict_feature=False, des=''):
    X, y = split_dataset(t)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    gs = create_gs(estimator)
    gs.fit(X_train, y_train)
    cross(estimator=gs.best_estimator_, X=X, y=y, des=f't={t}/predict_feature={predict_feature}/{des}')


def save_report(results_):
    """
    Salva il report dei risultati
    """
    results = {}
    for el in results_.items():
        if el[0] != 'cross_val_predict':
            results.update({el[0]: str(el[1])})
    new_row = pd.DataFrame([results])
    file_path = 'report.xlsx'

    if os.path.exists(file_path):
        # Legge il file esistente e aggiunge la nuova riga
        existing_df = pd.read_excel(file_path)
        updated_df = pd.concat([existing_df, new_row], ignore_index=True)
    else:
        # Crea il DataFrame con la nuova riga
        updated_df = new_row

    # Salva il file aggiornato
    updated_df.to_excel(file_path, index=False)


search_best_model(estimator='svc', t='base', predict_feature=False)

# model = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42, n_jobs=-1)
# # model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
#
# [print(f'{el[0]}\n : ', el[1]) for el in cross(estimator=model).items()]
# [print(f'{el[0]}\n : ', el[1]) for el in cross(estimator=model, t='base').items()]
# [print(f'{el[0]}\n : ', el[1]) for el in cross(estimator=model, t='std').items()]
#
# [print(f'{el[0]}\n : ', el[1]) for el in cross(estimator=model, predict_feature=True).items()]
# [print(f'{el[0]}\n : ', el[1]) for el in cross(estimator=model, t='base', predict_feature=True).items()]
# [print(f'{el[0]}\n : ', el[1]) for el in cross(estimator=model, t='std', predict_feature=True).items()]

# print(fit_stacking_classifier(t='base'))
# print(fit_stacking_classifier(t='base', passthrough=False))
# print(fit_stacking_classifier(t='base', passthrough=False, stack_method='auto'))
# print(fit_stacking_classifier(t='base', stack_method='auto'))
#
# print(fit_stacking_classifier(t='std'))
# print(fit_stacking_classifier(t='std', passthrough=False))
# print(fit_stacking_classifier(t='std', passthrough=False, stack_method='auto'))
# print(fit_stacking_classifier(t='std', stack_method='auto'))

# print(fit_xgb())
# print(fit_xgb(t='std'))

# print(fit_gradient_boosting())
# print(fit_gradient_boosting(t='std'))

# print(fit_adaboost())
# print(fit_adaboost(t='std'))
#
# print(fit_adaboost(alg_def=False))
# print(fit_adaboost(t='std', alg_def=False))

# print(fit_random_(predict_feature=False, t='base'))
# print(fit_random_(predict_feature=False, t='base'))
# print(fit_random_(t='std'))
#
# print(fit_random_(t='base'))
# print(fit_random_(oob=False))
# print(fit_random_(t='std', oob=False))

# print(fit_process_bagging_pasting())
# print(fit_process_bagging_pasting(t='std'))
#
# print(fit_process_bagging_pasting(bagging=False))
# print(fit_process_bagging_pasting(t='std', bagging=False))
#
# print(fit_process_voting())
# print(fit_process_voting(t='std'))
#
# print(fit_process_voting(hard=False))
# print(fit_process_voting(t='std', hard=False))
