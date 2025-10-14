import logging

import numpy as np
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.experimental import enable_halving_search_cv as enable  # obbligatorio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, RandomizedSearchCV,
                                     HalvingGridSearchCV, HalvingRandomSearchCV)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.service_ia.training.fit_utility import FitUtility

logging.basicConfig(level=logging.DEBUG)
logging.info(enable)

# Support Vector Machine (SVM)
"""
Sintassi base
np.logspace(start, stop, num)
start → esponente iniziale (base 10)
stop → esponente finale
num → quanti valori vuoi generare
Esempio pratico
np.logspace(-2, 2, 7)
array([0.01, 0.046, 0.215, 1.0, 4.64, 21.54, 100.0])
"""
param_svc = {
    # 'classifier': [SVC(probability=True)],
    'C': np.logspace(-2, 2, 7),  # 0.01 … 100
    # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    # 'degree': [2, 3, 4],
    "gamma": ["scale", 0.01, 0.1, 1.0]
    # 'class_weight': ['balanced'],
    # 'shrinking': [True, False]
}

# Logistic Regression
param_lg = {
    # 'classifier': [LogisticRegression()],
    "C": np.logspace(-2, 2, 9),
    # 'penalty': ['l2', 'l1', 'elasticnet', 'none'],
    'penalty': ['l2'],
    # 'solver': ['lbfgs', 'saga', 'liblinear'],
    # max_iter NON APPLICABILE QUANDO SI EFFETTUA HALVING PERCHE' CON IL PARAMETRO RESOURCE, GLI INCREMENTA AUTOMATICAMENTE
    # 'max_iter': [100, 200, 500],
    # 'class_weight': [None, 'balanced']
}

# Decision Tree
param_decision = {
    # 'classifier': [DecisionTreeClassifier()],
    # 'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 10, 10],
    'min_samples_leaf': [1, 2, 5],
    # 'max_features': [None, 'sqrt', 'log2'],
    # 'class_weight': [None, 'balanced']
}

# K-Nearest Neighbors
param_nei = {
    # 'classifier': [KNeighborsClassifier()],
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    # 'metric': ['minkowski', 'euclidean', 'manhattan'],
    'p': [1, 2]
}

# Random Forest -> ensemble
param_random = {
    # 'classifier': [RandomForestClassifier()],
    # n_estimators NON APPLICABILE QUANDO SI EFFETTUA HALVING PERCHE' CON IL PARAMETRO RESOURCE, GLI INCREMENTA AUTOMATICAMENTE
    # 'n_estimators': [50, 100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    # 'class_weight': [None, 'balanced'],
    'max_features': ['sqrt', 'log2']
}

# Gradient Boosting (LightGBM) -> ensemble
param_lgbm = {
    # 'classifier': [LGBMClassifier()],
    # n_estimators NON APPLICABILE QUANDO SI EFFETTUA HALVING PERCHE' CON IL PARAMETRO RESOURCE, GLI INCREMENTA AUTOMATICAMENTE
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [-1, 5, 10],
    'num_leaves': [31, 50, 100],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'scale_pos_weight': [1, 2, 5, 10]  # Per bilanciare le classi
}

# XGBClassifier -> ensemble
param_xgb = {
    # 'classifier': [XGBClassifier()],
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 5, 10],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'scale_pos_weight': [1, 2, 5, 10]  # Per gestire classi sbilanciate
}

# GradientBoosting -> ensemble
param_gb = {
    # 'classifier': [GradientBoostingClassifier()],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [200, 400, 800],
    "max_depth": [3, 5],
    "subsample": [0.7, 1.0],
    # 'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4],
    # 'max_features': ['sqrt', 'log2', None],
    # 'loss': ['log_loss', 'exponential']
}

# AdaBoost -> ensemble
params_ada = {
    # 'classifier': [AdaBoostClassifier()],
    "n_estimators": [200, 400, 800],
    "learning_rate": [0.05, 0.1, 0.5, 1.0],
    "algorithm": ["SAMME.R", "SAMME"]
}


class FitUtilitySearchBestModel(FitUtility):
    """
    | Attributo         | Descrizione                                                                |
    | ----------------- | -------------------------------------------------------------------------- |
    | `best_estimator_` | Il miglior modello addestrato con i migliori iperparametri trovati.        |
    | `best_params_`    | Dizionario con la combinazione ottimale di iperparametri.                  |
    | `best_score_`     | Lo score (secondo la metrica scelta) del miglior modello su CV.            |
    | `cv_results_`     | Dizionario dettagliato con tutti i risultati di cross-validation.          |
    | `refit`           | Se `True` (default su tutti), addestra il miglior modello finale sull’intero `X,y`. |
    | `scorer_`         | Funzione usata per calcolare la metrica (`scoring`).                       |

    es con refit=False:
        best_model = RandomForestClassifier(**search.best_params_)
        best_model.fit(X_train, y_train)
    """

    def __init__(self, X, y, name, **kwargs):
        super().__init__(X, y, **kwargs)
        self.X = X
        self.y = y
        self.name = name
        self.estimator, self.params = self.return_est_par_classifier()
        self.adapter_estimator_params()  # Adatta l'estimatore con SMOTE e Scaler se richiesto
        self.resource = self.check_resource_halving()
        self.cv = kwargs.get('cv') or StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.score = kwargs.get('score') or make_scorer(f1_score, average='weighted', zero_division=1)
        self.max_resources = kwargs.get('max_resources') or len(self.X)
        self.min_resources = kwargs.get('min_resources') or max(50, int(len(
            self.X) * 0.1))  # almeno 10% del dataset o 50 righe

    def adapter_estimator_params(self):
        # Qui la pipeline con SMOTE e Scaler (solo se richiesto)
        self.estimator = self.build_estimator(estimator=self.estimator)
        if isinstance(self.estimator, Pipeline):
            self.params = {f"model__{k}": v for k, v in self.params.items()}

    def return_est_par_classifier(self):
        match self.name:
            case 'random':
                return RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=42,
                                              class_weight='balanced'), param_random
            case 'svc':
                return SVC(kernel="rbf", probability=True, class_weight='balanced',  # proba per metriche su soglia
                           random_state=42), param_svc
            case 'lg':
                return LogisticRegression(
                    solver="lbfgs", max_iter=2000, n_jobs=-1, class_weight='balanced', random_state=42
                ), param_lg
            case 'decision':
                return DecisionTreeClassifier(
                    random_state=42, class_weight='balanced'
                ), param_decision
            case 'nei':
                return KNeighborsClassifier(), param_nei
            case 'lgbn':
                return LGBMClassifier(objective="binary", boosting_type="gbdt",
                                      n_jobs=-1, random_state=42), param_lgbm
            case 'xgb':
                return XGBClassifier(objective="binary:logistic", tree_method="hist",
                                     n_jobs=-1, random_state=42, eval_metric="logloss"), param_xgb
            case 'gb':
                return GradientBoostingClassifier(random_state=42), param_gb
            case 'ada':
                return AdaBoostClassifier(random_state=42), params_ada
            case _:
                raise Exception

    def fit_grid_search(self, **kwargs):
        """
        Usando il GridSearchCV, addestra e ritorna il miglior modello trovato tra quelli indicati utilizzando tutte le probabilità
        :return: GridSearch
        """
        gs = GridSearchCV(scoring=self.score, estimator=self.estimator,
                          param_grid=self.params, verbose=2, n_jobs=-1,
                          cv=self.cv, return_train_score=True)

        logging.info(f"<<< Start {gs} >>>")
        gs.fit(self.X, self.y)
        logging.info(f"<<< End {gs} >>>")

        check_best_cross = self.check_cross_save(estimator=gs, **kwargs)
        return self.check_grid_val(check_best_cross, gs)

    def fit_random_search(self, **kwargs):
        """
        Usando il RandomizedSearchCV, addestra e ritorna il miglior modello trovato tra quelli indicati in maniera random
        :return: RandomizedSearchCV
        """
        rs = RandomizedSearchCV(scoring=self.score, estimator=self.estimator, random_state=42,
                                n_iter=100,  # Regola questo valore in base al tempo disponibile
                                param_distributions=self.params, verbose=2, n_jobs=-1,
                                cv=self.cv, return_train_score=True)

        logging.info(f"<<< Start {rs} >>>")
        rs.fit(self.X, self.y)
        logging.info(f"<<< End {rs} >>>")

        check_best_cross = self.check_cross_save(estimator=rs, **kwargs)
        return self.check_grid_val(check_best_cross, rs)

    def fit_halving_search(self, factor=2, **kwargs):
        """
        È una versione più efficiente di GridSearchCV, che:
            parte con tante combinazioni di iperparametri,
            all’inizio assegna poche risorse a tutte le combinazioni,
            seleziona le migliori e assegna loro più risorse nei round successivi.
        Questa strategia è chiamata Successive Halving.

        | Parametro       | Descrizione                                                                     |
        | --------------- | ------------------------------------------------------------------------------- |
        | `estimator`     | Il modello da ottimizzare.                                                      |
        | `param_grid`    | Griglia completa di combinazioni di iperparametri (tutti testati inizialmente). |
        | `resource`      | Parametro da incrementare progressivamente, es. `n_estimators`.                 |
        | `min_resources` | Quante risorse usare inizialmente.                                              |
        | `max_resources` | Quante risorse al massimo assegnare.                                            |
        | `factor`        | Fattore di riduzione del numero di combinazioni (es. 2 → dimezza ogni volta).   |
        | `cv`            | Cross-validation.                                                               |
        | `scoring`       | Metrica da ottimizzare.                                                         |
        | `n_jobs`        | Parallelizzazione (usa `-1` per tutti i core).                                  |

        resource = iperparametro che crescerà (tipicamente n_estimators, max_iter, n_samples, ecc.)
        factor = quanti modelli scarti ogni volta (più alto → meno round, più drastici)
        :return: HalvingGridSearchCV

        | Valore `resource` | Per quali modelli?                                                      | Significato / Uso                                         | Note importanti                               |
        | ----------------- | ----------------------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------- |
        | `'n_estimators'`  | `RandomForest`, `GradientBoosting`, `XGB`                               | Numero di alberi nella foresta o boosting                 | Richiede `warm_start=True`                    |
        | `'max_iter'`      | `SGDClassifier`, `MLPClassifier`, `LogisticRegression(solver='saga')`   | Numero massimo di iterazioni di training                  | ✔ automatico senza warm_start                |
        | `'n_samples'`     | Qualsiasi modello                                                       | Numero di campioni usati (subset del dataset di training) | ✔ di default se non specifichi `resource`     |
        | `'iterations'`    | Algoritmi personalizzati (es. in reinforcement learning o custom model) | Iterazioni totali di addestramento                        | Devi implementarlo tu                         |
        | `'epochs'`        | Modelli deep learning (Keras, PyTorch via wrapper)                      | Numero di epoche per l'addestramento                      | Solo se integrato in un estimator compatibile |
        | `'n_calls'`       | In modelli tipo `BayesSearchCV`, `optuna`                               | Numero di chiamate valutative della funzione obiettivo    | Richiede wrapper ad hoc                       |

         I più usati in pratica:
            'n_estimators' → per ensemble model come:
                RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
                ⚠ Necessario warm_start=True per accumulare alberi in più round.

            'max_iter' → per modelli iterativi:
                SGDClassifier, MLPClassifier, LogisticRegression (solo con solver='saga' o liblinear)
                Funziona senza warm_start, perché l’iterazione è gestita internamente.
                'n_samples' (default se non imposti resource)
                Usa subset crescenti del dataset.

        Funziona con qualsiasi modello, ma è meno controllabile (non aumenta un iperparametro, ma riduce i dati iniziali).
        """
        hs = HalvingGridSearchCV(
            estimator=self.estimator, param_grid=self.params,
            factor=factor,  # ogni round dimezza i candidati migliori
            resource=self.resource,  # parametro da aumentare progressivamente -> la "risorsa" è il numero di alberi
            max_resources=self.max_resources,
            min_resources=self.min_resources,  # n_estimators iniziali # punto di partenza (alberi iniziali)
            cv=self.cv,  # validazione incrociata
            scoring=self.score,  # o 'accuracy'
            # random_state=42,  # riproducibilità
            verbose=2, return_train_score=True,
            n_jobs=-1  # usa tutti i core
        )

        logging.info(f"<<< Start {hs} >>>")
        hs.fit(self.X, self.y)
        logging.info(f"<<< End {hs} >>>")

        check_best_cross = self.check_cross_save(estimator=hs, **kwargs)
        return self.check_grid_val(check_best_cross, hs)

    def fit_halving_random_search(self, factor=2, **kwargs):
        """
        È una ricerca randomica degli iperparametri ottimizzata con Successive Halving. In pratica:

        Sceglie casualmente molte combinazioni di iperparametri.
        All’inizio allena velocemente (poche risorse) tutte le combinazioni.
        Scarta le peggiori e raddoppia le risorse per le migliori.
        Ripete fino a trovare il miglior modello.

        | Parametro             | Descrizione                                                                 |
        | --------------------- | --------------------------------------------------------------------------- |
        | `estimator`           | Il modello ML che vuoi ottimizzare.                                         |
        | `param_distributions` | Dizionario con le distribuzioni (o liste) da cui estrarre valori randomici. |
        | `factor`              | Fattore con cui **si riduce** il numero di candidati ad ogni iterazione.    |
        | `resource`            | Il parametro che rappresenta la "risorsa", es. `n_estimators`, `epochs`.    |
        | `max_resources`       | Quante risorse al massimo assegnare all’ultimo step.                        |
        | `min_resources`       | Quante risorse assegnare nel primo round.                                   |
        | `cv`                  | K-fold cross-validation.                                                    |
        | `scoring`             | La metrica da ottimizzare, es. `accuracy`, `f1`, `roc_auc`, ecc.            |
        | `random_state`        | Per rendere l’estrazione randomica ripetibile.                              |
        | `n_jobs`              | Numero di core usati. `-1` = tutti disponibili.                             |
                             |

        resource = iperparametro che crescerà (tipicamente n_estimators, max_iter, n_samples, ecc.)
        factor = quanti modelli scarti ogni volta (più alto → meno round, più drastici)
        :return: HalvingRandomSearchCV


        | Valore `resource` | Per quali modelli?                                                      | Significato / Uso                                         | Note importanti                               |
        | ----------------- | ----------------------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------- |
        | `'n_estimators'`  | `RandomForest`, `GradientBoosting`, `XGB`                               | Numero di alberi nella foresta o boosting                 | Richiede `warm_start=True`                    |
        | `'max_iter'`      | `SGDClassifier`, `MLPClassifier`, `LogisticRegression(solver='saga')`   | Numero massimo di iterazioni di training                  | ✔ automatico senza warm_start                |
        | `'n_samples'`     | Qualsiasi modello                                                       | Numero di campioni usati (subset del dataset di training) | ✔ di default se non specifichi `resource`     |
        | `'iterations'`    | Algoritmi personalizzati (es. in reinforcement learning o custom model) | Iterazioni totali di addestramento                        | Devi implementarlo tu                         |
        | `'epochs'`        | Modelli deep learning (Keras, PyTorch via wrapper)                      | Numero di epoche per l'addestramento                      | Solo se integrato in un estimator compatibile |
        | `'n_calls'`       | In modelli tipo `BayesSearchCV`, `optuna`                               | Numero di chiamate valutative della funzione obiettivo    | Richiede wrapper ad hoc                       |

        I più usati in pratica:
            'n_estimators' → per ensemble model come:
                RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
                ⚠ Necessario warm_start=True per accumulare alberi in più round.

            'max_iter' → per modelli iterativi:
                SGDClassifier, MLPClassifier, LogisticRegression (solo con solver='saga' o liblinear)
                Funziona senza warm_start, perché l’iterazione è gestita internamente.
                'n_samples' (default se non imposti resource)
                Usa subset crescenti del dataset.

        Funziona con qualsiasi modello, ma è meno controllabile (non aumenta un iperparametro, ma riduce i dati iniziali).

        """
        hrs = HalvingRandomSearchCV(
            estimator=self.estimator, param_distributions=self.params,
            factor=factor,  # ogni round dimezza i candidati migliori
            resource=self.resource,  # parametro da aumentare progressivamente -> la "risorsa" è il numero di alberi
            max_resources=self.max_resources,
            min_resources=self.min_resources,  # n_estimators iniziali # punto di partenza (alberi iniziali)
            cv=self.cv,  # validazione incrociata
            scoring=self.score,  # o 'accuracy'
            # random_state=42,  # riproducibilità
            verbose=2, return_train_score=True,
            n_jobs=-1  # usa tutti i core
        )

        logging.info(f"<<< Start {hrs} >>>")
        hrs.fit(self.X, self.y)
        logging.info(f"<<< End {hrs} >>>")

        check_best_cross = self.check_cross_save(estimator=hrs, **kwargs)
        return self.check_grid_val(check_best_cross, hrs)

    def check_resource_halving(self):
        """
         Determina la risorsa in base al tipo di modello
        :return:
            | Tipo modello                 | Resource consigliato |
            | ---------------------------- | -------------------- |
            | RandomForest, LGBM, XGB      | `'n_estimators'`     |
            | LogisticRegression (saga)    | `'max_iter'`         |
            | SGDClassifier, MLPClassifier | `'max_iter'`         |
            | Altro                        | `'n_samples'`        |
        """
        if isinstance(self.estimator, (RandomForestClassifier, LGBMClassifier)):
            self.estimator.set_params(warm_start=True)  # obbligatorio per RandomForest
            param_random.pop('n_estimators')
            param_lgbm.pop('n_estimators')
            return 'n_estimators'
        elif isinstance(self.estimator, (LogisticRegression,)):
            param_lg.pop('max_iter')
            return 'max_iter'
        else:
            return 'n_samples'  # fallback generico

    def check_grid_val(self, check_best_cross, grid):
        """
        Controlla se restituire il modello di ricerca o quello salvato
        :param grid: modello di ricerca (GridSearch, RandomizedSearch, HalvingGrid
        :param check_best_cross: modello salvato
        :return: modello di ricerca o quello salvato
        """
        if check_best_cross == 'fit':
            return grid
        elif self.best_cross_save == 'best_cross':
            return check_best_cross
        return None
