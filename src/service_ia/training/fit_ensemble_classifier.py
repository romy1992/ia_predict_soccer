import logging

import numpy as np
import xgboost as xgb
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, StackingClassifier

from src.service_ia.training.fit_utility import FitUtility

logging.basicConfig(level=logging.DEBUG)


class FitUtilityEnsembleClassifier(FitUtility):
    """
    Classe che utilizza gli algoritmi di ENSEMBLED di Classificazione per velocizzare i processi.
    Tutti i metodi ritornano il modello già addestrato(SE "cross_save" è None) e non con predizioni con OBBLIGO di inserire X e y:
        X e y possono essere sia quelli originali che quelli con split_train_test a patto che se si vuole provare entrambi
        si devono creare almeno 2 istanze di questa classe (uno per gli originali e uno per i test)
    """

    def __init__(self, X, y, **kwargs):
        """
        :param X: features
        :param y: label
        :param kwargs: cross_save:
            'cross' -> ritorna tutte le valutazioni
            'cross_save' -> valuta e salva il report finale
            'None' -> ritorna solo il modello addestrato
        """
        super().__init__(X, y, **kwargs)
        self.X = X
        self.y = y

    def fit_voting(self, estimators, voting_hs, **kwargs):
        """
          Con Voting che accumula i modelli e restituisce il migliore
        """
        model = VotingClassifier(estimators=estimators, voting=voting_hs, verbose=2, n_jobs=-1)
        # Qui la pipeline con SMOTE e Scaler (solo se richiesto)
        estimator = self.build_estimator(estimator=model)
        check_cross_val = self.check_cross_save(estimator=estimator, **kwargs)
        return self.check_val(check_cross_val, estimator)

    def fit_bagging_pasting(self, estimator, bagging=True, **kwargs):
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
        base_param = {'n_estimators': 500, 'random_state': 42, 'max_samples': 100}
        override_params = kwargs.get('override_params', {})

        params = override_params if len(override_params) > 0 else base_param
        model = BaggingClassifier(estimator=estimator, bootstrap=bagging, **params, n_jobs=-1, verbose=1)

        # Qui la pipeline con SMOTE e Scaler (solo se richiesto)
        estimator_pipe = self.build_estimator(estimator=model)
        check_cross_val = self.check_cross_save(estimator=estimator_pipe, **kwargs)
        return self.check_val(check_cross_val, estimator)

    def fit_random_(self, oob=True, **kwargs):
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
        base_param = {'max_leaf_nodes': 16, 'n_estimators': 500, 'random_state': 42}
        override_params = kwargs.get('override_params', {})

        params = override_params if len(override_params) > 0 else base_param
        model = RandomForestClassifier(**params, oob_score=oob, n_jobs=-1, verbose=1)

        # Qui la pipeline con SMOTE e Scaler (solo se richiesto)
        estimator = self.build_estimator(estimator=model)
        check_cross_val = self.check_cross_save(estimator=estimator, **kwargs)
        return self.check_val(check_cross_val, estimator)

    def fit_adaboost(self, estimator, alg_def=True, **kwargs):
        """
         Sulla base di un Algoritmo, proverà ad addestrare e controllare le stime.
         Se alcune stime saranno errate, si riaddestrerà sulla base di quelle stime cercando di migliorare sui suoi errori.
         - Estimator e n_estimator che sono i numeri di cloni dell'algoritmo.
         - Algorithm con SAMME.R (che è il default) userà le probabilità il che lo rende più robusto a SAMME che usa solo 1 o 0
         Quindi in base a estimator e al n_estimator, lui man mano si riaddestrerà sulla base degli errori precedenti
        """
        base_param = {'learning_rate': 0.5, 'n_estimators': 1000, 'random_state': 42}
        override_params = kwargs.get('override_params', {})
        params = override_params if len(override_params) > 0 else base_param

        algorithm = 'SAMME.R' if alg_def else 'SAMME'
        model = AdaBoostClassifier(estimator=estimator, **params, algorithm=algorithm)

        # Qui la pipeline con SMOTE e Scaler (solo se richiesto)
        estimator = self.build_estimator(estimator=model)
        check_cross_val = self.check_cross_save(estimator=estimator, **kwargs)
        return self.check_val(check_cross_val, estimator)

    def fit_gradient_boosting(self, **kwargs):
        """
        Gradient Boosting è simile ad ADABOOST ma invece che correggere le stime del precedente, lui le riadatta.
        Anche qui troviamo n_estimators che indica quante volte deve ripetere l'operazione per il riadattamento che di
        default usa l'algoritmo di DecisionTreeRegressor(max_depth=3)(anche se è per classificazione poi lui lo riadatta a 0 o 1)
        Quindi se il mio n_estimators ha 200, significa che userà 200 cloni dove ognuno dei quali riadatterà/correggerà
        il precedente.
        """
        base_param = {'learning_rate': 0.5, 'n_estimators': 200, 'random_state': 42, 'max_depth': 2}
        override_params = kwargs.get('override_params', {})
        params = override_params if len(override_params) > 0 else base_param
        model = GradientBoostingClassifier(**params)

        # Qui la pipeline con SMOTE e Scaler (solo se richiesto)
        estimator = self.build_estimator(estimator=model)
        check_cross_val = self.check_cross_save(estimator=estimator, **kwargs)
        return self.check_val(check_cross_val, estimator)

    def fit_xgb(self, **kwargs):
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
        base_param = {'learning_rate': 0.05, 'n_estimators': 500, 'max_depth': 3,
                      'subsample': 0.8, 'eval_metric': 'logloss'}
        override_params = kwargs.get('override_params', {})
        params = override_params if len(override_params) > 0 else base_param

        # Bilanciamento classi automatico
        if 'scale_pos_weight' not in params and not self.with_smote:
            pos = np.sum(self.y == 1)
            neg = np.sum(self.y == 0)
            ratio = neg / pos  # es. 9.0 se 10% positivi
            if ratio > 2:  # sbilanciamento significativo
                scale_pos_weight = ratio
            else:
                scale_pos_weight = 1  # bilanciato, nessun peso
            params.update({'scale_pos_weight': scale_pos_weight})  # bilancia il peso delle classi

        model = xgb.XGBClassifier(**params)

        # Qui la pipeline con SMOTE e Scaler (solo se richiesto)
        estimator = self.build_estimator(estimator=model)
        check_cross_val = self.check_cross_save(estimator=estimator, **kwargs)
        return self.check_val(check_cross_val, estimator)

    def fit_stacking_classifier(self,estimators, final_estimator, passthrough=True,
                                stack_method='predict_proba', **kwargs):
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
        # Costruisci (name, pipeline) con scaler→smote→est
        estimators_stak = [(name, self.build_estimator(estimator=est, name_estimator=name))
                               for name, est in estimators]
        stacking = StackingClassifier(
            estimators=estimators_stak,
            final_estimator=final_estimator,
            cv=5,  # K-fold è il cross validation
            verbose=2,
            n_jobs=-1,
            passthrough=passthrough,
            stack_method=stack_method
        )
        check_cross_val = self.check_cross_save(estimator=stacking, **kwargs)
        return self.check_val(check_cross_val, stacking)

    def check_val(self, check_cross_val, estimator):
        """
        Controlla se deve fare il fit o restituire il cross_val
        :param check_cross_val: 'fit' o 'cross'
        :param estimator: modello
        :return: model o cross_val
        """
        if check_cross_val == 'fit':
            logging.info(f'Start fit {estimator}')
            estimator.fit(self.X, self.y)
            logging.info(f'End fit {estimator}')
            return estimator
        elif self.cross_save == 'cross':
            return check_cross_val
        return None


# TODO
""" 
| Algoritmo                  | early_stopping / n_iter_no_change | eval_set richiesto | Dove si imposta?     |
|-----------------------------|-----------------------------------|--------------------|----------------------|
| **XGBClassifier**           | early_stopping_rounds             | Sì                 | Nel `.fit()`         |
| **LGBMClassifier**          | early_stopping_rounds             | Sì                 | Nel `.fit()`         |
| **HistGradientBoostingClassifier** | early_stopping, n_iter_no_change | No           | Nel costruttore      |
| **GradientBoostingClassifier**     | n_iter_no_change           | No                 | Nel costruttore      |
| **RandomForestClassifier**  | No                                | No                 | —                    |
| **LogisticRegression**      | No                                | No                 | —                    |
| **SVC**                     | No                                | No                 | —                    |
| **AdaBoostClassifier**      | No                                | No                 | —                    |

**Note:**
- Per XGBoost e LightGBM, `early_stopping_rounds` e `eval_set` vanno passati nel metodo `.fit()`.
- Per i modelli scikit-learn con `n_iter_no_change` o `early_stopping`, si imposta nel costruttore.
- I modelli classici (`RandomForest`, `SVC`, `LogisticRegression`, `AdaBoost`) non supportano l’early stopping.

I modelli che supportano l’early stopping (come XGBoost, LightGBM, GradientBoosting) lo fanno perché l’addestramento avviene in modo iterativo, aggiungendo ogni volta un nuovo “estimator” (tipicamente un albero) o passando per più “epoche” (iterazioni).
Boosting (XGB, LGBM, GradientBoosting): aggiungono un albero alla volta, correggendo gli errori dei precedenti.
Early stopping: controlla dopo ogni iterazione/epoca se la metrica di validazione migliora. Se non migliora per un certo numero di iterazioni, l’addestramento si ferma prima.


# Regole di scaler
Modelli ad albero: DecisionTree, RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM, CatBoost
Modelli rule-based: RuleFit, Explainable Boosting Machine
Naive Bayes: MultinomialNB, CategoricalNB

"""
