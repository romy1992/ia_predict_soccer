import datetime
import os

import pandas as pd
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    precision_recall_curve, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, \
    cross_val_score


class FitEnsembleClassifier:
    def __init__(self, **kwargs):
        super().__init__()
        self.X = kwargs.get('X')
        self.y = kwargs.get('y')
        self.file_path_report = 'report.xlsx'

    def fit_voting(self, estimators, voting_hs):
        """
          Con Voting che accumula i modelli e restituisce il migliore
        """
        voting = VotingClassifier(estimators=estimators, voting=voting_hs, verbose=0, n_jobs=-1)
        voting.fit(self.X, self.y)
        return voting

    def fit_process_bagging_pasting(self, estimator, bagging=True):
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
        model = BaggingClassifier(estimator=estimator, n_estimators=500, n_jobs=-1, verbose=1, max_samples=100,
                                  bootstrap=bagging, random_state=42)
        model.fit(self.X, self.y)
        return model

    def cross(self, estimator, **kwargs):
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
                return cross_val_predict(estimator, self.X, self.y, cv=cv, method='decision_function') > threshold
            except Exception as e:
                # print(str(e))
                return cross_val_predict(estimator, self.X, self.y, cv=cv) > threshold

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

        cross_score = cross_val_score(estimator, self.X, self.y,
                                      cv=cv)  # Restituisce array accuracy_score in base agli split
        cross_val = cross_validate(estimator, self.X, self.y, cv=cv,  # return_estimator=True,
                                   scoring=scoring)  # Come val_score con return di estimatori
        predict_score = predict_score().astype(int)
        accuracy = accuracy_score(self.y, predict_score)
        precision = precision_score(self.y, predict_score)  # Falsi positivi
        recall = recall_score(self.y, predict_score)  # Veri positivi
        f1 = f1_score(self.y, predict_score)  # Unisce il recall e la precisione
        cm = confusion_matrix(self.y,
                              predict_score)  # Un array che va vedere i falsi/veri positivi/negativi per le classi
        precisions, recalls, thresholds = precision_recall_curve(self.y, predict_score)  # Return stesse descritte prima

        report = classification_report(self.y, predict_score, output_dict=True)
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
        return results

    def save_report(self, results_):
        """
        Salva il report dei risultati
        """
        results = {}
        for el in results_.items():
            if el[0] != 'cross_val_predict':
                results.update({el[0]: str(el[1])})
        new_row = pd.DataFrame([results])

        if os.path.exists(self.file_path_report):
            # Legge il file esistente e aggiunge la nuova riga
            existing_df = pd.read_excel(self.file_path_report)
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        else:
            # Crea il DataFrame con la nuova riga
            updated_df = new_row

        # Salva il file aggiornato
        updated_df.to_excel(self.file_path_report, index=False)
