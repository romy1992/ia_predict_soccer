import datetime
import logging
import os

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    precision_recall_curve, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, cross_val_predict

logging.basicConfig(level=logging.DEBUG)


class UtilityFit:
    """
    Classe di supporto per poter valutare e salvare i report
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
        super().__init__()
        self.X = X
        self.y = y
        self.file_path_report = 'report_fit_grid.xlsx'
        self.cross_save = kwargs.get('cross_save')
        self.best_cross_save = kwargs.get('best_cross_save')

    def check_cross_save(self, estimator, **kwargs):
        """
        :param estimator: modello da valutare/salvare
        :param kwargs: cross_save:
            'cross' -> ritorna tutte le valutazioni
            'cross_save' -> valuta e salva il report finale
            'None' -> ritorna solo il modello addestrato
        """
        des = kwargs.get('des')
        threshold = kwargs.get('threshold')
        cv = kwargs.get('cv')
        match self.cross_save or self.best_cross_save:
            case 'cross':
                return self.cross(estimator=estimator, des=des, threshold=threshold, cv=cv)
            case 'cross_save':
                result = self.cross(estimator=estimator, des=des, threshold=threshold, cv=cv)
                self.save_report(results_=result)
            case 'best_cross':
                return self.cross_best_search(estimator, des=des)
            case 'best_cross_save':
                result_best = self.cross_best_search(estimator, des=des)
                self.save_report(results_=result_best, sheet_name='Report BEST FIT')
            case _:
                return 'fit'

    def cross(self, estimator, **kwargs):
        """
        Unisce metodi di cross per valutare gli algoritmi
        | Metodo                | Tipo di output              | Supportato da                        | Uso tipico                 |
        | --------------------- | --------------------------- | ------------------------------------ | -------------------------- |
        | `predict_proba()`     | Probabilità per ogni classe | RandomForest, GradientBoosting, ecc. | ROC, Precision/Recall      |
        | `decision_function()` | Score continuo grezzo       | SVC, LogisticRegression              | ROC, SVM margine           |
        | `predict()`           | Classe predetta             | Tutti                                | Accuracy, Confusion Matrix |

        """
        logging.info(f'Start cross {estimator} and {kwargs}')
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
                logging.warning(f"Errore in cross_val_predict: {e}")
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
                              predict_score)  # Un array che fa vedere i falsi/veri positivi/negativi per le classi
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

        logging.info(f'Start cross {estimator}, {kwargs} and {results}')
        return results

    def cross_best_search(self, search, **kwargs):
        logging.info(f'Start cross_best_search {search}')
        best_estimator = search.best_estimator_
        best_params = search.best_params_
        best_scorer = search.best_score_
        scorer = search.scorer_
        logging.info(search.cv_results_)  # Restituisce tutti i risultati
        logging.info(f'End cross_best_search {search}')
        return {
            'data': datetime.date.today(),
            'description': kwargs.get('des'),
            'features': None,  # self.X.columns,
            'best_estimator': str(best_estimator),
            'best_params': str(best_params),
            'best_scorer': str(best_scorer),
            'scorer': str(scorer),
        }

    def save_report(self, results_, sheet_name='Report ML FIT'):
        """
        Salva il report dei risultati nel foglio specificato
        """
        results = {k: str(v) for k, v in results_.items() if k != 'cross_val_predict'}
        new_row = pd.DataFrame([results])

        # Se il file esiste e contiene altri fogli, lo carichiamo
        if os.path.exists(self.file_path_report):
            try:
                with pd.ExcelWriter(self.file_path_report, mode='a', engine='openpyxl',
                                    if_sheet_exists='overlay') as writer:
                    try:
                        existing_df = pd.read_excel(self.file_path_report, sheet_name=sheet_name)
                        updated_df = pd.concat([existing_df, new_row], ignore_index=True)
                    except ValueError:
                        # Il foglio non esiste ancora
                        updated_df = new_row
                    updated_df.to_excel(writer, sheet_name=sheet_name, index=False)
            except Exception as e:
                logging.error(f'Errore salvataggio report: {e}')
        else:
            # File non esiste, creiamo da zero
            with pd.ExcelWriter(self.file_path_report, engine='openpyxl') as writer:
                new_row.to_excel(writer, sheet_name=sheet_name, index=False)

        logging.info(f'Salvato il report nel foglio: {sheet_name}')
