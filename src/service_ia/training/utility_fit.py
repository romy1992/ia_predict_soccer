import datetime
import logging
import os

import numpy as np
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
        self.keys = kwargs.get('keys')
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
        logging.info(f'Start cross {estimator} and {kwargs}')
        cv = kwargs.get('cv') or StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 1) Probabilità della classe positiva con CV
        proba = cross_val_predict(estimator, self.X, self.y, cv=cv, method='predict_proba')
        # inferisci le classi dal target (ordine crescente come in sklearn)
        classes = np.array(sorted(np.unique(self.y)))
        # scegli la classe positiva (assumo 1 se presente; altrimenti l’ultima)
        positive_label = 1 if 1 in classes else classes[-1]
        pos_idx = int(np.where(classes == positive_label)[0][0])
        pos_proba = proba[:, pos_idx].astype(float)

        # 2) Soglia ottimale (F1 sulla PR-curve)
        prec, rec, thr = precision_recall_curve(self.y, pos_proba)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        best_i = int(np.nanargmax(f1s))
        best_thr = float(thr[best_i]) if best_i < len(thr) else 0.5

        # 3) Predizione binaria con la soglia scelta
        y_pred = (pos_proba >= best_thr).astype(int)

        # 4) Report metriche
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision',
                   'neg_log_loss', 'neg_brier_score', 'f1_macro', 'f1_weighted', 'balanced_accuracy']
        cross_score = cross_val_score(estimator, self.X, self.y, cv=cv)
        cross_val = cross_validate(estimator, self.X, self.y, cv=cv, scoring=scoring)

        acc = accuracy_score(self.y, y_pred)
        prec1 = precision_score(self.y, y_pred, zero_division=0)
        rec1 = recall_score(self.y, y_pred, zero_division=0)
        f1 = f1_score(self.y, y_pred, zero_division=0)
        cm = confusion_matrix(self.y, y_pred, labels=[0, 1])
        report = classification_report(self.y, y_pred, labels=[0, 1],
                                       target_names=['neg', 'pos'], output_dict=True, zero_division=0)

        results = {
            'estimator': estimator,
            'data': datetime.date.today(),
            'description': kwargs.get('des'),
            'cross_val_score': cross_score,
            'cross_val': cross_val,
            'cross_val_predict': y_pred,
            'accuracy': acc,
            'f1': f1,
            'precision': prec1,
            'recall': rec1,
            'threshold': best_thr,
            'confusion matrix': cm,
        }
        results.update(report)
        logging.info(f'End cross with threshold={best_thr:.4f}, classes={classes}')
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
            'features': self.keys if self.keys else self.X.columns,
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
