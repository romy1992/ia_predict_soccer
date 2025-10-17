import datetime
import logging
import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    log_loss, brier_score_loss, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.service_ia.training.utility_training.save_load import SaveLoad

logging.basicConfig(level=logging.DEBUG)


class FitUtility(SaveLoad):
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
            'with_scaler' -> True/false
            'with_smote' -> True/false
        """
        super().__init__(**kwargs)
        self.X = X
        self.y = y
        self.keys = kwargs.get('keys')
        self.file_path_report = 'report_fit_grid.xlsx'
        self.cross_save = kwargs.get('cross_save')
        self.best_cross_save = kwargs.get('best_cross_save')
        self.with_smote = kwargs.get('with_smote', False)
        self.with_scaler = kwargs.get('with_scaler', False)

    def check_cross_save(self, estimator, **kwargs):
        """
        :param estimator: modello da valutare/salvare
        :param kwargs: cross_save:
            'cross' -> ritorna tutte le valutazioni
            'cross_save' -> valuta e salva il report finale
            'None' -> ritorna solo il modello addestrato
        """
        des = kwargs.get('des')
        threshold = kwargs.get('threshold', None)
        cv = kwargs.get('cv')
        match self.cross_save or self.best_cross_save:
            case 'cross':
                result = self.cross(estimator=estimator, des=des, threshold=threshold, cv=cv)
                self.save_model(estimator=estimator)
                return result
            case 'cross_save':
                # result = self.cross(estimator=estimator, des=des, threshold=threshold, cv=cv)
                result_2 = self.cross_2(estimator=estimator, des=des, threshold=threshold, cv=cv)
                # result_3 = self.evaluate_classification(estimator=estimator, des=des, threshold=threshold, cv=cv)
                # print(result)
                # print(result_2)
                # print(result_3)
                self.save_report(results_=result_2)
                self.save_model(estimator=estimator)
                return None
            case 'best_cross':
                result = self.cross_best_search(estimator, des=des)
                self.save_model(estimator=estimator)
                return result
            case 'best_cross_save':
                result_best = self.cross_best_search(estimator, des=des)
                self.save_report(results_=result_best, sheet_name='Report BEST FIT')
                self.save_model(estimator=estimator)
                return None
            case _:
                return 'fit'

    def evaluate_classification(self, estimator, **kwargs):
        """
        Valutazione completa e pilotabile per classificazione binaria.

        Kwargs utili:
          - threshold: float opzionale -> soglia manuale principale
          - pos_label: etichetta positiva (default 1)
          - cost_fp, cost_fn: pesi errore FP/FN per soglia min-costo (default 1.0/1.0)
          - target_precision: vincolo minimo precision (es. 0.90)
          - target_recall: vincolo minimo recall (es. 0.95)
          - cv: oggetto CV (default StratifiedKFold 5x)
          - zero_division: 0/1 (default 0)

        Ritorna dict con:
          meta, oof_scores, thresholds_table, cv_metrics, metrics_global,
          confusion_matrix_abs/norm, report, curves(roc/pr), main_threshold, main_pred
        """
        # -------------------- setup --------------------
        cv = kwargs.get('cv') or StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pos_label = kwargs.get('pos_label', 1)
        user_thr = kwargs.get('threshold', None)
        cost_fp = float(kwargs.get('cost_fp', 1.0))
        cost_fn = float(kwargs.get('cost_fn', 1.0))
        target_precision = kwargs.get('target_precision', None)
        target_recall = kwargs.get('target_recall', None)
        zero_div = kwargs.get('zero_division', 0)

        X = self.X
        y = np.asarray(self.y)
        classes = np.array(sorted(np.unique(y)))
        positive_label = pos_label if pos_label in classes else classes[-1]

        # -------------------- helpers ------------------
        def _metrics_row(y_true, y_pred, scores, name, thr):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            bal_acc = (recall_score(y_true, y_pred, zero_division=zero_div) + spec) / 2.0
            denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
            row = {
                'name': name,
                'threshold': float(thr),
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=zero_div),
                'recall': recall_score(y_true, y_pred, zero_division=zero_div),
                'f1': f1_score(y_true, y_pred, zero_division=zero_div),
                'specificity': spec,
                'npv': npv,
                'balanced_accuracy': bal_acc,
                'mcc': mcc
            }
            if scores is not None:
                s_clip = np.clip(scores, 1e-12, 1 - 1e-12)
                try:
                    row['log_loss'] = log_loss(y_true, s_clip)
                except Exception:
                    row['log_loss'] = np.nan
                try:
                    row['brier'] = brier_score_loss(y_true, s_clip)
                except Exception:
                    row['brier'] = np.nan
            else:
                row['log_loss'] = np.nan
                row['brier'] = np.nan
            return row

        def _thr_max_f1(precision, recall, thr):
            if len(thr) == 0: return 0.5
            f1s = 2 * precision * recall / (precision + recall + 1e-12)
            idx = int(np.nanargmax(f1s[:-1]))  # ultime entry di P/R non hanno thr
            return float(thr[idx])

        def _thr_max_j(fpr, tpr, thr):
            if len(thr) == 0: return 0.5
            j = tpr - fpr
            idx = int(np.argmax(j[:-1])) if len(j) > 1 else 0
            return float(thr[idx])

        def _thr_min_cost(y_true, s, c_fp, c_fn, thr_grid):
            if len(thr_grid) == 0: return 0.5
            best_t, best_cost = thr_grid[0], np.inf
            for t in thr_grid:
                yp = np.array(s >= t, dtype=int)  # correzione: garantisce array int
                tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
                cost = c_fp * fp + c_fn * fn
                if cost < best_cost:
                    best_cost, best_t = cost, t
            return float(best_t)

        def _thr_for_precision(target, precision, thr):
            if target is None or len(thr) == 0: return None
            for i, t in enumerate(thr):
                if precision[i] >= target:
                    return float(t)
            return None

        def _thr_for_recall(target, recall, thr):
            if target is None or len(thr) == 0: return None
            for i, t in enumerate(thr):
                if recall[i] >= target:
                    return float(t)
            return None

        # -------------------- OOF scores ----------------
        scores = None
        try:
            proba = cross_val_predict(estimator, X, y, cv=cv, method='predict_proba')
            pos_idx = int(np.where(classes == positive_label)[0][0])
            scores = proba[:, pos_idx].astype(float)
        except Exception:
            try:
                scores = cross_val_predict(estimator, X, y, cv=cv, method='decision_function').astype(float)
            except Exception:
                # fallback: etichette
                y_oof = cross_val_predict(estimator, X, y, cv=cv)
                if set(np.unique(y_oof)) <= {0, 1}:
                    y_oof = y_oof.astype(int)
                scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision',
                           'neg_log_loss', 'neg_brier_score', 'f1_macro', 'f1_weighted', 'balanced_accuracy']
                cv_metrics = cross_validate(estimator, X, y, cv=cv, scoring=scoring)
                cm_abs = confusion_matrix(y, y_oof, labels=[0, 1]) if set(np.unique(y)) <= {0, 1} else confusion_matrix(
                    y, y_oof)
                cm_norm = cm_abs / cm_abs.sum(axis=1, keepdims=True)
                rep = classification_report(y, y_oof, output_dict=True, zero_division=zero_div)
                return {
                    'meta': {'estimator': estimator, 'date': str(datetime.date.today()),
                             'pos_label': positive_label, 'notes': 'No continuous scores: label-only evaluation'},
                    'oof_scores': None, 'thresholds_table': None, 'cv_metrics': cv_metrics,
                    'metrics_global': None, 'confusion_matrix_abs': cm_abs, 'confusion_matrix_norm': cm_norm,
                    'report': rep, 'curves': None, 'main_threshold': None, 'main_pred': y_oof
                }

        # -------------------- curve ---------------------
        fpr, tpr, thr_roc = roc_curve(y, scores, pos_label=positive_label)
        roc_auc = auc(fpr, tpr)
        prec, rec, thr_pr = precision_recall_curve(y, scores, pos_label=positive_label)
        ap = average_precision_score(y, scores)

        # -------------------- soglie --------------------
        thr_f1 = _thr_max_f1(prec, rec, thr_pr)
        thr_j = _thr_max_j(fpr, tpr, thr_roc)
        thr_cost = _thr_min_cost(y, scores, cost_fp, cost_fn, thr_pr)
        thr_prec = _thr_for_precision(target_precision, prec, thr_pr)
        thr_rec_ = _thr_for_recall(target_recall, rec, thr_pr)

        thr_candidates = {
            'user': float(user_thr) if user_thr is not None else None,
            'f1': thr_f1,
            'youden_j': thr_j,
            'min_cost': thr_cost,
            'prec_at_target': thr_prec,
            'rec_at_target': thr_rec_
        }

        rows = []
        for name, tval in thr_candidates.items():
            if tval is None:
                continue
            y_pred = (scores >= tval).astype(int)
            rows.append(_metrics_row(y, y_pred, scores, name, tval))
        thresholds_table = pd.DataFrame(rows).sort_values('f1', ascending=False) if rows else None

        # -------------------- CV metrics ----------------
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision',
                   'neg_log_loss', 'neg_brier_score', 'f1_macro', 'f1_weighted', 'balanced_accuracy']
        cv_metrics = cross_validate(estimator, X, y, cv=cv, scoring=scoring)

        # -------------------- main threshold ------------
        main_thr = float(user_thr) if user_thr is not None else float(thr_f1)
        y_main = (scores >= main_thr).astype(int)

        cm_abs = confusion_matrix(y, y_main, labels=[0, 1])
        cm_norm = cm_abs / cm_abs.sum(axis=1, keepdims=True)
        rep = classification_report(y, y_main, labels=[0, 1], target_names=['neg', 'pos'],
                                    output_dict=True, zero_division=zero_div)

        metrics_global = {
            'roc_auc': float(roc_auc),
            'avg_precision': float(ap),
            'log_loss': float(log_loss(y, np.clip(scores, 1e-12, 1 - 1e-12))),
            'brier': float(brier_score_loss(y, np.clip(scores, 1e-12, 1 - 1e-12))),
        }

        return {
            'data': datetime.date.today(),
            'path_save_model': self.filename,
            'description': kwargs.get('des'),
            'meta': {'estimator': estimator, 'date': str(datetime.date.today()),
                     'pos_label': positive_label, 'cv': str(cv),
                     'notes': 'OOF scores; thresholds evaluated on OOF'},
            'oof_scores': scores,
            'thresholds_table': thresholds_table,
            'cv_metrics': cv_metrics,
            'metrics_global': metrics_global,
            'confusion_matrix_abs': cm_abs,
            'confusion_matrix_norm': cm_norm,
            'report': rep,
            'curves': {'roc': {'fpr': fpr, 'tpr': tpr, 'thr': thr_roc},
                       'pr': {'precision': prec, 'recall': rec, 'thr': thr_pr}},
            'main_threshold': main_thr,
            'main_pred': y_main
        }

    def cross(self, estimator, **kwargs):
        """
        Esegue una valutazione di classificazione binaria tramite cross-validation.
        
        Parametri:
            estimator: modello sklearn compatibile (deve supportare predict_proba)
            des: descrizione opzionale (str)
            threshold: soglia manuale opzionale (float, non usata: la soglia viene ottimizzata su F1)
            cv: oggetto cross-validation (default StratifiedKFold 5x)
        
        Funzionamento:
            - Calcola le probabilità della classe positiva tramite cross_val_predict
            - Ottimizza la soglia di classificazione per il massimo F1 sulla curva precision-recall
            - Genera le predizioni binarie con la soglia ottimale
            - Calcola metriche di classificazione (accuracy, precision, recall, f1, confusion matrix, report)
            - Restituisce un dizionario con tutti i risultati principali
        
        Output:
            dict con: estimator, data, description, cross_val_score, cross_val, cross_val_predict,
            accuracy, f1, precision, recall, threshold, confusion matrix, report
        """
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
            'path_save_model': self.filename,
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

    def cross_2(self, estimator, **kwargs):
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
        pos_label = kwargs.get('pos_label', 1)
        # threshold:
        # - se punteggi di probabilità: default 0.5
        # - se decision_function: default 0.0
        threshold = kwargs.get('threshold', None)

        # 1) Scegliamo il metodo migliore disponibile
        has_proba = hasattr(estimator, "predict_proba")
        has_decision = hasattr(estimator, "decision_function")

        method_used, scores_continuous, y_pred_labels = self.choice_score(cv, estimator, has_decision, has_proba,
                                                                          pos_label, threshold)

        # Restituisce array accuracy_score in base agli split
        # 3) CV scores (sklearn calcola internamente proba/decision se servono per alcune metriche)
        cross_score = cross_val_score(estimator, self.X, self.y, cv=cv)
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
        cross_val = cross_validate(estimator, self.X, self.y, cv=cv,
                                   scoring=scoring,
                                   # return_estimator=True, n_jobs=-1
                                   # return_estimator=True mi ritorna gli stimatori di ogni fold fittati
                                   )
        # fold_estimators = cross_val['estimator'] # lista di modelli fittati (uno per fold)
        # Volendo puoi scegliere “il migliore” tra i fold in base a una metrica e poi ri-fittarlo su tutto il dataset:
        # metric = 'f1'  # o quello che preferisci
        # best_idx = int(np.nanargmax(cv_res[f'test_{metric}']))
        # best_fold_est = fold_estimators[best_idx]
        # final_estimator = clone(best_fold_est).fit(self.X, self.y)

        # 4) Gestione binary/multiclasse
        unique_classes = np.unique(self.y)
        is_binary = (len(unique_classes) == 2)
        avg_type = 'binary' if is_binary else 'macro'

        # 5) Metriche su OOF labels
        accuracy = accuracy_score(self.y, y_pred_labels)
        precision = precision_score(self.y, y_pred_labels, average=avg_type, zero_division=0)
        recall = recall_score(self.y, y_pred_labels, average=avg_type, zero_division=0)
        f1 = f1_score(self.y, y_pred_labels, average=avg_type, zero_division=0)
        cm = confusion_matrix(self.y, y_pred_labels)
        report = classification_report(self.y, y_pred_labels, output_dict=True, zero_division=0)

        # 6) Curve PR/ROC solo se binario e abbiamo punteggi continui
        precisions = recalls = pr_thresholds = None
        fpr = tpr = roc_thresholds = None
        ap = roc_auc = None

        if is_binary and scores_continuous is not None:
            try:
                precisions, recalls, pr_thresholds = precision_recall_curve(self.y, scores_continuous,
                                                                            pos_label=pos_label)
                ap = average_precision_score(self.y, scores_continuous, pos_label=pos_label)
            except Exception as e:
                logging.warning(f"Errore precision_recall_curve: {e}")

            try:
                fpr, tpr, roc_thresholds = roc_curve(self.y, scores_continuous, pos_label=pos_label)
                roc_auc = roc_auc_score(self.y, scores_continuous)
            except Exception as e:
                logging.warning(f"Errore roc_curve/roc_auc: {e}")

        results = {
            'estimator': estimator,
            'data': datetime.date.today(),
            'path_save_model': self.filename,
            'description': kwargs.get('des'),
            'method_used': method_used,
            'cross_val_score': cross_score,
            'cross_val': cross_val,
            'cross_val_predict': y_pred_labels,  # SEMPRE 0/1
            'scores_continuous': scores_continuous,  # opzionale, se disponibile
            'threshold_used': (0.5 if method_used == 'predict_proba' and threshold is None
                               else 0.0 if method_used == 'decision_function' and threshold is None
            else threshold),
            'pos_label': pos_label,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'classification_report': report,

            'pr_curve': {
                'precisions': precisions,
                'recalls': recalls,
                'thresholds': pr_thresholds,
                'average_precision': ap
            } if is_binary else None,

            'roc_curve': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': roc_thresholds,
                'roc_auc': roc_auc
            } if is_binary else None
        }

        logging.info(f'End cross {estimator}, {kwargs}')
        return results

    def choice_score(self, cv, estimator, has_decision: bool, has_proba: bool, pos_label: int, threshold):
        method_used = None
        scores_continuous = None  # punteggi continui OOF per ROC/PR
        try:
            if has_proba:
                method_used = "predict_proba"
                proba = cross_val_predict(estimator, self.X, self.y, cv=cv, method="predict_proba")
                # colonna della classe positiva
                if proba.ndim == 2 and proba.shape[1] > 1:
                    scores_continuous = proba[:, 1] if pos_label == 1 else proba[:, pos_label]
                else:
                    # caso raro (binario ma restituito in shape (n_samples,1))
                    scores_continuous = proba.ravel()
                thr = 0.5 if threshold is None else float(threshold)
                y_pred_labels = (scores_continuous >= thr).astype(int)

            elif has_decision:
                method_used = "decision_function"
                decision = cross_val_predict(estimator, self.X, self.y, cv=cv, method="decision_function")
                scores_continuous = decision.ravel() if decision.ndim > 1 else decision
                thr = 0.0 if threshold is None else float(threshold)
                y_pred_labels = (scores_continuous >= thr).astype(int)
            else:
                method_used = "predict"
                y_pred_labels = cross_val_predict(estimator, self.X, self.y, cv=cv)
                # niente punteggi continui disponibili
        except Exception as e:
            logging.warning(f"Errore in cross_val_predict con metodo {method_used}: {e}. Fallback a 'predict'.")
            method_used = "predict"
            y_pred_labels = cross_val_predict(estimator, self.X, self.y, cv=cv)
            scores_continuous = None

        return method_used, scores_continuous, y_pred_labels

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
            'path_save_model': self.filename,
            'description': kwargs.get('des'),
            'features': self.keys if self.keys else self.X.columns,
            'best_estimator': str(best_estimator),
            'best_params': str(best_params),
            'best_scorer': str(best_scorer),
            'scorer': str(scorer),
        }

    def build_estimator(self, estimator, name_estimator='model'):
        """
        Costruisce un pipeline con scaler e smote opzionali
        :param name_estimator: nome del modello
        :param estimator: algoritmo sklearn
        :return: Pipeline o modello singolo
        """
        steps = []
        # Alcuni algoritmi non necessitano di scaling
        alg_not_scale = ['RandomForestClassifier', 'GradientBoostingClassifier', 'DecisionTreeClassifier',
                         'ExtraTreesClassifier', 'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier']
        if self.with_scaler and estimator.__class__.__name__ not in alg_not_scale:
            steps.append(('scaler', StandardScaler()))

        if self.with_smote:
            steps.append(('smote', SMOTE(random_state=42)))
            # Disabilita i pesi di classe se si usa SMOTE (non ha senso)
            if not isinstance(estimator, KNeighborsClassifier):
                if hasattr(estimator, 'class_weight'):
                    estimator.set_params(class_weight=None)
                if hasattr(estimator, 'scale_pos_weight'):
                    estimator.set_params(scale_pos_weight=None)

        steps.append((name_estimator, estimator))

        # Se c'è più di un passo, crea una pipeline
        return Pipeline(steps) if len(steps) > 1 else estimator

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
