import os
from collections import Counter

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score,
                             precision_recall_curve, average_precision_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from tabulate import tabulate


class BaseHierOverUnderClassifier(BaseEstimator, ClassifierMixin):
    """
    Classe base per HierOverUnderClassifier.
    Definisce l'interfaccia e alcune utility comuni.
    """

    def __init__(self, base_1, base_2, base_3, calibrator='isotonic', thresholds=(0.5, 0.5, 0.5)):
        """
        Inizializza il classificatore gerarchico con i modelli di base, il metodo di calibrazione e le soglie.
        :param base_1:
        :param base_2:
        :param base_3:
        :param calibrator: Metodo di calibrazione ('isotonic' o 'sigmoid').
        :param thresholds: Soglie per le tre classi (over 1.5, over 2.5, over 3.5).
        """
        self.base_1 = base_1
        self.base_2 = base_2
        self.base_3 = base_3
        self.calibrator = calibrator
        self.thresholds = thresholds

        # ---------- UTILITIES PRIVATE ----------

    @staticmethod
    def _best_threshold_f1(y_true, y_proba):
        """
        Ritorna (thr_best, f1_best) massimizzando l'F1 sulla curva Precision-Recall.
        precision_recall_curve restituisce precisions/recalls di len(thr)+1.
        Allineiamo i punti a thr usando f1[1:].
        """
        precisions, recalls, thr = precision_recall_curve(y_true, y_proba, pos_label=1)
        # Evita divisioni per zero / NaN
        f1 = 2 * precisions * recalls / (precisions + recalls + 1e-12)
        if thr.size == 0:
            # fallback: nessuna soglia calcolabile, usa 0.5
            return 0.5, 0.0
        f1_on_thr = f1[1:]  # allineamento con thr
        idx = int(np.nanargmax(f1_on_thr))
        return float(thr[idx]), float(f1_on_thr[idx])

    def _cal(self, est, prefit=True):
        """
        Calibra un classificatore binario usando CalibratedClassifierCV.
        :param est: Classificatore binario da calibrare.
        :param prefit: Se True, usa CalibratedClassifierCV con prefit=True.
        :return: Classificatore calibrato.
        """
        if prefit:
            return CalibratedClassifierCV(est, method=self.calibrator, cv='prefit')
        return CalibratedClassifierCV(est, method=self.calibrator,
                                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))

    def fit(self, X, y):
        """
        Fit del modello.
        :param X:
        :param y:
        :return:
        """
        raise NotImplementedError("Metodo fit() non implementato.")

    def predict_proba(self, X, **kwargs):
        """
        Predice le probabilità per ciascuna delle tre classi.
        :param X:
        :param kwargs:
        :return:
        """
        raise NotImplementedError("Metodo predict_proba() non implementato.")

    def cross_hierarchical(self, estimator, y, X, stratify_on=1):
        """
        Valuta HierOverUnderClassifier su 3 target (>[1.5, 2.5, 3.5]) usando 3 dataset distinti.
        - Passa EITHER X=[X1, X2, X3] OR X1,X2,X3 separatamente.
        - stratify_on: indice del target per lo split stratificato (default 2.5 -> 1)
        """
        raise NotImplementedError("Metodo cross_hierarchical() non implementato.")

    def predict(self, X):
        """
        Effettua le predizioni finali applicando le soglie alle probabilità predette.
        :param X:
        :return: Predizioni finali per ciascuna delle tre classi.
        """
        # Ottieni le probabilità predette
        proba = self.predict_proba(X=X)
        p1, p2, p3 = proba[:, 1], proba[:, 3], proba[:, 5]

        # Applica le soglie
        t1, t2, t3 = self.thresholds
        y1 = (p1 >= t1).astype(int)
        y2 = (p2 >= t2).astype(int)
        y3 = (p3 >= t3).astype(int)

        # Assicura la coerenza gerarchica delle predizioni
        y2 = np.minimum(y2, y1)
        y3 = np.minimum(y3, y2)
        return np.vstack((y1, y2, y3)).T

        # ---------- PERSISTENZA ----------

    def evaluate(self, X, y_true):
        """
        Valuta il modello su tre soglie (1.5, 2.5, 3.5)
        Restituisce un dizionario con metriche per target e macro.
        """
        y_true = np.asarray(y_true)
        y_pred = self.predict(X=X)

        def _m(y_t, y_p):
            return {
                'accuracy': float(accuracy_score(y_t, y_p)),
                'precision': float(precision_score(y_t, y_p, zero_division=0)),
                'recall': float(recall_score(y_t, y_p, zero_division=0)),
                'f1': float(f1_score(y_t, y_p, zero_division=0))
            }

        per_target = {
            'over15': _m(y_true[:, 0], y_pred[:, 0]),
            'over25': _m(y_true[:, 1], y_pred[:, 1]),
            'over35': _m(y_true[:, 2], y_pred[:, 2]),
        }

        macro = {
            'accuracy': np.mean([v['accuracy'] for v in per_target.values()]),
            'precision': np.mean([v['precision'] for v in per_target.values()]),
            'recall': np.mean([v['recall'] for v in per_target.values()]),
            'f1': np.mean([v['f1'] for v in per_target.values()]),
        }

        confusion = {
            'Confusion Over 1.5:': confusion_matrix(y_true[:, 0], y_pred[:, 0]),
            'Confusion Over 2.5:': confusion_matrix(y_true[:, 1], y_pred[:, 1]),
            'Confusion Over 3.5:': confusion_matrix(y_true[:, 2], y_pred[:, 2])
        }

        report = {
            'Report Over 1.5:': classification_report(y_true[:, 0], y_pred[:, 0]),
            'Report Over 2.5:': classification_report(y_true[:, 1], y_pred[:, 1]),
            'Report Over 3.5:': classification_report(y_true[:, 2], y_pred[:, 2])
        }

        return {'per_target': per_target,
                'macro': macro,
                'confusion_matrices': confusion,
                'classification_reports': report}

    @staticmethod
    def print_results_table(results):
        table = []

        # --- 1️⃣ Metriche per ogni target ---
        for target, metrics in results['per_target'].items():
            best_thresh = metrics['best_threshold_f1']
            if isinstance(best_thresh, dict):
                best_thresh = best_thresh.get('threshold', 0)
            f1_best = metrics['best_threshold_f1'].get('f1_at_threshold', 0)
            auc = metrics['roc_curve'].get('roc_auc', 0)
            ap = metrics['pr_curve'].get('average_precision', 0)
            cm = metrics.get('confusion_matrix', np.zeros((2, 2)))

            tn, fp, fn, tp = cm.ravel()  # Confusion matrix 2x2

            table.append([
                target,
                round(metrics['accuracy'], 3),
                round(metrics['precision'], 3),
                round(metrics['recall'], 3),
                round(metrics['f1'], 3),
                round(best_thresh, 3),
                round(f1_best, 3),
                round(auc, 3),
                round(ap, 3),
                int(tn), int(fp), int(fn), int(tp)
            ])

        # --- 2️⃣ Macro averages ---
        macro = results['macro']
        table.append([
            "MACRO AVG",
            round(macro['accuracy_macro_3targets'], 3),
            round(macro['precision_macro_3targets'], 3),
            round(macro['recall_macro_3targets'], 3),
            round(macro['f1_macro_3targets'], 3),
            "-", "-", "-", "-",
            "-", "-", "-", "-"
        ])

        headers = [
            "Target", "Accuracy", "Precision", "Recall", "F1",
            "Best Thresh (F1)", "F1@Thresh", "ROC AUC", "Avg Precision",
            "TN", "FP", "FN", "TP"
        ]
        print(tabulate(table, headers=headers, tablefmt="pretty"))

    def save(self, path):
        """
        Salva l'istanza completa (modelli interni + soglie).
        Nota: per ricaricare correttamente, il modulo che definisce la classe
        deve essere importabile con lo stesso path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """
        Carica un'istanza salvata con .save(path).
        Assicurati che il codice della classe sia disponibile con lo stesso import path.
        """
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Oggetto caricato non è {cls.__name__}")
        return obj


class FlatOverUnderClassifier(BaseHierOverUnderClassifier):
    """
    CLasse flat per il classificatore over/under con tre classificatori di base.
    1. Il secondo classificatore usa come feature anche la predizione del primo.
    2. Il terzo classificatore usa come feature anche le predizioni del primo e del secondo.
    3. Le predizioni finali sono rese coerenti gerarchicamente.
    4. Non richiede dataset separati per ciascun livello.Quindi un unico X per tutti e tre i livelli.
    """

    def fit(self, X, y):
        """
        Fit del classificatore flat.
        :param X: Input singola features.
        :param y: Target multilabel (n_samples, 3).
        :return: Self
        """
        # Fit del primo classificatore
        print(y[:, 0])
        self.clf1_ = clone(self.base_1).fit(X, y[:, 0])
        p1 = self.clf1_.predict_proba(X)[:, 1].reshape(-1, 1)

        # Fit del secondo classificatore
        print(y[:, 1])
        X2 = np.hstack((X, p1))  # Aggiunge le predizioni del primo classificatore come feature
        self.clf2_ = clone(self.base_2).fit(X2, y[:, 1])
        p2 = self.clf2_.predict_proba(X2)[:, 1].reshape(-1, 1)

        # Fit del terzo classificatore
        print(y[:, 2])
        X3 = np.hstack((X, p1, p2))  # Aggiunge le predizioni del primo e secondo classificatore come feature
        self.clf3_ = clone(self.base_3).fit(X3, y[:, 2])
        return self

    def predict_proba(self, X, **kwargs):
        """
        Predice le probabilità utilizzando il classificatore flat.
        :param X: Input features.
        :return: Probabilità predette per ciascuna delle tre classi.
        """
        only_positive = kwargs.get('only_positive', False)

        # Predizioni del primo classificatore
        p1 = self.clf1_.predict_proba(X)[:, 1].reshape(-1, 1)

        # Predizioni del secondo classificatore
        X2 = np.hstack((X, p1))
        p2 = self.clf2_.predict_proba(X2)[:, 1].reshape(-1, 1)

        # Predizioni del terzo classificatore
        X3 = np.hstack((X, p1, p2))
        p3 = self.clf3_.predict_proba(X3)[:, 1].reshape(-1, 1)

        # Assicura la coerenza gerarchica delle probabilità
        p2 = np.minimum(p2, p1)  # Assicura coerenza gerarchica
        p3 = np.minimum(p3, p2)  # Assicura coerenza gerarchica
        if only_positive:
            return np.hstack([p1, p2, p3])
        return np.hstack([1 - p1, p1, 1 - p2, p2, 1 - p3, p3])  # Restituisce le probabilità per ciascuna classe

    def cross_hierarchical(self, estimator, y, X, stratify_on=1):
        """
        Valuta HierOverUnderClassifier su 3 target (>[1.5, 2.5, 3.5]) usando 3 dataset distinti.
        - Passa EITHER X=[X1, X2, X3] OR X1,X2,X3 separatamente.
        - stratify_on: indice del target per lo split stratificato (default 2.5 -> 1)
        """

        y = np.asarray(y)
        n = y.shape[0]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        keys = ['over15', 'over25', 'over35']
        oof_pred = np.zeros_like(y, dtype=int)
        oof_proba = np.zeros_like(y, dtype=float)

        # stratifica su Y[:, stratify_on] (di solito 2.5)
        for tr, va in cv.split(X, y[:, stratify_on]):
            Xtr, Xva = X[tr], X[va]
            Ytr, Yva = y[tr], y[va]

            # fit + inferenza gerarchica con 3 dataset diversi
            est = clone(estimator).fit(X=Xtr, y=Ytr)
            P = est.predict_proba(X=Xva, only_positive=True)  # (n_va,3)
            Yhat = est.predict(X=Xva)  # (n_va,3) già coerenti

            oof_proba[va] = P
            oof_pred[va] = Yhat

        # metriche per target
        results = {'per_target': {}, 'macro': {}}
        accs, f1s, reccs, precs = [], [], [], []

        for i, k in enumerate(keys):
            y_true = y[:, i]
            y_hat = oof_pred[:, i]
            p_hat = oof_proba[:, i]

            acc = accuracy_score(y_true, y_hat)
            prec = precision_score(y_true, y_hat, zero_division=0)
            rec = recall_score(y_true, y_hat, zero_division=0)
            f1 = f1_score(y_true, y_hat, zero_division=0)
            cm = confusion_matrix(y_true, y_hat)
            rep = classification_report(y_true, y_hat, output_dict=True, zero_division=0)

            # (curva PR/ROC calcolabili se servono)
            precisions, recalls, thr = precision_recall_curve(y_true, p_hat, pos_label=stratify_on)
            ap = average_precision_score(y_true, p_hat, pos_label=stratify_on)
            fpr, tpr, thr_roc = roc_curve(y_true, p_hat, pos_label=stratify_on)
            auc = roc_auc_score(y_true, p_hat)

            # --- Soglia migliore per F1 (da PR curve) ---
            thr_best_f1, f1_at_thr = self._best_threshold_f1(y_true, p_hat)

            results['per_target'][k] = {
                'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                'confusion_matrix': cm, 'classification_report': rep,
                'pr_curve': {
                    'precisions': precisions, 'recalls': recalls,
                    'thresholds': thr, 'average_precision': ap
                },
                'roc_curve': {
                    'fpr': fpr, 'tpr': tpr,
                    'thresholds': thr_roc, 'roc_auc': auc
                },
                'best_threshold_f1': {
                    'threshold': thr_best_f1,
                    'f1_at_threshold': f1_at_thr
                }
            }
            # Youden J su ROC
            youden = tpr - fpr
            j_idx = int(np.argmax(youden))
            thr_best_j = float(thr_roc[j_idx])
            results['per_target'][k]['best_threshold_roc_youden'] = {
                'threshold': thr_best_j,
                'youden_j': float(youden[j_idx])
            }

            accs.append(acc)
            precs.append(prec)
            reccs.append(rec)
            f1s.append(f1)

        results['macro']['accuracy_macro_3targets'] = float(np.mean(accs))
        results['macro']['precision_macro_3targets'] = float(np.mean(precs))
        results['macro']['recall_macro_3targets'] = float(np.mean(reccs))
        results['macro']['f1_macro_3targets'] = float(np.mean(f1s))
        results['macro']['best_thresholds_f1'] = {
            'over15': results['per_target']['over15']['best_threshold_f1']['threshold'],
            'over25': results['per_target']['over25']['best_threshold_f1']['threshold'],
            'over35': results['per_target']['over35']['best_threshold_f1']['threshold'],
        }

        self.print_results_table(results)
        return results


class HierOverUnderClassifier(BaseHierOverUnderClassifier):
    """
    Classe gerarchica per il classificatore over/under con tre classificatori di base.
    1. Ogni classificatore usa un dataset separato.
    2. Il secondo classificatore usa come feature anche la predizione del primo.
    3. Il terzo classificatore usa come feature anche le predizioni del primo e del secondo.
    4. Le predizioni finali sono rese coerenti gerarchicamente.
    5. Richiede dataset separati per ciascun livello. Quindi X = [X1, X2, X3].
    6. Utile quando si hanno feature diverse per ciascun livello di over/under
    """

    # ---------- UTILITIES PRIVATE ----------

    @staticmethod
    def _ensure_three(X):
        if X is None or len(X) != 3:
            raise ValueError("Passa sempre X=[X1, X2, X3].")
        return X

    def _best_thresholds_on_val(self, X_va, y_va):
        P = self.predict_proba(X=X_va, only_positive=True)  # (n,3)
        thr = []
        for i in range(3):
            t_i, _ = self._best_threshold_f1(y_va[:, i], P[:, i])
            thr.append(t_i if np.isfinite(t_i) else 0.5)
        return tuple(thr)

    @staticmethod
    def count_labels(y):
        for i, name in enumerate(['over1.5', 'over2.5', 'over3.5']):
            c = Counter(y[:, i])
            n_total = len(y[:, i])
            print(f"\n{name}")
            for k, v in sorted(c.items()):
                perc = v / n_total * 100
                print(f"  classe {k}: {v} ({perc:.1f}%)")  # Esempio d’uso: tre modelli diversi

    # ---------- FIT ----------
    def fit(self, X, y):
        """
        Fit in maniera gerarchica i tre classificatori di base.
        :param X:
        :param y:
        :return: Self
        """

        # Divide X se fornito
        X1, X2, X3 = self._ensure_three(X)

        # Converte y in un array numpy
        y = np.asarray(y)
        if y.ndim != 2 or y.shape[1] != 3:
            raise ValueError("y deve essere (n_samples, 3) per [>1.5, >2.5, >3.5].")

        # Fit del primo classificatore
        self.clf1_ = clone(self.base_1).fit(X1, y[:, 0])
        p1 = self.clf1_.predict_proba(X1)[:, 1].reshape(-1, 1)
        # self.clf1_ = clone(self._cal(prefit=True, est=self.clf1_)).fit(X1, y[:, 0])

        # Fit del secondo classificatore
        X2 = np.hstack((X2, p1))  # Aggiunge le predizioni del primo classificatore come feature
        self.clf2_ = clone(self.base_2).fit(X2, y[:, 1])
        # self.clf2_ = clone(self._cal(prefit=True, est=self.clf2_)).fit(X2, y[:, 1])
        p2 = self.clf2_.predict_proba(X2)[:, 1].reshape(-1, 1)

        # Fit del terzo classificatore
        X3 = np.hstack((X3, p1, p2))  # Aggiunge le predizioni del primo e secondo classificatore come feature
        self.clf3_ = clone(self.base_3).fit(X3, y[:, 2])
        # self.clf3_ = clone(self._cal(prefit=True, est=self.clf3_)).fit(X3, y[:, 2])


        return self

    def fit_and_select_thresholds(self, X, y):
        """
        Processo unico: split train/val -> fit su train -> scelta soglie su val.
        Salva le soglie in self.thresholds e ritorna self.
        """
        y = np.asarray(y)
        if y.ndim != 2 or y.shape[1] != 3:
            raise ValueError("y deve essere (n_samples, 3).")

        # fit su train
        self.fit(X=X, y=y)

        # soglie su validation
        # best_thr = self._best_thresholds_on_val(X_va, y_va)
        # self.thresholds = best_thr
        return self

    def predict_proba(self, X, **kwargs):
        """
        Predice le probabilità utilizzando i tre classificatori in maniera gerarchica.
        :param X: Input features.
        :return: Probabilità predette per ciascuna delle tre classi.
        """
        only_positive = kwargs.get('only_positive', False)

        # Divide X se fornito
        X1, X2, X3 = self._ensure_three(X)

        # Predizioni del primo classificatore
        p1 = self.clf1_.predict_proba(X1)[:, 1].reshape(-1, 1)

        # Predizioni del secondo classificatore
        X2 = np.hstack((X2, p1))
        p2 = self.clf2_.predict_proba(X2)[:, 1].reshape(-1, 1)

        # Predizioni del terzo classificatore
        X3 = np.hstack((X3, p1, p2))
        p3 = self.clf3_.predict_proba(X3)[:, 1].reshape(-1, 1)

        # Assicura la coerenza gerarchica delle probabilità
        p2 = np.minimum(p2, p1)  # Assicura coerenza gerarchica
        p3 = np.minimum(p3, p2)  # Assicura coerenza gerarchica
        if only_positive:
            return np.hstack([p1, p2, p3])
        return np.hstack([1 - p1, p1, 1 - p2, p2, 1 - p3, p3])  # Restituisce le probabilità per ciascuna classe

    def cross_hierarchical(self, estimator, y, X, stratify_on=1):
        """
        Valuta HierOverUnderClassifier su 3 target (>[1.5, 2.5, 3.5]) usando 3 dataset distinti.
        - Passa EITHER X=[X1, X2, X3] OR X1,X2,X3 separatamente.
        - stratify_on: indice del target per lo split stratificato (default 2.5 -> 1)
        """
        X1_, X2_, X3_ = X

        y = np.asarray(y)
        n = y.shape[0]
        if not (len(X1_) == len(X2_) == len(X3_) == n):
            raise ValueError("X1, X2, X3 devono avere lo stesso numero di righe di Y.")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        keys = ['over15', 'over25', 'over35']
        oof_pred = np.zeros_like(y, dtype=int)
        oof_proba = np.zeros_like(y, dtype=float)

        # stratifica su Y[:, stratify_on] (di solito 2.5)
        for tr, va in cv.split(X2_, y[:, stratify_on]):
            # slice indipendenti per ogni dataset
            X1_tr, X1_va = X1_[tr], X1_[va]
            X2_tr, X2_va = X2_[tr], X2_[va]
            X3_tr, X3_va = X3_[tr], X3_[va]
            Ytr, Yva = y[tr], y[va]

            # fit + inferenza gerarchica con 3 dataset diversi
            est = clone(estimator).fit(X=[X1_tr, X2_tr, X3_tr], y=Ytr)
            P = est.predict_proba(X=[X1_va, X2_va, X3_va], only_positive=True)  # (n_va,3)
            Yhat = est.predict(X=[X1_va, X2_va, X3_va])  # (n_va,3) già coerenti

            oof_proba[va] = P
            oof_pred[va] = Yhat

        # metriche per target
        results = {'per_target': {}, 'macro': {}}
        accs, f1s, reccs, precs = [], [], [], []

        for i, k in enumerate(keys):
            y_true = y[:, i]
            y_hat = oof_pred[:, i]
            p_hat = oof_proba[:, i]

            acc = accuracy_score(y_true, y_hat)
            prec = precision_score(y_true, y_hat, zero_division=0)
            rec = recall_score(y_true, y_hat, zero_division=0)
            f1 = f1_score(y_true, y_hat, zero_division=0)
            cm = confusion_matrix(y_true, y_hat)
            rep = classification_report(y_true, y_hat, output_dict=True, zero_division=0)

            # (curva PR/ROC calcolabili se servono)
            precisions, recalls, thr = precision_recall_curve(y_true, p_hat, pos_label=stratify_on)
            ap = average_precision_score(y_true, p_hat, pos_label=stratify_on)
            fpr, tpr, thr_roc = roc_curve(y_true, p_hat, pos_label=stratify_on)
            auc = roc_auc_score(y_true, p_hat)

            # --- Soglia migliore per F1 (da PR curve) ---
            thr_best_f1, f1_at_thr = self._best_threshold_f1(y_true, p_hat)

            results['per_target'][k] = {
                'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                'confusion_matrix': cm, 'classification_report': rep,
                'pr_curve': {
                    'precisions': precisions, 'recalls': recalls,
                    'thresholds': thr, 'average_precision': ap
                },
                'roc_curve': {
                    'fpr': fpr, 'tpr': tpr,
                    'thresholds': thr_roc, 'roc_auc': auc
                },
                'best_threshold_f1': {
                    'threshold': thr_best_f1,
                    'f1_at_threshold': f1_at_thr
                }
            }
            # Youden J su ROC
            youden = tpr - fpr
            j_idx = int(np.argmax(youden))
            thr_best_j = float(thr_roc[j_idx])
            results['per_target'][k]['best_threshold_roc_youden'] = {
                'threshold': thr_best_j,
                'youden_j': float(youden[j_idx])
            }

            accs.append(acc)
            precs.append(prec)
            reccs.append(rec)
            f1s.append(f1)

        results['macro']['accuracy_macro_3targets'] = float(np.mean(accs))
        results['macro']['precision_macro_3targets'] = float(np.mean(precs))
        results['macro']['recall_macro_3targets'] = float(np.mean(reccs))
        results['macro']['f1_macro_3targets'] = float(np.mean(f1s))
        results['macro']['best_thresholds_f1'] = {
            'over15': results['per_target']['over15']['best_threshold_f1']['threshold'],
            'over25': results['per_target']['over25']['best_threshold_f1']['threshold'],
            'over35': results['per_target']['over35']['best_threshold_f1']['threshold'],
        }

        self.print_results_table(results)
        return results
