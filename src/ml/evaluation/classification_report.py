"""Report di classificazione COMPLETO (2026-09-12) per un array di
probabilita' OOF - richiesto esplicitamente dall'operatore ("vorrei avere
tutte le metriche possibili") per il benchmark multi-linea di Corners/Cards
(MARKET-05/06): confusion matrix, precision/recall/F1 per classe e pesati,
curva ROC (+ AUC), curva Precision-Recall (+ average precision) e metriche
alla soglia OTTIMALE (statistica J di Youden, stesso criterio gia' usato per
gli Under/Over in `scripts/analysis/find_optimal_thresholds.py`), tutte
insieme in una singola chiamata.

Complementare, MAI un sostituto, di `compute_probability_metrics` (che copre
SOLO le metriche di qualita' della probabilita' - log loss/Brier/ECE/AUC,
indipendenti da qualunque soglia di decisione, ed e' usato ovunque nel
progetto per la calibrazione/il gate di promozione): quella funzione NON e'
stata toccata qui per non allargare il perimetro a tutti gli altri mercati
che la usano (h2h, goal_no_goal, under_over_*, ...) - questo modulo e' nuovo
e additivo, per ora usato solo da corners_market.py/cards_market.py."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def _threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Accuracy/confusion matrix/precision/recall/F1 per classe + pesati,
    per un vettore di predizioni GIA' binarizzate a una soglia qualunque."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "class0": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "class1": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "weighted": {"precision": float(precision_w), "recall": float(recall_w), "f1": float(f1_w)},
    }


def compute_full_classification_report(y_true, probabilities, threshold: float = 0.5) -> dict[str, Any]:
    """Report completo per un array di probabilita' OOF binarie.

    Restituisce SEMPRE un dict con `status`: "ok" quando entrambe le classi
    sono osservate nell'OOF (confusion matrix/ROC/PR calcolabili), oppure
    "single_class" quando non lo sono (stesso principio "mai un valore
    inventato" gia' seguito da `model_diagnostics_service.py`: il chiamante
    decide come comunicare l'assenza di dato)."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)

    if np.unique(y).size < 2:
        return {"status": "single_class", "threshold": float(threshold), "sample_size": int(len(y))}

    y_pred = (p >= threshold).astype(int)
    report: dict[str, Any] = {
        "status": "ok",
        "sample_size": int(len(y)),
        "threshold": float(threshold),
        **_threshold_metrics(y, y_pred),
    }

    fpr, tpr, roc_thresholds = roc_curve(y, p)
    report["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc_score(y, p))}

    pr_precision, pr_recall, _ = precision_recall_curve(y, p)
    report["pr_curve"] = {
        "precision": pr_precision.tolist(),
        "recall": pr_recall.tolist(),
        "average_precision": float(average_precision_score(y, p)),
    }

    # Soglia ottimale (J di Youden = argmax(tpr - fpr)) calcolata sulla STESSA
    # curva ROC gia' ottenuta sopra, cosi' il report e' autosufficiente senza
    # dover lanciare uno script di analisi separato per ogni linea/mercato.
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    # roc_curve antepone sempre una soglia sentinella (thresholds[0] = inf o
    # max(p)+1): clip a [0, 1] per restituire una soglia di decisione valida.
    optimal_threshold = float(np.clip(roc_thresholds[best_idx], 0.0, 1.0))
    y_pred_opt = (p >= optimal_threshold).astype(int)
    report["optimal_threshold"] = {"threshold": optimal_threshold, **_threshold_metrics(y, y_pred_opt)}

    return report
