"""Metriche probabilistiche per classificazione MULTICLASSE (MARKET-01).

Le funzioni in `probability_metrics.py` sono scritte per classificazione
BINARIA (log_loss con labels=[0, 1], brier_score_loss, roc_auc_score
scalare) e sono gia' usate in produzione dai mercati "diretti" esistenti
(h2h binario, under/over, ecc., vedi EXP-05): NON vengono modificate qui,
per non introdurre regressioni. Questo modulo aggiunge l'equivalente
multiclasse, necessario per il vero 1X2 (HOME/DRAW/AWAY):

- log-loss multiclasse nativo (sklearn supporta piu' di 2 classi);
- Brier score generalizzato (Brier, 1950): media della somma dei quadrati
  delle differenze tra probabilita' predetta e one-hot del vero valore
  (in [0, 2], normalizzato a [0, 1] solo nel composite score);
- ECE basato sulla "confidence" (probabilita' della classe predetta) vs
  accuratezza nel bin: approccio standard per il multiclasse
  (Guo et al., 2017), la reliability curve binaria classica non si applica
  direttamente a 3+ classi;
- AUC one-vs-rest (macro), disponibile solo quando tutte le classi sono
  presenti nel campione valutato (altrimenti None, mai un valore inventato).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss, roc_auc_score


def _class_index_map(class_labels: Sequence[str]) -> dict[str, int]:
    return {str(label): idx for idx, label in enumerate(class_labels)}


def reorder_probabilities_to_labels(
    estimator_classes: Sequence[Any],
    raw_probabilities: np.ndarray,
    class_labels: Sequence[str],
) -> np.ndarray:
    """Rimappa le colonne di `predict_proba` nell'ordine canonico `class_labels`.

    sklearn ordina `estimator.classes_` alfabeticamente per le stringhe: NON
    si puo' assumere che coincida con l'ordine desiderato in output (qui
    HOME/DRAW/AWAY), quindi si rimappa sempre esplicitamente per indice
    invece di assumere posizioni fisse.
    """
    raw = np.asarray(raw_probabilities, dtype=float)
    target_index = _class_index_map(class_labels)
    ordered = np.zeros((raw.shape[0], len(class_labels)), dtype=float)
    for col_idx, label in enumerate(estimator_classes):
        target_col = target_index.get(str(label))
        if target_col is None:
            continue
        ordered[:, target_col] = raw[:, col_idx]
    return ordered


def temporal_oof_multiclass_probabilities(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[tuple[list[int], list[int]]],
    class_labels: Sequence[str],
) -> pd.DataFrame:
    """Equivalente multiclasse di `temporal_oof_probabilities` (ML-02/ML-05).

    Stesso principio del framework binario esistente: ogni fold viene
    addestrato SOLO sul passato (`cv_splits` deve provenire da
    `expanding_window_splits`/`rolling_window_splits`, mai da split random).
    """
    rows: list[dict[str, Any]] = []
    prob_cols = [f"prob_{label}" for label in class_labels]

    for train_idx, valid_idx in cv_splits:
        if not train_idx or not valid_idx:
            continue
        model = clone(estimator)
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        if y_train.nunique() < 2:
            continue

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_valid)
        ordered = reorder_probabilities_to_labels(model.classes_, proba, class_labels)

        for row_pos, idx in enumerate(valid_idx):
            row: dict[str, Any] = {"index": int(idx), "y_true": str(y_valid.iloc[row_pos])}
            for col_idx, col_name in enumerate(prob_cols):
                row[col_name] = float(ordered[row_pos, col_idx])
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["index", "y_true", *prob_cols])

    frame = pd.DataFrame(rows)
    agg = {col: "mean" for col in prob_cols}
    agg["y_true"] = "first"
    return (
        frame.groupby("index", as_index=False)
        .agg(agg)
        .sort_values(by=["index"])
        .reset_index(drop=True)
    )


def multiclass_brier_score(y_true_idx: np.ndarray, probabilities: np.ndarray) -> float:
    """Brier score generalizzato (Brier, 1950) per problemi multiclasse. Range [0, 2]."""
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true_idx)), y_true_idx] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def confidence_expected_calibration_error(
    y_true_idx: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE su 'confidence' (probabilita' della classe predetta) vs accuratezza nel bin."""
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == y_true_idx).astype(float)

    total = len(confidences)
    if total == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, max(2, int(n_bins)) + 1)
    ece = 0.0
    for idx in range(len(edges) - 1):
        left, right = edges[idx], edges[idx + 1]
        if idx == len(edges) - 2:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_confidence = float(np.mean(confidences[mask]))
        bin_accuracy = float(np.mean(correct[mask]))
        ece += (count / total) * abs(bin_confidence - bin_accuracy)
    return float(ece)


def compute_multiclass_probability_metrics(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    class_labels: Sequence[str],
    n_bins: int = 10,
) -> dict[str, Any]:
    labels = [str(label) for label in class_labels]
    label_index = _class_index_map(labels)
    y_idx = np.asarray([label_index[str(value)] for value in y_true], dtype=int)

    proba = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)  # rinormalizza dopo il clip

    metrics: dict[str, Any] = {
        "log_loss": float(log_loss(y_idx, proba, labels=list(range(len(labels))))),
        "brier": multiclass_brier_score(y_idx, proba),
        "ece": confidence_expected_calibration_error(y_idx, proba, n_bins=n_bins),
        "sample_size": int(len(y_idx)),
        "classes": labels,
    }

    present_classes = np.unique(y_idx)
    if present_classes.size < len(labels):
        metrics["auc_ovr_macro"] = None
    else:
        try:
            metrics["auc_ovr_macro"] = float(
                roc_auc_score(y_idx, proba, multi_class="ovr", average="macro", labels=list(range(len(labels))))
            )
        except ValueError:
            metrics["auc_ovr_macro"] = None

    return metrics


def multiclass_champion_score(metrics: dict[str, Any], f1_weighted: float) -> float:
    """Composite score coerente con `champion_probability_score` (ML-05), versione multiclasse."""
    logloss = float(metrics.get("log_loss") or 99.0)
    brier = float(metrics.get("brier") or 99.0)  # range [0, 2]
    ece = float(metrics.get("ece") or 99.0)

    return float(
        (0.35 * float(f1_weighted))
        + (0.30 * (1.0 / (1.0 + logloss)))
        + (0.20 * (1.0 - min(1.0, brier / 2.0)))
        + (0.15 * (1.0 - min(1.0, ece)))
    )

