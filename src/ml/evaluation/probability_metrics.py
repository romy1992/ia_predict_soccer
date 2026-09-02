from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def _to_native(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_numpy(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr.reshape(-1)


def _clip_probabilities(probabilities, eps: float = 1e-12) -> np.ndarray:
    arr = _to_numpy(probabilities)
    return np.clip(arr, eps, 1.0 - eps)


def _class1_probability(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.shape[1] == 1:
        return arr[:, 0]
    return arr[:, -1]


def temporal_oof_probabilities(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[tuple[list[int], list[int]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for train_idx, valid_idx in cv_splits:
        if not train_idx or not valid_idx:
            continue
        model = clone(estimator)
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_valid)
            p1 = _class1_probability(proba)
        else:
            pred = model.predict(X_valid)
            p1 = np.asarray(pred, dtype=float).reshape(-1)

        for idx, prob, target in zip(valid_idx, p1, y_valid):
            rows.append(
                {
                    "index": int(idx),
                    "probability": float(prob),
                    "y_true": int(target),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["index", "probability", "y_true"])

    frame = pd.DataFrame(rows)
    # Se uno stesso indice compare più volte, media probabilità e mantiene target.
    grouped = (
        frame.groupby("index", as_index=False)
        .agg({"probability": "mean", "y_true": "first"})
        .sort_values(by=["index"])
        .reset_index(drop=True)
    )
    return grouped


def reliability_table(y_true, probabilities, n_bins: int = 10) -> list[dict[str, Any]]:
    y = _to_numpy(y_true).astype(int)
    p = _clip_probabilities(probabilities)
    n_bins = max(2, int(n_bins))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for idx in range(n_bins):
        left = float(edges[idx])
        right = float(edges[idx + 1])
        if idx == n_bins - 1:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)

        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": idx,
                    "lower": left,
                    "upper": right,
                    "count": 0,
                    "mean_probability": None,
                    "event_rate": None,
                    "abs_gap": None,
                }
            )
            continue

        mean_probability = float(np.mean(p[mask]))
        event_rate = float(np.mean(y[mask]))
        abs_gap = abs(mean_probability - event_rate)
        rows.append(
            {
                "bin": idx,
                "lower": left,
                "upper": right,
                "count": count,
                "mean_probability": mean_probability,
                "event_rate": event_rate,
                "abs_gap": float(abs_gap),
            }
        )
    return rows


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    rows = reliability_table(y_true=y_true, probabilities=probabilities, n_bins=n_bins)
    total = sum(int(row.get("count") or 0) for row in rows)
    if total <= 0:
        return 0.0

    ece = 0.0
    for row in rows:
        count = int(row.get("count") or 0)
        gap = row.get("abs_gap")
        if count <= 0 or gap is None:
            continue
        ece += (count / total) * float(gap)
    return float(ece)


def compute_probability_metrics(y_true, probabilities, n_bins: int = 10) -> dict[str, Any]:
    y = _to_numpy(y_true).astype(int)
    p = _clip_probabilities(probabilities)

    metrics: dict[str, Any] = {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(expected_calibration_error(y, p, n_bins=n_bins)),
        "sample_size": int(len(y)),
    }

    # AUC è secondaria e non è definita su fold mono-classe.
    if np.unique(y).size < 2:
        metrics["auc"] = None
    else:
        metrics["auc"] = float(roc_auc_score(y, p))

    metrics["reliability"] = reliability_table(y_true=y, probabilities=p, n_bins=n_bins)
    return metrics


def grouped_probability_report(
    frame: pd.DataFrame,
    probability_col: str,
    target_col: str,
    group_cols: list[str],
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    valid_groups = [col for col in group_cols if col in frame.columns]
    if not valid_groups:
        return []

    report_rows: list[dict[str, Any]] = []
    grouped = frame.groupby(valid_groups, dropna=False)
    for keys, chunk in grouped:
        if isinstance(keys, tuple):
            values = list(keys)
        else:
            values = [keys]

        row = {col: _to_native(values[idx]) for idx, col in enumerate(valid_groups)}
        metrics = compute_probability_metrics(
            y_true=chunk[target_col].astype(int).to_numpy(),
            probabilities=chunk[probability_col].astype(float).to_numpy(),
            n_bins=n_bins,
        )
        row.update(
            {
                "sample_size": metrics.get("sample_size"),
                "log_loss": metrics.get("log_loss"),
                "brier": metrics.get("brier"),
                "ece": metrics.get("ece"),
                "auc": metrics.get("auc"),
            }
        )
        report_rows.append(row)

    report_rows.sort(key=lambda item: tuple(str(item.get(col, "")) for col in valid_groups))
    return report_rows


def champion_probability_score(metrics: dict[str, Any], f1_weighted: float) -> float:
    logloss = float(metrics.get("log_loss") or 99.0)
    brier = float(metrics.get("brier") or 99.0)
    ece = float(metrics.get("ece") or 99.0)

    return float(
        (0.35 * float(f1_weighted))
        + (0.30 * (1.0 / (1.0 + logloss)))
        + (0.20 * (1.0 - min(1.0, brier)))
        + (0.15 * (1.0 - min(1.0, ece)))
    )



