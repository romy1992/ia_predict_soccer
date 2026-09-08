"""Metriche di classificazione dettagliate (precision/recall/F1 per classe,
confusion matrix, curva ROC+AUC, curva PR) per i 4 modelli 'champion' gia'
salvati in best_models/ (Under/Over 1.5/2.5/3.5/4.5).

Le metriche aggregate salvate in ModelRegistry (log_loss/brier/ece/auc/
selection_score) non includono queste - questo script le rigenera via
walk-forward OOF (identico a `temporal_oof_probabilities` gia' in uso in
`train_multi_market.py`/`CalibrationService`): `clone()` del champion.pkl gia'
addestrato (preserva l'intera architettura/iperparametri, incluso il wrapper
di calibrazione, senza bisogno di rifare la grid search) rifittato fold per
fold sullo stesso walk-forward espandente usato in produzione - MAI valutato
sui dati di training, stessa garanzia anti-leakage.

Richiede i CSV esportati (vedi export_datasets_for_cloud_training.py) nella
directory passata via --export-dir.
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from src.service_ia.training.train_multi_market import _build_temporal_cv, _filter_valid_splits

MARKETS = ["under_over_1_5", "under_over_2_5", "under_over_3_5", "under_over_4_5"]


def _class1_probability(raw_proba: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw_proba, dtype=float)
    return arr[:, 1] if arr.ndim == 2 and arr.shape[1] == 2 else arr.reshape(-1)


def evaluate_market(market: str, export_dir: str, models_dir: str) -> dict:
    df = pd.read_csv(os.path.join(export_dir, f"{market}.csv"))
    df["prediction_at"] = pd.to_datetime(df["prediction_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)

    y = df["y"].astype(int)
    X = df.drop(columns=["y", "market"], errors="ignore")
    X = X.drop(columns=["id_fixture", "season", "league", "prediction_at"], errors="ignore")

    raw_splits = _build_temporal_cv(df)
    cv_splits = _filter_valid_splits(y=y, splits=raw_splits)

    champion_path = os.path.join(models_dir, f"{market}_champion.pkl")
    champion = joblib.load(champion_path)

    n = len(df)
    oof_proba = np.full(n, np.nan)
    for train_idx, valid_idx in cv_splits:
        if not train_idx or not valid_idx or y.iloc[train_idx].nunique() < 2:
            continue
        model = clone(champion)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[valid_idx])
        oof_proba[valid_idx] = _class1_probability(proba)

    oof_index = [i for i in range(n) if not np.isnan(oof_proba[i])]
    y_true = y.to_numpy()[oof_index]
    p1 = oof_proba[oof_index]
    y_pred = (p1 >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    accuracy = float((y_true == y_pred).mean())

    fpr, tpr, roc_thresholds = roc_curve(y_true, p1)
    auc = float(roc_auc_score(y_true, p1))
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, p1)

    return {
        "market": market,
        "n_oof": len(oof_index),
        "champion_model_family": type(champion).__name__,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy": accuracy,
        "per_class": {
            "class_0_under": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0]), "support": int(support[0])},
            "class_1_over": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "support": int(support[1])},
        },
        "weighted_avg": {"precision": float(precision_w), "recall": float(recall_w), "f1": float(f1_w)},
        "macro_avg": {"precision": float(precision_m), "recall": float(recall_m), "f1": float(f1_m)},
        "roc_auc": auc,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresholds.tolist()},
        "pr_curve": {"precision": pr_precision.tolist(), "recall": pr_recall.tolist(), "thresholds": pr_thresholds.tolist()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", default=os.path.join("scripts", "analysis", "_export"))
    parser.add_argument("--models-dir", default="best_models")
    parser.add_argument("--output", default=os.path.join("best_models", "champions_detailed_metrics.json"))
    parser.add_argument("--markets", default=",".join(MARKETS), help="Lista mercati separati da virgola")
    parser.add_argument("--merge", action="store_true", help="Unisce ai risultati gia' presenti in --output invece di sovrascriverli")
    args = parser.parse_args()

    markets = [m.strip() for m in args.markets.split(",") if m.strip()]

    results = {}
    if args.merge and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            results = json.load(f)

    for market in markets:
        print(f"=== {market} ===", flush=True)
        result = evaluate_market(market=market, export_dir=args.export_dir, models_dir=args.models_dir)
        results[market] = result
        cm = result["confusion_matrix"]
        print(f"n_oof={result['n_oof']} | champion={result['champion_model_family']} | accuracy={result['accuracy']:.4f} | auc={result['roc_auc']:.4f}")
        print(f"confusion_matrix: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
        print(f"class 0 (under): precision={result['per_class']['class_0_under']['precision']:.4f} recall={result['per_class']['class_0_under']['recall']:.4f} f1={result['per_class']['class_0_under']['f1']:.4f}")
        print(f"class 1 (over) : precision={result['per_class']['class_1_over']['precision']:.4f} recall={result['per_class']['class_1_over']['recall']:.4f} f1={result['per_class']['class_1_over']['f1']:.4f}")
        print(f"weighted avg   : precision={result['weighted_avg']['precision']:.4f} recall={result['weighted_avg']['recall']:.4f} f1={result['weighted_avg']['f1']:.4f}\n", flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Salvato in {args.output}")


if __name__ == "__main__":
    main()
