"""Applica `enforce_monotonic_over_probabilities` (gia' in totals_market.py)
alle predizioni OOF dei 4 champion REALMENTE registrati in ModelRegistry
(under_over_1_5/2_5/3_5/4_5, gia' addestrati e salvati - vedi
evaluate_champions_detailed.py per la stessa metodologia OOF walk-forward).

A differenza di phase3c_coherence_experiment.py (che confrontava indipendente
vs cascata su un frame di feature CONDIVISO/ricostruito ad-hoc, 15243 righe),
qui si usano i 4 champion cosi' come sono stati registrati - ciascuno con le
proprie feature/quote di mercato, quindi ciascuno con la propria copertura di
fixture. La proiezione monotona richiede le 4 probabilita' sulla STESSA
fixture: si lavora quindi sull'intersezione degli id_fixture con OOF valido
su tutti e 4 i mercati (tipicamente piu' piccola della somma - allineamento
via id_fixture, non per posizione).

Richiede i CSV esportati (vedi export_datasets_for_cloud_training.py) nella
directory passata via --export-dir e i 4 *_champion.pkl in --models-dir.
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_curve

from src.ml.evaluation.probability_metrics import champion_probability_score, compute_probability_metrics
from src.ml.markets.totals.totals_market import (
    THRESHOLD_LABELS,
    THRESHOLDS,
    _count_monotonicity_violations,
    enforce_monotonic_over_probabilities,
)
from src.service_ia.training.train_multi_market import _build_temporal_cv, _filter_valid_splits

MARKETS = ["under_over_1_5", "under_over_2_5", "under_over_3_5", "under_over_4_5"]
MARKET_TO_LABEL = {"under_over_1_5": "over_1_5", "under_over_2_5": "over_2_5", "under_over_3_5": "over_3_5", "under_over_4_5": "over_4_5"}


def _class1_probability(raw_proba: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw_proba, dtype=float)
    return arr[:, 1] if arr.ndim == 2 and arr.shape[1] == 2 else arr.reshape(-1)


def compute_oof_by_fixture(market: str, export_dir: str, models_dir: str) -> tuple[dict[int, float], dict[int, int]]:
    df = pd.read_csv(os.path.join(export_dir, f"{market}.csv"))
    df["prediction_at"] = pd.to_datetime(df["prediction_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)

    y = df["y"].astype(int)
    fixture_ids = df["id_fixture"].astype(int)
    X = df.drop(columns=["y", "market"], errors="ignore")
    X = X.drop(columns=["id_fixture", "season", "league", "prediction_at"], errors="ignore")

    raw_splits = _build_temporal_cv(df)
    cv_splits = _filter_valid_splits(y=y, splits=raw_splits)

    champion = joblib.load(os.path.join(models_dir, f"{market}_champion.pkl"))

    n = len(df)
    oof_proba = np.full(n, np.nan)
    for train_idx, valid_idx in cv_splits:
        if not train_idx or not valid_idx or y.iloc[train_idx].nunique() < 2:
            continue
        model = clone(champion)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[valid_idx])
        oof_proba[valid_idx] = _class1_probability(proba)

    proba_by_fixture: dict[int, float] = {}
    y_by_fixture: dict[int, int] = {}
    for idx in range(n):
        if not np.isnan(oof_proba[idx]):
            fid = int(fixture_ids.iloc[idx])
            proba_by_fixture[fid] = float(oof_proba[idx])
            y_by_fixture[fid] = int(y.iloc[idx])
    return proba_by_fixture, y_by_fixture


def _selection_score(y_true: np.ndarray, p1: np.ndarray) -> dict:
    metrics = compute_probability_metrics(y_true=y_true, probabilities=p1, n_bins=10)
    predicted = (p1 >= 0.5).astype(int)
    f1_weighted = float(f1_score(y_true, predicted, average="weighted", zero_division=0))
    score = champion_probability_score(metrics=metrics, f1_weighted=f1_weighted)
    return {**metrics, "f1_weighted": f1_weighted, "selection_score": score}


def _youden_classification(y_true: np.ndarray, p1: np.ndarray) -> dict:
    fpr, tpr, th = roc_curve(y_true, p1)
    j = tpr - fpr
    idx = int(np.argmax(j))
    threshold = float(th[idx])
    y_pred = (p1 >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    accuracy = float((y_true == y_pred).mean())
    return {
        "threshold": round(threshold, 4),
        "accuracy": round(accuracy, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "class_0_under": {"precision": round(float(precision[0]), 4), "recall": round(float(recall[0]), 4), "f1": round(float(f1[0]), 4)},
        "class_1_over": {"precision": round(float(precision[1]), 4), "recall": round(float(recall[1]), 4), "f1": round(float(f1[1]), 4)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", default=os.path.join("scripts", "analysis", "_export"))
    parser.add_argument("--models-dir", default="best_models")
    parser.add_argument("--output", default=os.path.join("best_models", "champions_monotonic_result.json"))
    args = parser.parse_args()

    proba_by_market: dict[str, dict[int, float]] = {}
    y_by_market: dict[str, dict[int, int]] = {}
    for market in MARKETS:
        print(f"=== OOF {market} ===", flush=True)
        proba_by_market[market], y_by_market[market] = compute_oof_by_fixture(market, args.export_dir, args.models_dir)
        print(f"  fixture con OOF valido: {len(proba_by_market[market])}", flush=True)

    common_fixtures = set(proba_by_market[MARKETS[0]].keys())
    for market in MARKETS[1:]:
        common_fixtures &= set(proba_by_market[market].keys())
    common_fixtures = sorted(common_fixtures)
    print(f"\nfixture comuni a tutti e 4 i mercati: {len(common_fixtures)}", flush=True)

    raw = {}
    y_true_by_label = {}
    for market in MARKETS:
        label = MARKET_TO_LABEL[market]
        raw[label] = np.array([proba_by_market[market][fid] for fid in common_fixtures])
        y_true_by_label[label] = np.array([y_by_market[market][fid] for fid in common_fixtures])

    violations_before = _count_monotonicity_violations(raw, thresholds=THRESHOLDS)
    n_comparisons = len(common_fixtures) * 3
    print(f"violazioni monotonicita' prima della proiezione: {violations_before} / {n_comparisons} "
          f"({100*violations_before/n_comparisons:.1f}%)", flush=True)

    projected = enforce_monotonic_over_probabilities(raw, thresholds=THRESHOLDS)
    violations_after = _count_monotonicity_violations(projected, thresholds=THRESHOLDS)
    print(f"violazioni monotonicita' dopo la proiezione: {violations_after} / {n_comparisons} (deve essere 0)", flush=True)

    results = {"n_common_fixtures": len(common_fixtures), "monotonicity_violations_before": violations_before,
               "monotonicity_violations_after": violations_after, "n_comparisons": n_comparisons, "per_market": {}}

    print()
    for label in THRESHOLD_LABELS:
        y_true = y_true_by_label[label]
        before = _selection_score(y_true, raw[label])
        after = _selection_score(y_true, projected[label])
        youden_before = _youden_classification(y_true, raw[label])
        youden_after = _youden_classification(y_true, projected[label])
        results["per_market"][label] = {
            "selection_score_before_projection": before,
            "selection_score_after_projection": after,
            "youden_before_projection": youden_before,
            "youden_after_projection": youden_after,
        }
        print(f"=== {label} ===")
        print(f"selection_score: prima={before['selection_score']:.4f} dopo={after['selection_score']:.4f}")
        print(f"soglia ottimale (Youden) DOPO proiezione: {youden_after['threshold']:.3f} | accuracy={youden_after['accuracy']:.3f}")
        print(f"  under: precision={youden_after['class_0_under']['precision']:.3f} recall={youden_after['class_0_under']['recall']:.3f} f1={youden_after['class_0_under']['f1']:.3f}")
        print(f"  over : precision={youden_after['class_1_over']['precision']:.3f} recall={youden_after['class_1_over']['recall']:.3f} f1={youden_after['class_1_over']['f1']:.3f}")
        print()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Salvato in {args.output}")


if __name__ == "__main__":
    main()
