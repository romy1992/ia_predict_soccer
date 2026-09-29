"""FASE 3C: confronto "indipendente" vs "cascata sequenziale" (fase 3b),
ENTRAMBE con coerenza monotona forzata in output (`enforce_monotonic_over_probabilities`,
gia' in `totals_market.py`), valutate sia a livello di probabilita' (selection_score)
sia a livello di classificazione (confusion matrix/precision/recall/F1 alla
soglia di decisione ottimale, statistica J di Youden - stessa tecnica di
`find_optimal_thresholds.py`, ricalcolata qui per ciascuna variante).

Richiesta esplicita dell'operatore (2026-09-08): "prova entrambe" (cascata E
coerenza monotona) prima di decidere. Risponde a due domande separate:
1. La cascata (gia' scartata in fase 3b sul solo selection_score) cambia
   verdetto se si guarda anche la classificazione a soglia ottimale?
2. Quanto conta la proiezione di coerenza monotona da sola (che NON richiede
   retraining, si applica a QUALSIASI approccio) - e quante violazioni
   correggerebbe davvero sui dati reali?

Dataset: stesso frame allineato di phase3b (`totals_evaluation_frame.csv`,
feature basate sulle quote 'under_over_2_5' come riferimento, STESSE righe
per tutte e 4 le soglie - necessario per parlare di coerenza tra soglie sulla
stessa partita, a differenza dei 4 modelli indipendenti gia' registrati che
usano ciascuno le proprie quote/feature e quindi righe diverse).
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve

from src.ml.evaluation.probability_metrics import champion_probability_score, compute_probability_metrics
from src.ml.markets.totals.totals_market import (
    THRESHOLD_LABELS,
    THRESHOLDS,
    _binary_independent_oof,
    _count_monotonicity_violations,
    _threshold_label,
    enforce_monotonic_over_probabilities,
)
from src.ml.validation.temporal_split import expanding_window_splits
from sklearn.metrics import f1_score

from scripts.analysis.phase3b_sequential_stacking_thresholds import _assert_cascade_no_future_leakage, _cascade_oof

EXPORT_DIR = os.path.join("scripts", "analysis", "_export")
OUTPUT_PATH = os.path.join("best_models", "phase3c_coherence_experiment_result.json")


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
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    accuracy = float((y_true == y_pred).mean())
    return {
        "threshold": round(threshold, 4),
        "accuracy": round(accuracy, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "class_0_under": {"precision": round(float(precision[0]), 4), "recall": round(float(recall[0]), 4), "f1": round(float(f1[0]), 4)},
        "class_1_over": {"precision": round(float(precision[1]), 4), "recall": round(float(recall[1]), 4), "f1": round(float(f1[1]), 4)},
    }


def build_cascade_probs(frame: pd.DataFrame, feature_columns: list[str], cv_splits) -> dict[str, np.ndarray]:
    """Stessa logica di run_sequential_cascade (phase3b), ma ritorna gli array
    OOF grezzi per soglia invece delle sole metriche aggregate."""
    oof_index_all = sorted({idx for _, valid_idx in cv_splits for idx in valid_idx})
    oof_index_set = set(oof_index_all)
    _assert_cascade_no_future_leakage(frame=frame, cv_splits=cv_splits, oof_index_all=oof_index_set)

    working_frame = frame.copy()
    probs_by_threshold: dict[str, np.ndarray] = {}
    prev_oof: Optional[np.ndarray] = None
    prev_label: Optional[str] = None

    for threshold in sorted(set(float(t) for t in THRESHOLDS)):
        label = _threshold_label(threshold)
        y_col = f"y_{label}"
        if prev_oof is None:
            probs = _cascade_oof(working_frame, feature_columns, cv_splits, y_col, restrict_train_to=None)
        else:
            eligible_train = {int(i) for i in oof_index_all if not np.isnan(prev_oof[i])}
            cascade_feature_col = f"cascade_oof_{prev_label}"
            working_frame[cascade_feature_col] = prev_oof
            probs = _cascade_oof(working_frame, feature_columns + [cascade_feature_col], cv_splits, y_col, restrict_train_to=eligible_train)
        probs_by_threshold[label] = probs
        prev_oof = probs
        prev_label = label

    return probs_by_threshold


def main() -> None:
    t0 = time.time()
    frame = pd.read_csv(os.path.join(EXPORT_DIR, "totals_evaluation_frame.csv"))
    with open(os.path.join(EXPORT_DIR, "totals_feature_columns.json"), "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    min_train = max(30, int(len(frame) * 0.45))
    min_valid = max(10, int(len(frame) * 0.1))
    cv_splits = expanding_window_splits(frame=frame, time_col="prediction_at", n_splits=5, min_train_size=min_train, min_valid_size=min_valid)
    oof_index_all = sorted({idx for _, valid_idx in cv_splits for idx in valid_idx})

    print(f"rows={len(frame)} | oof_index_all={len(oof_index_all)}", flush=True)

    independent_probs = _binary_independent_oof(frame, feature_columns, cv_splits, thresholds=THRESHOLDS)
    cascade_probs = build_cascade_probs(frame, feature_columns, cv_splits)
    print("OOF calcolati per entrambe le varianti", flush=True)

    # Popolazione comune per il check di coerenza cross-soglia: intersezione
    # dei domini non-NaN di TUTTE le 4 soglie, PER ENTRAMBE le varianti (la
    # cascata e' quella che restringe di piu' - vedi fase 3b).
    common_index = set(oof_index_all)
    for probs_by_threshold in (independent_probs, cascade_probs):
        for label in THRESHOLD_LABELS:
            valid = {i for i in oof_index_all if not np.isnan(probs_by_threshold[label][i])}
            common_index &= valid
    common_index = sorted(common_index)
    print(f"popolazione comune (tutte e 4 le soglie, entrambe le varianti) = {len(common_index)} righe", flush=True)

    results: dict = {"n_common_rows": len(common_index), "variants": {}}

    for variant_name, probs_by_threshold in (("independent", independent_probs), ("cascade", cascade_probs)):
        restricted = {label: np.asarray(probs_by_threshold[label], dtype=float)[common_index] for label in THRESHOLD_LABELS}
        violations_before = _count_monotonicity_violations(restricted, thresholds=THRESHOLDS)
        projected = enforce_monotonic_over_probabilities(restricted, thresholds=THRESHOLDS)

        per_threshold = {}
        agg_before = []
        agg_after = []
        for label in THRESHOLD_LABELS:
            y_true = frame[f"y_{label}"].astype(int).to_numpy()[common_index]
            before = _selection_score(y_true, restricted[label])
            after = _selection_score(y_true, projected[label])
            youden_after = _youden_classification(y_true, projected[label])
            per_threshold[label] = {"before_projection": before, "after_projection": after, "youden_after_projection": youden_after}
            agg_before.append(before["selection_score"])
            agg_after.append(after["selection_score"])

        results["variants"][variant_name] = {
            "monotonicity_violations_before_projection": violations_before,
            "monotonicity_violation_rate": round(violations_before / (len(common_index) * 3), 4),
            "aggregate_selection_score_before": round(float(np.mean(agg_before)), 4),
            "aggregate_selection_score_after": round(float(np.mean(agg_after)), 4),
            "per_threshold": per_threshold,
        }
        print(f"\n=== {variant_name} ===")
        print(f"violazioni monotonicita' prima della proiezione: {violations_before} / {len(common_index)*3} confronti possibili "
              f"({results['variants'][variant_name]['monotonicity_violation_rate']*100:.1f}%)")
        print(f"selection_score aggregato: prima={results['variants'][variant_name]['aggregate_selection_score_before']:.4f} "
              f"dopo proiezione={results['variants'][variant_name]['aggregate_selection_score_after']:.4f}")
        for label in THRESHOLD_LABELS:
            ya = per_threshold[label]["youden_after_projection"]
            print(f"  {label}: soglia_ottimale={ya['threshold']:.3f} accuracy={ya['accuracy']:.3f} "
                  f"under(P={ya['class_0_under']['precision']:.3f}/R={ya['class_0_under']['recall']:.3f}) "
                  f"over(P={ya['class_1_over']['precision']:.3f}/R={ya['class_1_over']['recall']:.3f})")

    print(f"\nelapsed={time.time()-t0:.1f}s")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"Salvato in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
