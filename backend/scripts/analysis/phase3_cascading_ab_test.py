"""FASE 3 (cascading): A/B test mirato SOLO sulla soglia 2.5, dove il
confronto reale (Fase 1+2, vedi `phase1_2_totals_benchmark_result.json`) ha
mostrato che 'hierarchical' perde sistematicamente contro 'binary_independent'
(selection_score 0.6074 vs 0.6683 - unico caso tra le 4 soglie).

Esperimento: un classificatore binario per over_2_5 con LE STESSE feature di
'binary_independent' PIU' una feature aggiuntiva `gd_over_1_5` (P(Over 1.5)
dal Goal Distribution Expert, EXP-02 - calcolo deterministico da rating
point-in-time, quindi gia' 'out-of-fold' per costruzione: non c'e' training
supervisionato che potrebbe overfittare in-sample). Stesso `cv_splits`
walk-forward, stesso RandomForest (`_RF_KWARGS`) di `totals_market.py` per un
confronto onesto (nessuna modifica alla logica di monotonicita' esistente:
questo script e' puramente di ANALISI, non tocca `totals_market.py`).

Se il selection_score con la feature cascading NON supera nettamente sia
'binary_independent' puro sia 'hierarchical', l'adozione viene scartata e
l'esito (anche negativo) viene documentato.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logging.basicConfig(level=logging.WARNING)

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.ml.evaluation.probability_metrics import champion_probability_score, compute_probability_metrics
from src.ml.markets.totals.totals_market import (
    _RF_KWARGS,
    _class1_probability,
    build_totals_evaluation_frame,
)
from src.ml.validation.temporal_split import expanding_window_splits
from src.repository.match_repository import MatchRepository
from src.service_ia.utility.utils import convert_orm_match_to_dict

OUTPUT_PATH = os.path.abspath(os.path.join("best_models", "phase3_cascading_ab_result.json"))
TARGET_LABEL = "over_2_5"
CASCADING_FEATURE = "gd_over_1_5"


def _binary_oof_with_features(frame, feature_columns, cv_splits, y_col: str) -> np.ndarray:
    n = len(frame)
    result = np.full(n, np.nan)
    X = frame[feature_columns]
    y = frame[y_col].astype(int)
    for train_idx, valid_idx in cv_splits:
        if not train_idx or not valid_idx or y.iloc[train_idx].nunique() < 2:
            continue
        model = RandomForestClassifier(**_RF_KWARGS)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[valid_idx])
        result[valid_idx] = _class1_probability(proba, model.classes_)
    return result


def _score(frame, y_col: str, probs: np.ndarray, oof_index: list[int]) -> dict[str, Any]:
    y_true = frame[y_col].astype(int).to_numpy()[oof_index]
    p = probs[oof_index]
    metrics = compute_probability_metrics(y_true=y_true, probabilities=p, n_bins=10)
    predicted = (p >= 0.5).astype(int)
    f1_weighted = float(f1_score(y_true, predicted, average="weighted", zero_division=0))
    selection_score = champion_probability_score(metrics=metrics, f1_weighted=f1_weighted)
    return {**metrics, "f1_weighted": f1_weighted, "selection_score": selection_score}


def main() -> None:
    t0 = time.time()
    match_repo = MatchRepository()
    filters = {"statistics": "not None", "mean_statistics": "not None", "odds": "not None", "status": ["FT"]}
    matches = convert_orm_match_to_dict(match_repo.search_filter(filters=filters))
    print(f"matches caricati: {len(matches)} ({time.time() - t0:.1f}s)")

    frame, feature_columns = build_totals_evaluation_frame(matches=matches)
    if frame.empty:
        print("Dataset vuoto, esperimento non eseguibile.")
        return
    print(f"rows={len(frame)} | feature_columns={len(feature_columns)} | ha '{CASCADING_FEATURE}': {CASCADING_FEATURE in frame.columns}")

    min_train = max(30, int(len(frame) * 0.45))
    min_valid = max(10, int(len(frame) * 0.1))
    cv_splits = expanding_window_splits(frame=frame, time_col="prediction_at", n_splits=5, min_train_size=min_train, min_valid_size=min_valid)
    if not cv_splits:
        print("CV insufficiente, esperimento non eseguibile.")
        return

    oof_index_all = sorted({idx for _, valid_idx in cv_splits for idx in valid_idx})

    y_col = f"y_{TARGET_LABEL}"

    t1 = time.time()
    baseline_probs = _binary_oof_with_features(frame, feature_columns, cv_splits, y_col=y_col)
    print(f"baseline (binary_independent puro) calcolato ({time.time() - t1:.1f}s)")

    t2 = time.time()
    cascading_feature_columns = feature_columns + [CASCADING_FEATURE]
    cascading_probs = _binary_oof_with_features(frame, cascading_feature_columns, cv_splits, y_col=y_col)
    print(f"cascading (+ {CASCADING_FEATURE}) calcolato ({time.time() - t2:.1f}s)")

    valid_mask = ~np.isnan(baseline_probs[oof_index_all]) & ~np.isnan(cascading_probs[oof_index_all])
    oof_index = [idx for idx, keep in zip(oof_index_all, valid_mask) if keep]
    if not oof_index:
        print("Nessuna riga OOF valida per il confronto.")
        return

    baseline_metrics = _score(frame, y_col, baseline_probs, oof_index)
    cascading_metrics = _score(frame, y_col, cascading_probs, oof_index)

    delta = cascading_metrics["selection_score"] - baseline_metrics["selection_score"]
    print(f"\n=== A/B su {TARGET_LABEL} (n_oof={len(oof_index)}) ===")
    print(f"binary_independent puro   : selection_score={baseline_metrics['selection_score']:.4f} "
          f"log_loss={baseline_metrics['log_loss']:.4f} brier={baseline_metrics['brier']:.4f} ece={baseline_metrics['ece']:.4f}")
    print(f"binary_independent + {CASCADING_FEATURE}: selection_score={cascading_metrics['selection_score']:.4f} "
          f"log_loss={cascading_metrics['log_loss']:.4f} brier={cascading_metrics['brier']:.4f} ece={cascading_metrics['ece']:.4f}")
    print(f"delta selection_score (cascading - baseline) = {delta:+.4f}")

    # Confronto anche con hierarchical puro (gia' calcolato in Fase 1+2, riportato qui per riferimento)
    hierarchical_score_phase12 = 0.6074  # da phase1_2_totals_benchmark_result.json, over_2_5/hierarchical

    verdict = (
        "ADOTTATO: la feature cascading migliora selection_score sia rispetto a binary_independent puro sia a hierarchical."
        if (delta > 0.01 and cascading_metrics["selection_score"] > hierarchical_score_phase12)
        else "SCARTATO: la feature cascading non porta un miglioramento sufficiente rispetto alle alternative gia' disponibili."
    )
    print(f"\nVERDETTO: {verdict}")

    payload = {
        "target_label": TARGET_LABEL,
        "cascading_feature": CASCADING_FEATURE,
        "n_oof": len(oof_index),
        "baseline_binary_independent": baseline_metrics,
        "cascading_binary_independent_plus_prev_threshold": cascading_metrics,
        "delta_selection_score": delta,
        "hierarchical_selection_score_reference_phase1_2": hierarchical_score_phase12,
        "verdict": verdict,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSalvato in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

