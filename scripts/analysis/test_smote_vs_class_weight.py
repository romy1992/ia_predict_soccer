"""SMOTE vs class_weight='balanced' (l'approccio attualmente in uso in
`train_multi_market.py`/`_model_space`, MAI confrontato con SMOTE in questa
pipeline - SMOTE esiste solo nel codice legacy pre-v2, morto, non importato
da nulla di attivo: `src/service_ia/training/under_over/{inconsistent,consistent}/`).

Confronto controllato: STESSO RandomForest (iperparametri fissi, stile
`_RF_KWARGS` di totals_market.py - non e' una nuova grid search, l'obiettivo
e' isolare l'effetto di SMOTE vs class_weight, non ri-ottimizzare) sullo
stesso walk-forward (`_build_temporal_cv`/`_filter_valid_splits`, identico a
train_multi_market.py), con SMOTE applicato SOLO al training fold di ciascun
fold via `imblearn.pipeline.Pipeline` (mai al validation fold - la stessa
garanzia anti-leakage gia' rispettata altrove nel progetto).

Valuta sia il selection_score aggregato sia - dato che l'operatore vuole
puntare quando il modello dice "Over" - la precisione sulla classe Over a
parita' di recall (stessa metodologia di find_betting_thresholds.py).
"""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_recall_curve

from src.ml.evaluation.probability_metrics import champion_probability_score, compute_probability_metrics
from src.service_ia.training.train_multi_market import _build_temporal_cv, _filter_valid_splits

warnings.filterwarnings("ignore", category=UserWarning)

EXPORT_DIR = os.path.join("scripts", "analysis", "_export")
OUTPUT_PATH = os.path.join("best_models", "smote_vs_class_weight_result.json")
MARKETS = ["under_over_1_5", "under_over_2_5", "under_over_3_5", "under_over_4_5"]

_RF_FIXED_KWARGS = dict(n_estimators=150, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)
RECALL_FLOORS = [0.10, 0.20, 0.30]


def _class1_probability(raw_proba: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw_proba, dtype=float)
    return arr[:, 1] if arr.ndim == 2 and arr.shape[1] == 2 else arr.reshape(-1)


def _oof_probs(build_pipeline, X: pd.DataFrame, y: pd.Series, cv_splits) -> np.ndarray:
    n = len(X)
    oof = np.full(n, np.nan)
    for train_idx, valid_idx in cv_splits:
        if not train_idx or not valid_idx or y.iloc[train_idx].nunique() < 2:
            continue
        pipeline = build_pipeline()
        try:
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        except ValueError:
            continue  # fold troppo piccolo per SMOTE (k_neighbors) - saltato, non forzato
        proba = pipeline.predict_proba(X.iloc[valid_idx])
        oof[valid_idx] = _class1_probability(proba)
    return oof


def _selection_score(y_true: np.ndarray, p1: np.ndarray) -> dict:
    metrics = compute_probability_metrics(y_true=y_true, probabilities=p1, n_bins=10)
    predicted = (p1 >= 0.5).astype(int)
    f1_weighted = float(f1_score(y_true, predicted, average="weighted", zero_division=0))
    score = champion_probability_score(metrics=metrics, f1_weighted=f1_weighted)
    return {**metrics, "f1_weighted": f1_weighted, "selection_score": score}


def _precision_at_recall_floors(y_true: np.ndarray, p1: np.ndarray) -> dict:
    precision, recall, _ = precision_recall_curve(y_true, p1)
    out = {}
    for floor in RECALL_FLOORS:
        mask = recall[:-1] >= floor
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        best = idx[np.argmax(precision[idx])]
        out[f"{floor:.0%}"] = {"precision": round(float(precision[best]), 4), "recall": round(float(recall[best]), 4)}
    return out


def evaluate_market(market: str) -> dict:
    df = pd.read_csv(os.path.join(EXPORT_DIR, f"{market}.csv"))
    df["prediction_at"] = pd.to_datetime(df["prediction_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)

    y = df["y"].astype(int)
    X = df.drop(columns=["y", "market"], errors="ignore")
    X = X.drop(columns=["id_fixture", "season", "league", "prediction_at"], errors="ignore")

    raw_splits = _build_temporal_cv(df)
    cv_splits = _filter_valid_splits(y=y, splits=raw_splits)

    def build_balanced():
        return ImbPipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(class_weight="balanced", **_RF_FIXED_KWARGS)),
        ])

    def build_smote():
        return ImbPipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=42)),
            ("model", RandomForestClassifier(class_weight=None, **_RF_FIXED_KWARGS)),
        ])

    oof_balanced = _oof_probs(build_balanced, X, y, cv_splits)
    oof_smote = _oof_probs(build_smote, X, y, cv_splits)

    valid = ~np.isnan(oof_balanced) & ~np.isnan(oof_smote)
    idx = np.where(valid)[0]
    y_true = y.to_numpy()[idx]

    result = {"n_oof": int(len(idx)), "minority_class": "under" if y.mean() > 0.5 else "over"}
    for name, probs in (("class_weight_balanced", oof_balanced), ("smote", oof_smote)):
        p1 = probs[idx]
        result[name] = {
            "selection_score": _selection_score(y_true, p1),
            "precision_at_recall_over": _precision_at_recall_floors(y_true, p1),
        }
    return result


def main() -> None:
    t0 = time.time()
    results = {}
    for market in MARKETS:
        print(f"=== {market} ===", flush=True)
        t1 = time.time()
        result = evaluate_market(market)
        results[market] = result
        print(f"n_oof={result['n_oof']} | elapsed={time.time()-t1:.1f}s")
        for name in ("class_weight_balanced", "smote"):
            r = result[name]
            print(f"  {name:22s}: selection_score={r['selection_score']['selection_score']:.4f} "
                  f"log_loss={r['selection_score']['log_loss']:.4f} brier={r['selection_score']['brier']:.4f}")
            for floor, point in r["precision_at_recall_over"].items():
                print(f"      recall_over>={floor}: precision_over={point['precision']:.3f} (recall reale={point['recall']:.3f})")
        print()

    print(f"elapsed totale={time.time()-t0:.1f}s")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Salvato in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
