"""Quote separate per lato (Over vs Under) vs quote in un unico paniere
(comportamento attuale di produzione, `FilterMarketService._extract_market_odds_features`)
- idea proposta dall'operatore, 2026-09-09.

Oggi il paniere unico mette insieme le quote Over e Under di tutti i
bookmaker (`odds_count/mean/std/min/max` + 10 quote ordinate, 15 feature) -
`odds_count` conta quindi il doppio dei bookmaker reali e mean/std
mescolano i due lati. La variante qui testata (vedi
`export_split_odds_for_cloud_training.py`) calcola invece 5 statistiche
PER LATO (count/mean/std/min/max, 10 feature totali - senza le 10 quote
ordinate della versione pooled, quindi il confronto ha anche 5 feature in
meno: se la versione split vince comunque e' un segnale piu' forte, se
perde va letto con questa cautela).

Confronto controllato: stesso RandomForest class_weight='balanced'
(`_RF_FIXED_KWARGS`, stile gia' usato in test_smote_vs_class_weight.py /
test_gradient_boosting_vs_baseline.py), stesso walk-forward, feature
mean_statistics IDENTICHE in entrambe le varianti (cambiano solo le feature
quota) - le due esportazioni (pooled/split) sono state fatte in momenti
diversi quindi possono avere righe leggermente diverse (nuove fixture nel
frattempo): il confronto e' ristretto all'intersezione per `id_fixture`,
stesso principio anti-distorsione gia' usato in fase 4/5.
"""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.ml.evaluation.probability_metrics import champion_probability_score, compute_probability_metrics
from src.ml.validation.temporal_split import expanding_window_splits

warnings.filterwarnings("ignore", category=UserWarning)

EXPORT_DIR = os.path.join("scripts", "analysis", "_export")
OUTPUT_PATH = os.path.join("best_models", "split_odds_vs_pooled_result.json")
MARKETS = ["under_over_1_5", "under_over_2_5", "under_over_3_5", "under_over_4_5"]

_RF_FIXED_KWARGS = dict(n_estimators=150, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight="balanced")
ADOPTION_DELTA_THRESHOLD = 0.01


def _class1_probability(raw_proba: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw_proba, dtype=float)
    return arr[:, 1] if arr.ndim == 2 and arr.shape[1] == 2 else arr.reshape(-1)


def _oof_probs(X: pd.DataFrame, y: pd.Series, cv_splits) -> np.ndarray:
    n = len(X)
    oof = np.full(n, np.nan)
    for train_idx, valid_idx in cv_splits:
        if not train_idx or not valid_idx or y.iloc[train_idx].nunique() < 2:
            continue
        pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(**_RF_FIXED_KWARGS)),
        ])
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = pipeline.predict_proba(X.iloc[valid_idx])
        oof[valid_idx] = _class1_probability(proba)
    return oof


def _selection_score(y_true: np.ndarray, p1: np.ndarray) -> dict:
    metrics = compute_probability_metrics(y_true=y_true, probabilities=p1, n_bins=10)
    predicted = (p1 >= 0.5).astype(int)
    f1_weighted = float(f1_score(y_true, predicted, average="weighted", zero_division=0))
    score = champion_probability_score(metrics=metrics, f1_weighted=f1_weighted)
    return {**metrics, "f1_weighted": f1_weighted, "selection_score": score}


def evaluate_market(market: str) -> dict:
    pooled = pd.read_csv(os.path.join(EXPORT_DIR, f"{market}.csv"))
    split = pd.read_csv(os.path.join(EXPORT_DIR, f"{market}_split_odds.csv"))

    pooled_odds_cols = [c for c in pooled.columns if c.startswith("odds_")]
    split_odds_cols = [c for c in split.columns if c.startswith("over_odds") or c.startswith("under_odds")]
    mean_cols = [c for c in pooled.columns if c.startswith("mean_")]

    common_fixtures = sorted(set(pooled["id_fixture"]) & set(split["id_fixture"]))
    pooled = pooled[pooled["id_fixture"].isin(common_fixtures)]
    split = split[["id_fixture"] + split_odds_cols]

    frame = pooled.merge(split, on="id_fixture", how="inner", validate="one_to_one")
    frame["prediction_at"] = pd.to_datetime(frame["prediction_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)

    y = frame["y"].astype(int)
    raw_splits = expanding_window_splits(
        frame=frame, time_col="prediction_at", n_splits=5,
        min_train_size=max(30, int(len(frame) * 0.45)),
        min_valid_size=max(10, int(len(frame) * 0.1)),
    )
    cv_splits = [(tr, va) for tr, va in raw_splits if tr and va and y.iloc[tr].nunique() >= 2]

    X_pooled = frame[mean_cols + pooled_odds_cols]
    X_split = frame[mean_cols + split_odds_cols]

    oof_pooled = _oof_probs(X_pooled, y, cv_splits)
    oof_split = _oof_probs(X_split, y, cv_splits)

    valid = ~np.isnan(oof_pooled) & ~np.isnan(oof_split)
    idx = np.where(valid)[0]
    y_true = y.to_numpy()[idx]

    pooled_score = _selection_score(y_true, oof_pooled[idx])
    split_score = _selection_score(y_true, oof_split[idx])
    delta = split_score["selection_score"] - pooled_score["selection_score"]

    return {
        "n_common_fixtures": len(common_fixtures),
        "n_oof": int(len(idx)),
        "pooled_odds_n_features": len(pooled_odds_cols),
        "split_odds_n_features": len(split_odds_cols),
        "pooled": pooled_score,
        "split_by_outcome": split_score,
        "delta_selection_score": delta,
        "verdict": (
            f"ADOTTATO: delta {delta:+.4f} supera {ADOPTION_DELTA_THRESHOLD}"
            if delta > ADOPTION_DELTA_THRESHOLD
            else f"SCARTATO: delta {delta:+.4f} non supera {ADOPTION_DELTA_THRESHOLD}"
        ),
    }


def main() -> None:
    t0 = time.time()
    results = {}
    for market in MARKETS:
        print(f"=== {market} ===", flush=True)
        t1 = time.time()
        result = evaluate_market(market)
        results[market] = result
        print(f"n_common_fixtures={result['n_common_fixtures']} n_oof={result['n_oof']} elapsed={time.time()-t1:.1f}s")
        print(f"  pooled ({result['pooled_odds_n_features']} feature quota) : selection_score={result['pooled']['selection_score']:.4f} auc={result['pooled']['auc']:.4f}")
        print(f"  split  ({result['split_odds_n_features']} feature quota) : selection_score={result['split_by_outcome']['selection_score']:.4f} auc={result['split_by_outcome']['auc']:.4f}")
        print(f"  -> {result['verdict']}\n", flush=True)

    print(f"elapsed totale={time.time()-t0:.1f}s")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Salvato in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
