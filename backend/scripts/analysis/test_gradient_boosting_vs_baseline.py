"""XGBoost/LightGBM vs gli approcci gia' in uso (`class_weight='balanced'`
RandomForest e RandomForest+SMOTE, gia' confrontati in
`test_smote_vs_class_weight.py`) - MAI provati nella pipeline v2
(`train_multi_market.py`/`_model_space`): entrambi sono gia' dipendenze del
progetto (`requirements.txt`) e usati nel vecchio codice pre-v2 legacy
(`src/service_ia/training/under_over/...`), ma mai portati nella pipeline
attuale che produce i champion realmente registrati.

Confronto controllato, stesso principio del test SMOTE: STESSO walk-forward
(`_build_temporal_cv`/`_filter_valid_splits`, identico a train_multi_market.py),
iperparametri fissi per ciascun modello (non una nuova grid search - qui
serve solo isolare l'effetto della famiglia di modello), stesso
`SimpleImputer` per tutte le pipeline (isolare l'effetto modello, non la
gestione dei NaN - anche se XGBoost/LightGBM gestirebbero i NaN nativamente).

Eseguito sui 5 mercati gia' testati oggi (4 Under/Over + goal_no_goal).
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
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.ml.evaluation.probability_metrics import champion_probability_score, compute_probability_metrics
from src.service_ia.training.train_multi_market import _build_temporal_cv, _filter_valid_splits

warnings.filterwarnings("ignore", category=UserWarning)

EXPORT_DIR = os.path.join("scripts", "analysis", "_export")
OUTPUT_PATH = os.path.join("best_models", "gradient_boosting_vs_baseline_result.json")
MARKETS = ["under_over_1_5", "under_over_2_5", "under_over_3_5", "under_over_4_5", "goal_no_goal"]

_RF_FIXED_KWARGS = dict(n_estimators=150, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)
ADOPTION_DELTA_THRESHOLD = 0.01


def _class1_probability(raw_proba: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw_proba, dtype=float)
    return arr[:, 1] if arr.ndim == 2 and arr.shape[1] == 2 else arr.reshape(-1)


def _oof_probs(build_pipeline, X: pd.DataFrame, y: pd.Series, cv_splits) -> np.ndarray:
    n = len(X)
    oof = np.full(n, np.nan)
    for train_idx, valid_idx in cv_splits:
        if not train_idx or not valid_idx or y.iloc[train_idx].nunique() < 2:
            continue
        pipeline = build_pipeline(y.iloc[train_idx])
        try:
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        except ValueError:
            continue  # fold troppo piccolo (es. SMOTE k_neighbors) - saltato, non forzato
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
    df = pd.read_csv(os.path.join(EXPORT_DIR, f"{market}.csv"))
    df["prediction_at"] = pd.to_datetime(df["prediction_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)

    y = df["y"].astype(int)
    X = df.drop(columns=["y", "market"], errors="ignore")
    X = X.drop(columns=["id_fixture", "season", "league", "prediction_at"], errors="ignore")

    raw_splits = _build_temporal_cv(df)
    cv_splits = _filter_valid_splits(y=y, splits=raw_splits)

    def build_balanced(_y_train):
        return ImbPipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(class_weight="balanced", **_RF_FIXED_KWARGS)),
        ])

    def build_smote(_y_train):
        return ImbPipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=42)),
            ("model", RandomForestClassifier(class_weight=None, **_RF_FIXED_KWARGS)),
        ])

    def build_xgboost(y_train):
        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0
        return ImbPipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", random_state=42, n_jobs=-1,
            )),
        ])

    def build_lightgbm(_y_train):
        return ImbPipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
            )),
        ])

    builders = {
        "class_weight_balanced": build_balanced,
        "smote": build_smote,
        "xgboost": build_xgboost,
        "lightgbm": build_lightgbm,
    }

    oofs = {name: _oof_probs(builder, X, y, cv_splits) for name, builder in builders.items()}

    valid = np.ones(len(X), dtype=bool)
    for arr in oofs.values():
        valid &= ~np.isnan(arr)
    idx = np.where(valid)[0]
    y_true = y.to_numpy()[idx]

    result = {"n_oof": int(len(idx)), "base_rate_over": float(y.mean())}
    for name, probs in oofs.items():
        result[name] = {"selection_score": _selection_score(y_true, probs[idx])}

    baseline_score = result["class_weight_balanced"]["selection_score"]["selection_score"]
    best_name = max(builders.keys(), key=lambda n: result[n]["selection_score"]["selection_score"])
    best_score = result[best_name]["selection_score"]["selection_score"]
    result["best_model"] = best_name
    result["best_selection_score"] = best_score
    result["delta_vs_class_weight_balanced"] = best_score - baseline_score
    result["verdict"] = (
        f"ADOTTATO ({best_name}): delta {result['delta_vs_class_weight_balanced']:+.4f} supera {ADOPTION_DELTA_THRESHOLD}"
        if result["delta_vs_class_weight_balanced"] > ADOPTION_DELTA_THRESHOLD
        else f"SCARTATO: nessun modello supera class_weight_balanced di oltre {ADOPTION_DELTA_THRESHOLD} (migliore: {best_name}, delta {result['delta_vs_class_weight_balanced']:+.4f})"
    )
    return result


def main() -> None:
    t0 = time.time()
    results = {}
    for market in MARKETS:
        print(f"=== {market} ===", flush=True)
        t1 = time.time()
        result = evaluate_market(market)
        results[market] = result
        print(f"n_oof={result['n_oof']} | base_rate_over={result['base_rate_over']:.3f} | elapsed={time.time()-t1:.1f}s")
        for name in ("class_weight_balanced", "smote", "xgboost", "lightgbm"):
            r = result[name]["selection_score"]
            print(f"  {name:22s}: selection_score={r['selection_score']:.4f} log_loss={r['log_loss']:.4f} brier={r['brier']:.4f} auc={r['auc']:.4f}")
        print(f"  -> {result['verdict']}\n", flush=True)

    print(f"elapsed totale={time.time()-t0:.1f}s")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Salvato in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
