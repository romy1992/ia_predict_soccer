"""Quote specifiche per linea vs quote in un unico paniere (pooled) per
Corners/Cards - verifica EMPIRICA del fix descritto in
`src/service_ia/training/market_service/filter_market_service.py`
(`_extract_line_specific_odds_features`) e in
`src/ml/markets/{corners,cards}/*_market.py` (`_line_specific_odds_features`,
`_feature_columns_for`), 2026-09-12.

Prima del fix, `_extract_market_odds_features` metteva in un UNICO paniere le
quote di TUTTE le linee quotate (8.5/9.5/10.5/11.5 corners, 3.5/4.5/5.5/6.5
cards) per ogni bookmaker, perche' `switch_bet()` non separa Corners/Cards per
soglia come invece fa gia' per i gol (`under_over_1_5/_2_5/_3_5/_4_5`).
Risultato: `odds_count` fino a 218, `odds_max` fino a 66.0 con `odds_min` fino
a 1.0 - impossibile per una singola linea/evento.

Confronto controllato: stesso RandomForest class_weight='balanced' fisso
(`_RF_FIXED_KWARGS`, stesso stile di `test_split_odds_vs_pooled.py`), stesso
walk-forward, feature `mean_*` (+ `referee_*` per cards) IDENTICHE in
entrambe le varianti - cambiano SOLO le feature quota:
  - pooled: le 15 colonne `odds_*` storiche (mischiano tutte le linee)
  - line_specific: le 15 colonne `odds_*_line_X_Y` della sola linea attiva

Un'UNICA esportazione (`corners.csv`/`cards.csv`) contiene gia' entrambi i
set di colonne fianco a fianco (stesse righe, niente merge per `id_fixture`
necessario - a differenza del confronto pooled/split-by-outcome dei gol).
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

from src.ml.evaluation.classification_report import compute_full_classification_report
from src.ml.evaluation.probability_metrics import champion_probability_score, compute_probability_metrics
from src.ml.validation.temporal_split import expanding_window_splits

warnings.filterwarnings("ignore", category=UserWarning)

EXPORT_DIR = os.path.join("scripts", "analysis", "_export")
OUTPUT_PATH = os.path.join("best_models", "line_specific_odds_vs_pooled_result.json")

_ODDS_METRIC_KEYS = ["odds_count", "odds_mean", "odds_std", "odds_min", "odds_max"] + [f"odds_slot_{i}" for i in range(1, 11)]

_RF_FIXED_KWARGS = dict(n_estimators=150, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight="balanced")
ADOPTION_DELTA_THRESHOLD = 0.01

MARKET_CONFIG = {
    "corners": {
        "csv": "corners.csv",
        "lines": ["8_5", "9_5", "10_5", "11_5"],
        "extra_cols": [],
    },
    "cards": {
        "csv": "cards.csv",
        "lines": ["3_5", "4_5", "5_5", "6_5"],
        "extra_cols": [
            "referee_avg_cards_prior",
            "referee_severity_index_prior",
            "referee_matches_officiated_prior",
            "referee_has_history",
        ],
    },
}


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


def evaluate_line(frame: pd.DataFrame, mean_cols: list[str], extra_cols: list[str], line_label: str) -> dict:
    y_col = f"y_line_{line_label}"
    line_odds_cols = [f"{key}_line_{line_label}" for key in _ODDS_METRIC_KEYS]

    working = frame.dropna(subset=[y_col]).copy()
    working["prediction_at"] = pd.to_datetime(working["prediction_at"], utc=True, errors="coerce")
    working = working.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)

    y = working[y_col].astype(int)
    raw_splits = expanding_window_splits(
        frame=working, time_col="prediction_at", n_splits=5,
        min_train_size=max(30, int(len(working) * 0.45)),
        min_valid_size=max(10, int(len(working) * 0.1)),
    )
    cv_splits = [(tr, va) for tr, va in raw_splits if tr and va and y.iloc[tr].nunique() >= 2]

    X_pooled = working[mean_cols + extra_cols + _ODDS_METRIC_KEYS]
    X_line = working[mean_cols + extra_cols + line_odds_cols]

    oof_pooled = _oof_probs(X_pooled, y, cv_splits)
    oof_line = _oof_probs(X_line, y, cv_splits)

    valid = ~np.isnan(oof_pooled) & ~np.isnan(oof_line)
    idx = np.where(valid)[0]
    y_true = y.to_numpy()[idx]

    pooled_score = _selection_score(y_true, oof_pooled[idx])
    line_score = _selection_score(y_true, oof_line[idx])
    delta = line_score["selection_score"] - pooled_score["selection_score"]

    pooled_classification = compute_full_classification_report(y_true, oof_pooled[idx])
    line_classification = compute_full_classification_report(y_true, oof_line[idx])

    return {
        "n_rows": int(len(working)),
        "n_oof": int(len(idx)),
        "pooled": pooled_score,
        "line_specific": line_score,
        "pooled_classification": pooled_classification,
        "line_specific_classification": line_classification,
        "delta_selection_score": delta,
        "verdict": (
            f"ADOTTATO: delta {delta:+.4f} supera {ADOPTION_DELTA_THRESHOLD}"
            if delta > ADOPTION_DELTA_THRESHOLD
            else f"SCARTATO: delta {delta:+.4f} non supera {ADOPTION_DELTA_THRESHOLD}"
        ),
    }


def main() -> None:
    t0 = time.time()
    results: dict[str, dict] = {}
    for market, config in MARKET_CONFIG.items():
        frame = pd.read_csv(os.path.join(EXPORT_DIR, config["csv"]))
        mean_cols = [c for c in frame.columns if c.startswith("mean_")]
        results[market] = {}
        for line_label in config["lines"]:
            print(f"=== {market} line {line_label} ===", flush=True)
            t1 = time.time()
            result = evaluate_line(frame, mean_cols, config["extra_cols"], line_label)
            results[market][line_label] = result
            print(f"n_rows={result['n_rows']} n_oof={result['n_oof']} elapsed={time.time()-t1:.1f}s")
            pooled_acc = result["pooled_classification"].get("accuracy")
            line_acc = result["line_specific_classification"].get("accuracy")
            print(f"  pooled        : selection_score={result['pooled']['selection_score']:.4f} auc={result['pooled']['auc']:.4f} accuracy={pooled_acc:.4f}")
            print(f"  line_specific : selection_score={result['line_specific']['selection_score']:.4f} auc={result['line_specific']['auc']:.4f} accuracy={line_acc:.4f}")
            print(f"  -> {result['verdict']}\n", flush=True)

    print(f"elapsed totale={time.time()-t0:.1f}s")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Salvato in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
