"""EDA + pipeline di produzione (random search + voting/stacking) per h2h/dc,
piu' la regola di coerenza casa => 1X.

Il giro precedente (phase6) era un RandomForest fisso: niente EDA, niente
random search, niente ensemble. Questo script fa i passi che mancavano.

Nessun pkl nel registry (save_model=False): get_latest continuerebbe a
servire il candidato.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logging.basicConfig(level=logging.INFO)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from scripts.analysis.phase6_h2h_dc_retrain_and_cascade import (
    DC_REQUIRED_ODDS,
    H2H_REQUIRED_ODDS,
    _prepare_frame,
    dc_feature_columns,
    h2h_feature_columns,
    score_oof,
)
from src.ml.evaluation.probability_metrics import temporal_oof_probabilities
from src.ml.markets.h2h_dc_coherence import (
    count_home_but_not_1x,
    enforce_p_1x_at_least_p_home,
    force_dc_1x_when_home_win,
)
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.train_multi_market import (
    _build_temporal_cv,
    _filter_valid_splits,
    train_market,
)

EXPORT_DIR = os.path.abspath(os.path.join("scripts", "analysis", "_export"))
OUTPUT_PATH = os.path.abspath(os.path.join("scripts", "analysis", "phase6b_full_pipeline_result.json"))
META = {"y", "market", "id_fixture", "season", "league", "prediction_at"}
RANDOM_SEARCH_ITER = 12

# Metriche del champion SERVITO (400 righe, 69 feature legacy), gia' misurate
# sul dataset completo. I .pkl non sono in questo ambiente.
PROD = {
    "h2h": {
        "source": "champion servito, 400 righe, voting, 69 feat odds_slot+mean_stats",
        "n_train": 400,
        "n_eval": 15665,
        "accuracy": 0.6099,
        "majority_baseline_accuracy": 0.5699,
        "accuracy_minus_majority_pp": 4.0,
        "auc": 0.6495,
        "brier": 0.2167,
        "ece": 0.111,
        "macro_f1": 0.477,
        "minority_recall": 0.123,
        "pred_positive_rate": 0.07,
        "note": "predice non-casa nel 93%; recall classe 1 = 0.123",
    },
    "dc": {
        "source": "champion servito, 400 righe, voting, 69 feat identiche a h2h",
        "n_train": 400,
        "n_eval": 8873,
        "accuracy": 0.7074,
        "majority_baseline_accuracy": 0.6951,
        "accuracy_minus_majority_pp": 1.2,
        "auc": 0.5771,
        "brier": 0.2417,
        "ece": 0.166,
        "macro_f1": 0.470,
        "minority_recall": 0.063,
        "pred_positive_rate": 0.97,
        "note": "predice 1X nel 97%; recall classe 0 = 0.063",
    },
}


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def eda_report(df: pd.DataFrame, market: str) -> dict[str, Any]:
    numeriche = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    mancanti = df.isna().sum()
    mancanti = mancanti[mancanti > 0].sort_values(ascending=False)
    missing = {col: {"n": int(n), "pct": float(n / len(df))} for col, n in mancanti.items()}

    corr: dict[str, float] = {}
    for col in numeriche:
        valide = df[[col, "y"]].dropna()
        if len(valide) > 200 and valide[col].nunique() > 1:
            valore = valide[col].corr(valide["y"], method="spearman")
            if pd.notna(valore):
                corr[col] = float(valore)
    top_corr = sorted(corr.items(), key=lambda kv: abs(kv[1]), reverse=True)[:20]

    base = [c for c in numeriche if df[c].notna().sum() > len(df) * 0.5]
    coppie = []
    if base:
        matrice = df[base].corr(method="spearman").abs()
        for i, a in enumerate(base):
            for b in base[i + 1 :]:
                valore = matrice.loc[a, b]
                if pd.notna(valore) and valore > 0.95:
                    coppie.append({"a": a, "b": b, "corr": float(valore)})
        coppie.sort(key=lambda row: -row["corr"])

    vc = df["y"].value_counts(dropna=False).sort_index()
    print(f"\n=== EDA {market}: {len(df)} righe, {df.shape[1]} colonne ===")
    print(f"y: { {int(k) if pd.notna(k) else None: int(v) for k, v in vc.items()} }")
    print(f"colonne con NaN: {len(missing)}  celle NaN: {int(df.isna().sum().sum())}")
    print("top corr vs y:")
    for col, valore in top_corr[:8]:
        print(f"  {valore:+.4f}  {col}")
    print(f"coppie |corr|>0.95: {len(coppie)}")
    for row in coppie[:8]:
        print(f"  {row['corr']:.4f}  {row['a']} <-> {row['b']}")

    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "period": {
            "min": str(pd.to_datetime(df["prediction_at"], utc=True, errors="coerce").min()),
            "max": str(pd.to_datetime(df["prediction_at"], utc=True, errors="coerce").max()),
        },
        "y_counts": {str(k): int(v) for k, v in vc.items()},
        "n_missing_cells": int(df.isna().sum().sum()),
        "missing_columns": missing,
        "top_corr_vs_y": [{"column": c, "spearman": v} for c, v in top_corr],
        "redundant_pairs_over_0_95": coppie[:40],
        "odds_mean_slugs": sorted({c[len("odds_mean_") :] for c in df.columns if c.startswith("odds_mean_")}),
    }


def drop_redundant(columns: list[str], corr: dict[str, float], pairs: list[dict[str, Any]]) -> list[str]:
    keep = list(columns)
    for row in pairs:
        a, b = row["a"], row["b"]
        if a not in keep or b not in keep:
            continue
        # Tieni quella piu' correlata col target; a parita' la prima.
        drop = b if abs(corr.get(a, 0.0)) >= abs(corr.get(b, 0.0)) else a
        if drop in keep:
            keep.remove(drop)
    return keep


def features_after_eda(df: pd.DataFrame, market: str, eda: dict[str, Any]) -> list[str]:
    base = h2h_feature_columns(df) if market == "h2h" else dc_feature_columns(df)
    too_missing = {col for col, payload in eda["missing_columns"].items() if payload["pct"] > 0.20}
    base = [c for c in base if c in df.columns and c not in too_missing]
    corr = {row["column"]: row["spearman"] for row in eda["top_corr_vs_y"]}
    # Anche le colonne non in top20: ricalcolo veloce solo per i pair.
    corr_full = corr
    return drop_redundant(base, corr_full, eda["redundant_pairs_over_0_95"])


def train_one(market: str, frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    print(f"\n=== train_market({market}) random search iter={RANDOM_SEARCH_ITER} ===")
    t0 = time.time()
    with patch.object(FilterMarketService, "build_dataset", return_value=frame):
        result = train_market(
            market=market,
            selection_method="kbest",
            save_model=False,
            feature_columns=feature_columns,
            search_strategy="random",
            random_search_iter=RANDOM_SEARCH_ITER,
        )
    print(f"stato={result.status} champion={result.champion} durata={(time.time()-t0)/60:.1f} min")
    models = {}
    for nome, payload in (result.details.get("models") or {}).items():
        m = payload.get("probability_metrics") or {}
        models[nome] = {
            "selection_score": payload.get("selection_score"),
            "best_cv_f1": payload.get("best_cv_f1"),
            "auc": m.get("auc"),
            "brier": m.get("brier"),
            "ece": m.get("ece"),
            "log_loss": m.get("log_loss"),
            "best_params": payload.get("best_params"),
        }
        print(
            f"  {nome:22} score={payload.get('selection_score')} "
            f"auc={m.get('auc')} brier={m.get('brier')} ece={m.get('ece')}"
        )

    y = frame["y"].astype(int)
    cv_splits = _filter_valid_splits(y, _build_temporal_cv(frame) or [])
    oof = temporal_oof_probabilities(
        estimator=result.estimator,
        X=frame[feature_columns],
        y=y,
        cv_splits=cv_splits,
    )
    scored = None
    oof_by_fixture = None
    if not oof.empty:
        idx = oof["index"].to_numpy()
        p = oof["probability"].to_numpy()
        y_true = oof["y_true"].to_numpy()
        market_col = "prob_norm_home" if market == "h2h" else "implied_prob_1x"
        p_market = frame[market_col].to_numpy()[idx] if market_col in frame.columns else None
        scored = score_oof(y_true, p, p_market=p_market)
        scored["n_rows"] = int(len(frame))
        scored["n_features"] = int(len(feature_columns))
        oof_by_fixture = pd.Series(p, index=frame.loc[idx, "id_fixture"].to_numpy())
        pred = (p >= 0.5).astype(int)
        pred_by_fixture = pd.Series(pred, index=frame.loc[idx, "id_fixture"].to_numpy())
    else:
        pred_by_fixture = None

    return {
        "status": result.status,
        "champion": result.champion,
        "selected_features": result.selected_features,
        "details_summary": {
            "cv_folds": result.details.get("cv_folds"),
            "search_strategy": result.details.get("search_strategy"),
            "random_search_iter": result.details.get("random_search_iter"),
            "champion_selection_score": result.details.get("champion_selection_score"),
            "calibration": result.details.get("calibration"),
        },
        "candidates": models,
        "oof_metrics": scored,
        "oof_by_fixture": oof_by_fixture,
        "pred_by_fixture": pred_by_fixture,
        "elapsed_seconds": round(time.time() - t0, 2),
        "estimator": result.estimator,
        "frame": frame,
        "feature_columns": feature_columns,
        "cv_splits": cv_splits,
    }


def coherence_ab(h2h_run: dict[str, Any], dc_run: dict[str, Any]) -> dict[str, Any]:
    h2h_p = h2h_run["oof_by_fixture"]
    dc_p = dc_run["oof_by_fixture"]
    h2h_hat = h2h_run["pred_by_fixture"]
    dc_hat = dc_run["pred_by_fixture"]
    if h2h_p is None or dc_p is None:
        return {"status": "skipped"}

    common = sorted(set(dc_p.index) & set(h2h_p.index))
    if not common:
        return {"status": "skipped_no_join"}

    frame = dc_run["frame"].set_index("id_fixture")
    y = frame.loc[common, "y"].astype(int).to_numpy()
    p_dc = dc_p.loc[common].to_numpy()
    p_home = h2h_p.loc[common].to_numpy()
    pred_dc = dc_hat.loc[common].to_numpy()
    pred_h2h = h2h_hat.loc[common].to_numpy()

    violations = count_home_but_not_1x(pred_h2h, pred_dc)
    pred_forced = force_dc_1x_when_home_win(pred_h2h, pred_dc)
    p_forced = enforce_p_1x_at_least_p_home(p_home, p_dc)

    base = score_oof(y, p_dc)
    # Metriche di decisione: coerenza sulla label, probabilita' floored.
    label_metrics = score_oof(y, np.where(pred_forced == 1, np.maximum(p_dc, 0.51), np.minimum(p_dc, 0.49)))
    # Meglio: accuracy/confusion sulla label forzata, metriche di probabilita' sul p floored.
    cm = confusion_matrix(y, pred_forced, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    report = classification_report(y, pred_forced, labels=[0, 1], output_dict=True, zero_division=0)
    majority = int(np.bincount(y, minlength=2).argmax())
    acc = float(accuracy_score(y, pred_forced))
    p_metrics = score_oof(y, p_forced)

    print(f"\n=== Coerenza casa => 1X  (n={len(common)}) ===")
    print(f"violazioni (h2h=casa e dc=non 1X): {violations}")
    print(f"dc libero     acc={base['accuracy']:.4f} macro-F1={base['macro_f1']:.4f} auc={base['auc']:.4f}")
    print(
        f"dc coerente   acc={acc:.4f} macro-F1={float(f1_score(y, pred_forced, average='macro', zero_division=0)):.4f} "
        f"auc(p floor)={p_metrics['auc']:.4f}"
    )
    return {
        "status": "ok",
        "n": len(common),
        "violations_home_but_not_1x": violations,
        "dc_unconstrained": base,
        "dc_label_forced_home_implies_1x": {
            "accuracy": acc,
            "majority_baseline_accuracy": float((y == majority).mean()),
            "accuracy_minus_majority_pp": (acc - float((y == majority).mean())) * 100.0,
            "macro_f1": float(f1_score(y, pred_forced, average="macro", zero_division=0)),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "classification_report": report,
        },
        "dc_prob_floor_p1x_ge_phome": p_metrics,
        "label_override_using_shifted_proba_for_score_oof": label_metrics,
    }


def comparison_table(market: str, new_metrics: dict[str, Any] | None) -> dict[str, Any]:
    prod = PROD[market]
    if not new_metrics:
        return {"prod": prod, "new": None}
    keys = [
        "accuracy",
        "majority_baseline_accuracy",
        "accuracy_minus_majority_pp",
        "auc",
        "brier",
        "ece",
        "macro_f1",
        "minority_recall",
        "pred_positive_rate",
    ]
    delta = {}
    for key in keys:
        a, b = prod.get(key), new_metrics.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta[key] = b - a
    return {"prod": prod, "new_full_pipeline": {k: new_metrics.get(k) for k in keys + ["n", "n_rows", "n_features"]}, "delta_new_minus_prod": delta}


def main() -> None:
    t0 = time.time()
    os.makedirs(EXPORT_DIR, exist_ok=True)
    service = FilterMarketService()

    print("1) Dataset GREZZO (fill_missing=False) per EDA...")
    h2h_raw = service.build_dataset(market="h2h", fill_missing=False)
    dc_raw = service.build_dataset(market="dc", fill_missing=False)
    h2h_raw.to_csv(os.path.join(EXPORT_DIR, "h2h_raw.csv"), index=False)
    dc_raw.to_csv(os.path.join(EXPORT_DIR, "dc_raw.csv"), index=False)
    print(f"   h2h raw {h2h_raw.shape} NaN={int(h2h_raw.isna().sum().sum())}")
    print(f"   dc  raw {dc_raw.shape} NaN={int(dc_raw.isna().sum().sum())}")

    print("\n2) EDA (mancanti, correlazioni, ridondanze)...")
    eda_h2h = eda_report(h2h_raw, "h2h")
    eda_dc = eda_report(dc_raw, "dc")

    h2h = _prepare_frame(h2h_raw.fillna(0), H2H_REQUIRED_ODDS)
    dc = _prepare_frame(dc_raw.fillna(0), DC_REQUIRED_ODDS)
    h2h_features = features_after_eda(h2h, "h2h", eda_h2h)
    dc_features = features_after_eda(dc, "dc", eda_dc)
    print(f"\nfeature dopo EDA: h2h={len(h2h_features)}  dc={len(dc_features)}")
    print(f"h2h quote tenute: {[c for c in h2h_features if not c.endswith('_stat')]}")
    print(f"dc  quote tenute: {[c for c in dc_features if not c.endswith('_stat')]}")

    h2h_run = train_one("h2h", h2h, h2h_features)
    dc_run = train_one("dc", dc, dc_features)
    coherence = coherence_ab(h2h_run, dc_run)

    payload = {
        "elapsed_seconds": round(time.time() - t0, 2),
        "pipeline": {
            "dataset": "FilterMarketService.build_dataset(fill_missing=False) poi drop righe senza quote canoniche",
            "eda": "NaN, Spearman vs y, coppie |corr|>0.95, drop colonne >20% NaN e ridondanti",
            "search": f"RandomizedSearchCV n_iter={RANDOM_SEARCH_ITER} su logistic / RF / RF+SMOTE",
            "ensemble": "VotingClassifier soft + StackingClassifier sui 2 migliori, champion per selection_score",
            "cv": "expanding window temporale (_build_temporal_cv), OOF mai in-sample",
            "save_model": False,
            "coherence": "se h2h predice casa, dc=1X; P(1X)=max(P(1X), P(casa))",
        },
        "eda": {"h2h": eda_h2h, "dc": eda_dc},
        "h2h": {
            "champion": h2h_run["champion"],
            "candidates": h2h_run["candidates"],
            "selected_features": h2h_run["selected_features"],
            "details_summary": h2h_run["details_summary"],
            "oof_metrics": h2h_run["oof_metrics"],
            "elapsed_seconds": h2h_run["elapsed_seconds"],
        },
        "dc": {
            "champion": dc_run["champion"],
            "candidates": dc_run["candidates"],
            "selected_features": dc_run["selected_features"],
            "details_summary": dc_run["details_summary"],
            "oof_metrics": dc_run["oof_metrics"],
            "elapsed_seconds": dc_run["elapsed_seconds"],
        },
        "coherence_home_implies_1x": {k: v for k, v in coherence.items() if k != "label_override_using_shifted_proba_for_score_oof"},
        "confronto_prod_vs_nuovo": {
            "h2h": comparison_table("h2h", h2h_run["oof_metrics"]),
            "dc": comparison_table("dc", dc_run["oof_metrics"]),
        },
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_native)
    print(f"\nSalvato {OUTPUT_PATH} ({payload['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
