"""Step A: curva soglia/ROI per cards_line_{3_5,4_5,5_5,6_5}, quote-only."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.train_multi_market import _build_temporal_cv, _filter_valid_splits
from src.ml.calibration.calibration_service import CalibrationService


def roi(y, scelte, quote):
    if scelte.sum() == 0:
        return float("nan")
    incasso = np.where(y[scelte] == 1, quote[scelte], 0.0).sum()
    return (incasso - scelte.sum()) / scelte.sum() * 100


for linea in ("3_5", "4_5", "5_5", "6_5"):
    market = f"cards_line_{linea}"
    QUOTE = [
        f"prob_norm_over_{linea}_line_{linea}",
        f"odds_mean_over_{linea}_line_{linea}",
        f"odds_mean_under_{linea}_line_{linea}",
        f"odds_count_line_{linea}",
        f"odds_std_over_{linea}_line_{linea}",
        f"overround_line_{linea}",
    ]

    df = FilterMarketService().build_dataset(market=market, fill_missing=False)
    df = df[df[f"odds_count_line_{linea}"].notna()].copy()
    df["prediction_at"] = pd.to_datetime(df["prediction_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["prediction_at"]).sort_values("prediction_at").reset_index(drop=True)

    X, y = df[QUOTE], df["y"].astype(int)
    splits = _filter_valid_splits(y, _build_temporal_cv(df) or [])
    if not splits:
        print(market, "NESSUNO SPLIT VALIDO - skip")
        continue

    modello = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    ris = CalibrationService.calibrate_estimator(estimator=modello, X=X, y=y, cv_splits=splits)
    p = np.asarray(ris.post_probabilities)
    y_oof = np.asarray(ris.post_y_true)
    indici = np.concatenate([np.asarray(va) for tr, va in splits if len(tr) and len(va)])
    q_over = df[f"odds_mean_over_{linea}_line_{linea}"].to_numpy()[indici]
    q_under = df[f"odds_mean_under_{linea}_line_{linea}"].to_numpy()[indici]

    print(f"\n=== {market} (righe={len(df)}) ===")
    print("ECE pre->post:", ris.pre_metrics.get("ece"), "->", ris.post_metrics.get("ece"))
    print("AUC pre:", ris.pre_metrics.get("auc"), "AUC post:", ris.post_metrics.get("auc"))
    for soglia in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        scelte = p >= soglia
        n = int(scelte.sum())
        if n < 30:
            print(f"  OVER  soglia={soglia} n={n} (poche)")
            continue
        prec = y_oof[scelte].mean()
        print(f"  OVER  soglia={soglia} n={n:4d} precisione={prec:.1%} quota_media={q_over[scelte].mean():.3f} ROI={roi(y_oof, scelte, q_over):.1f}%")
    for soglia in (0.45, 0.40, 0.35, 0.30, 0.25, 0.20):
        scelte = p <= soglia
        n = int(scelte.sum())
        if n < 30:
            print(f"  UNDER soglia={soglia} n={n} (poche)")
            continue
        prec = (1 - y_oof[scelte]).mean()
        print(f"  UNDER soglia={soglia} n={n:4d} precisione={prec:.1%} quota_media={q_under[scelte].mean():.3f} ROI={roi(1 - y_oof, scelte, q_under):.1f}%")
