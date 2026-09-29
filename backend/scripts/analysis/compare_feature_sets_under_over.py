"""Confronto A/B/C/D/E dei set di feature su Under/Over 2.5 (2026-09-13).

Nasce dalla domanda dell'operatore "troppe feature... quali useresti?" dopo
la scoperta che le quote venivano mediate mescolando gli esiti. Invece di
scegliere il set a occhio, lo si misura: STESSO modello, STESSO split
temporale, STESSO seed su tutte le configurazioni, cosi' l'unica differenza
e' il set di feature (e i pesi di classe per E).

Serve a ORDINARE le configurazioni, non a giudicarle in assoluto: niente
grid search ne' ensemble, quindi i valori sono piu' bassi di quelli di
produzione. La pipeline completa si lancia poi solo sul vincitore.

`prob_norm` e `overround` NON vengono letti dal CSV: quelle colonne sommano
anche le quote "alternate" (presenti sul 30.8% delle righe), che gonfiano
l'overround a ~2.09 invece di ~1.05 e dimezzano la probabilita' normalizzata.
Qui sono ricalcolati dai soli esiti canonici over/under.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.service_ia.training.train_multi_market import _build_temporal_cv, _filter_valid_splits  # noqa: E402

CSV = os.path.join("scripts", "analysis", "_export", "under_over_2_5.csv")
REGISTRY = os.path.join("scripts", "analysis", "_export", "registry_index.jsonl")
LINE = "2_5"
SEED = 42

STAT_GOL = ["expected_goals", "shots_on_goal", "total_shots", "shots_insidebox", "goalkeeper_saves"]


def carica() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    over, under = f"odds_mean_over_{LINE}", f"odds_mean_under_{LINE}"
    # Solo righe con entrambi i lati canonici quotati: senza uno dei due non
    # esiste ne' la media richiesta ne' l'overround.
    df = df[(df[over] > 0) & (df[under] > 0)].copy()

    p_over_raw = 1.0 / df[over]
    p_under_raw = 1.0 / df[under]
    df["overround_canon"] = p_over_raw + p_under_raw
    df["prob_norm_over_canon"] = p_over_raw / df["overround_canon"]
    return df


def feature_correnti() -> list[str]:
    """Le 69 feature del modello under_over_2_5 realmente registrato."""
    ultimo = None
    with open(REGISTRY, encoding="utf-8") as f:
        for riga in f:
            r = json.loads(riga)
            if r.get("market") == "under_over_2_5":
                ultimo = r
    return list(ultimo.get("feature_names") or [])


def configurazioni(df: pd.DataFrame) -> dict[str, dict]:
    quote = [
        f"odds_mean_over_{LINE}",
        f"odds_mean_under_{LINE}",
        "prob_norm_over_canon",
        f"odds_count_over_{LINE}",
        f"odds_std_over_{LINE}",
        "overround_canon",
    ]
    stat = [f"mean_{g}_{lato}_stat" for g in STAT_GOL for lato in ("home", "away", "diff")]
    attuali = [c for c in feature_correnti() if c in df.columns]

    # Isola l'effetto del SOLO cambio quote: stesse identiche statistiche di
    # C, blocco quote sostituito da quello per esito. Senza questo confronto
    # C vs B mischia due variabili (quote diverse E statistiche diverse) e non
    # dice nulla sulla domanda di partenza.
    stat_attuali = [c for c in attuali if not c.startswith("odds_")]

    return {
        "A - solo quote": {"feature": quote, "pesi": None},
        "B - quote + stat gol": {"feature": quote + stat, "pesi": None},
        "C - attuale (quote vecchie)": {"feature": attuali, "pesi": None},
        "D - solo media Over": {"feature": [f"odds_mean_over_{LINE}"], "pesi": None},
        "E - come B, piu' peso Over": {"feature": quote + stat, "pesi": {0: 1.0, 1: 2.0}},
        "F - stat di C + quote nuove": {"feature": stat_attuali + quote, "pesi": None},
    }


def valuta(df: pd.DataFrame, feature: list[str], pesi) -> dict:
    frame = df.sort_values("prediction_at").reset_index(drop=True)
    # La STESSA CV del training di produzione (`_build_temporal_cv`: finestre
    # scalate sulla dimensione del dataset, 45% train / 10% validation), non i
    # default di `expanding_window_splits` - quelli sono tarati su dataset
    # piccoli e su 15k righe validerebbero su 150 righe in tutto.
    splits = _filter_valid_splits(frame["y"], _build_temporal_cv(frame) or [])
    y_all, p_all = [], []
    for train_idx, valid_idx in splits:
        X_tr, y_tr = frame.loc[train_idx, feature], frame.loc[train_idx, "y"]
        X_va, y_va = frame.loc[valid_idx, feature], frame.loc[valid_idx, "y"]
        modello = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=20,
                        random_state=SEED,
                        n_jobs=-1,
                        class_weight=pesi,
                    ),
                ),
            ]
        )
        modello.fit(X_tr, y_tr)
        p_all.extend(modello.predict_proba(X_va)[:, 1])
        y_all.extend(y_va)

    y = np.array(y_all)
    p = np.array(p_all)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(y),
        "auc": roc_auc_score(y, p),
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "accuracy": accuracy_score(y, pred),
        "prec_over": precision_score(y, pred, pos_label=1, zero_division=0),
        "rec_over": recall_score(y, pred, pos_label=1, zero_division=0),
        "matrice": (tn, fp, fn, tp),
    }


def main() -> int:
    df = carica()
    print(f"Righe utilizzabili: {len(df)}  |  base rate Over: {df['y'].mean():.4f}\n")

    risultati = {}
    for nome, cfg in configurazioni(df).items():
        risultati[nome] = valuta(df, cfg["feature"], cfg["pesi"])
        risultati[nome]["n_feature"] = len(cfg["feature"])

    intestazione = f"{'configurazione':28} {'feat':>5} {'AUC':>7} {'logloss':>8} {'brier':>7} {'acc':>7} {'precOver':>9} {'recOver':>8}"
    print(intestazione)
    print("-" * len(intestazione))
    for nome, r in risultati.items():
        print(
            f"{nome:28} {r['n_feature']:5} {r['auc']:7.4f} {r['log_loss']:8.4f} {r['brier']:7.4f} "
            f"{r['accuracy']:7.4f} {r['prec_over']:9.4f} {r['rec_over']:8.4f}"
        )

    print("\nMatrici di confusione (tn, fp, fn, tp):")
    for nome, r in risultati.items():
        print(f"  {nome:28} {r['matrice']}")

    migliore = max(risultati.items(), key=lambda kv: kv[1]["auc"])
    print(f"\nAUC migliore: {migliore[0]} ({migliore[1]['auc']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
