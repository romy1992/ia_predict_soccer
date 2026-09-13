"""Due misure su Under/Over 1.5, sul dataset pulito (alias unificati, NaN veri).

1) QUANTO VALGONO LE STATISTICHE, e in particolare i cartellini: si parte
   dalle sole quote e si aggiunge un blocco di statistiche alla volta,
   guardando se l'AUC si muove. Serve a rispondere con i numeri alla domanda
   dell'operatore "ma i cartellini hanno senso per un mercato sui gol?".

2) QUANTO INCIDE L'ORDINE TEMPORALE: stessa configurazione valutata con la
   CV walk-forward del progetto (allena sul passato, valida sul futuro) e
   con uno split casuale che ignora il tempo. La differenza fra le due e' la
   misura di quanto si illuderebbe chi non ordina per data.

Uso:
    python scripts/analysis/measure_stats_and_ordering.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.service_ia.training.train_multi_market import _build_temporal_cv, _filter_valid_splits  # noqa: E402

CSV = os.path.join("scripts", "analysis", "_export", "under_over_1_5_raw.csv")
SEED = 42

QUOTE = [
    "prob_norm_over_1_5",
    "odds_mean_over_1_5",
    "odds_mean_under_1_5",
    "odds_count",
    "odds_std_over_1_5",
    "overround",
]

# Blocchi di statistiche, raggruppati per significato calcistico.
BLOCCHI = {
    "tiri": ["shots_on_goal", "total_shots", "shots_insidebox", "shots_off_goal", "shots_outsidebox", "blocked_shots"],
    "attacco/xg": ["expected_goals", "goals_prevented", "goalkeeper_saves"],
    "possesso": ["ball_possession", "passes", "passes_accurate", "total_passes"],
    "disciplina": ["yellow_cards", "red_cards", "fouls"],
    "altro": ["corner_kicks", "offsides"],
}


def colonne(blocco: list[str], df: pd.DataFrame) -> list[str]:
    out = []
    for grandezza in blocco:
        for lato in ("home", "away", "diff"):
            nome = f"mean_{grandezza}_{lato}_stat"
            if nome in df.columns:
                out.append(nome)
    return out


def modello(pesi=None) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=20, random_state=SEED, n_jobs=-1, class_weight=pesi
                ),
            ),
        ]
    )


def auc_temporale(df: pd.DataFrame, feature: list[str]) -> float:
    frame = df.sort_values("prediction_at").reset_index(drop=True)
    splits = _filter_valid_splits(frame["y"], _build_temporal_cv(frame) or [])
    y_all, p_all = [], []
    for tr, va in splits:
        m = modello()
        m.fit(frame.loc[tr, feature], frame.loc[tr, "y"])
        p_all.extend(m.predict_proba(frame.loc[va, feature])[:, 1])
        y_all.extend(frame.loc[va, "y"])
    return roc_auc_score(y_all, p_all)


def auc_casuale(df: pd.DataFrame, feature: list[str]) -> float:
    """Stesso modello, ma con split casuale: il tempo viene ignorato, quindi
    il modello puo' allenarsi su partite SUCCESSIVE a quelle su cui viene
    valutato. E' l'errore che si commette non ordinando per data."""
    frame = df.reset_index(drop=True)
    y_all, p_all = [], []
    for tr, va in KFold(n_splits=5, shuffle=True, random_state=SEED).split(frame):
        m = modello()
        m.fit(frame.loc[tr, feature], frame.loc[tr, "y"])
        p_all.extend(m.predict_proba(frame.loc[va, feature])[:, 1])
        y_all.extend(frame.loc[va, "y"])
    return roc_auc_score(y_all, p_all)


def main() -> int:
    df = pd.read_csv(CSV)
    df = df[df[QUOTE].notna().all(axis=1)].copy()
    print(f"righe utilizzabili: {len(df):,}   base rate Over: {df['y'].mean():.4f}\n")

    print("=" * 72)
    print("1. QUANTO AGGIUNGE OGNI BLOCCO DI STATISTICHE (CV temporale)")
    print("=" * 72)
    base = auc_temporale(df, QUOTE)
    print(f"\n  solo quote ({len(QUOTE)} feature): AUC {base:.4f}\n")
    print(f"  {'aggiungendo':16} {'feature':>8} {'AUC':>8} {'delta':>9}")
    print("  " + "-" * 45)
    risultati = {}
    for nome, blocco in BLOCCHI.items():
        cols = colonne(blocco, df)
        auc = auc_temporale(df, QUOTE + cols)
        risultati[nome] = auc
        print(f"  {nome:16} {len(cols):8} {auc:8.4f} {auc-base:+9.4f}")

    tutte = [c for b in BLOCCHI.values() for c in colonne(b, df)]
    auc_tutte = auc_temporale(df, QUOTE + tutte)
    print(f"\n  {'TUTTE insieme':16} {len(tutte):8} {auc_tutte:8.4f} {auc_tutte-base:+9.4f}")

    senza_disciplina = [c for n, b in BLOCCHI.items() if n != "disciplina" for c in colonne(b, df)]
    auc_senza = auc_temporale(df, QUOTE + senza_disciplina)
    print(f"  {'senza cartellini':16} {len(senza_disciplina):8} {auc_senza:8.4f} {auc_senza-auc_tutte:+9.4f}  <- vs TUTTE")

    print("\n" + "=" * 72)
    print("2. QUANTO INCIDE ORDINARE PER DATA")
    print("=" * 72)
    print("\n  Stessa identica configurazione, due modi di validare:\n")
    print(f"  {'configurazione':22} {'temporale':>11} {'casuale':>11} {'illusione':>11}")
    print("  " + "-" * 58)
    for nome, feature in (("solo quote", QUOTE), ("quote + tutte stat", QUOTE + tutte)):
        t = auc_temporale(df, feature)
        c = auc_casuale(df, feature)
        print(f"  {nome:22} {t:11.4f} {c:11.4f} {c-t:+11.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
