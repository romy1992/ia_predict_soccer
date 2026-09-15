"""Confronto dei set di feature su Under/Over 1.5, sul dataset pulito.

Segue la misura per blocchi (`measure_stats_and_ordering.py`), che aveva
mostrato due cose: i blocchi di statistiche sono largamente intercambiabili
(ciascuno porta allo stesso AUC, tutti insieme non fanno meglio dei soli
tiri) e i cartellini contribuiscono quanto i tiri. Qui si verifica se un set
ridotto regge quanto quello completo.

Oltre alle metriche di probabilita' riporta la curva PRECISIONE/VOLUME sulla
classe Over: e' quella che conta per l'operatore, che su questo mercato
vuole puntare l'Over. Con un base rate del 77.3% un modello che dice sempre
Over ha gia' 77.3% di precisione: la soglia serve a superarlo, e il prezzo
e' il numero di partite su cui ci si sbilancia.

Stessa CV walk-forward del training di produzione, stesso modello e stesso
seed per tutte le configurazioni: l'unica variabile e' il set di feature.

Uso:
    python scripts/analysis/compare_feature_sets_over_1_5.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
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

TIRI = ["shots_on_goal", "total_shots", "shots_insidebox", "shots_off_goal", "shots_outsidebox", "blocked_shots"]
DISCIPLINA = ["yellow_cards", "red_cards", "fouls"]
XG = ["expected_goals", "goals_prevented", "goalkeeper_saves"]
POSSESSO = ["ball_possession", "passes", "passes_accurate", "total_passes"]
ALTRO = ["corner_kicks", "offsides"]


def colonne(grandezze: list[str], df: pd.DataFrame) -> list[str]:
    return [
        f"mean_{g}_{lato}_stat"
        for g in grandezze
        for lato in ("home", "away", "diff")
        if f"mean_{g}_{lato}_stat" in df.columns
    ]


def oof(df: pd.DataFrame, feature: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frame = df.sort_values("prediction_at").reset_index(drop=True)
    splits = _filter_valid_splits(frame["y"], _build_temporal_cv(frame) or [])
    y_all, p_all = [], []
    for tr, va in splits:
        modello = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=SEED, n_jobs=-1)),
            ]
        )
        modello.fit(frame.loc[tr, feature], frame.loc[tr, "y"])
        p_all.extend(modello.predict_proba(frame.loc[va, feature])[:, 1])
        y_all.extend(frame.loc[va, "y"])
    return np.array(y_all), np.array(p_all)


def main() -> int:
    df = pd.read_csv(CSV)
    df = df[df[QUOTE].notna().all(axis=1)].copy()
    base_rate = df["y"].mean()
    print(f"righe: {len(df):,}   base rate Over 1.5: {base_rate:.4f}\n")

    configurazioni = {
        "quote": QUOTE,
        "quote + tiri": QUOTE + colonne(TIRI, df),
        "quote + tiri + disciplina": QUOTE + colonne(TIRI + DISCIPLINA, df),
        "quote + tutte le stat": QUOTE + colonne(TIRI + DISCIPLINA + XG + POSSESSO + ALTRO, df),
    }

    print("=" * 78)
    print("QUALITA' DELLA PROBABILITA'")
    print("=" * 78)
    print(f"\n{'configurazione':28} {'feat':>5} {'AUC':>8} {'logloss':>9} {'brier':>8}")
    print("-" * 62)
    risultati = {}
    for nome, feature in configurazioni.items():
        y, p = oof(df, feature)
        risultati[nome] = (y, p)
        print(f"{nome:28} {len(feature):5} {roc_auc_score(y,p):8.4f} {log_loss(y,p):9.4f} {brier_score_loss(y,p):8.4f}")

    print("\n" + "=" * 78)
    print("PRECISIONE / VOLUME SULLA CLASSE OVER  (cio' che conta per puntare)")
    print("=" * 78)
    print(f"\nUn modello che dice SEMPRE Over avrebbe precisione {base_rate:.1%} su {len(df):,} partite.")
    print("Per essere utile la precisione deve superare quella riga.\n")

    for nome, (y, p) in risultati.items():
        print(f"{nome}:")
        print(f"   {'soglia':>8} {'precisione':>12} {'partite':>10} {'% del totale':>14} {'guadagno':>10}")
        print("   " + "-" * 58)
        for soglia in (0.50, 0.80, 0.85, 0.90, 0.95):
            scelte = p >= soglia
            n = int(scelte.sum())
            if n < 30:
                print(f"   {soglia:8.2f} {'-':>12} {n:10} {'troppo poche':>14}")
                continue
            precisione = y[scelte].mean()
            print(
                f"   {soglia:8.2f} {precisione:11.1%} {n:10,} {n/len(y):13.1%} {precisione-base_rate:+9.1%}"
            )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
