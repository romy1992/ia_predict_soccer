"""Passi 4 e 5 della procedura di rifacimento, per un mercato qualsiasi.

Generalizza `measure_stats_and_ordering.py` e `compare_feature_sets_over_1_5.py`,
scritti su misura per Under/Over 1.5: qui il mercato e la linea sono argomenti,
cosi' la stessa misura si ripete identica su 2.5, 3.5, 4.5 e sui restanti
mercati senza riscrivere niente (richiesta dell'operatore: "questo deve essere
fatto per ogni mercato", "seguire sempre gli stessi passaggi").

Tre blocchi:

  4a. quanto aggiunge ogni famiglia di statistiche, partendo dalle sole quote;
  4b. quanto incide validare per data invece che a caso;
  5.  confronto dei set candidati, con la curva precisione/volume sulle DUE
      classi (su un mercato bilanciato come 2.5 non c'e' una classe
      "maggioritaria" da battere: vanno guardate entrambe).

In piu', rispetto alla versione 1.5, tratta come VARIABILE MISURATA la
pulizia delle quote contaminate (vedi `report_quote_2_5.md`): invece di
decidere a giudizio se scartare le righe sospette, addestra con e senza e
guarda la differenza.

Uso:
    python scripts/analysis/rifacimento_mercato.py under_over_2_5 2_5
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.service_ia.training.train_multi_market import _build_temporal_cv, _filter_valid_splits  # noqa: E402

EXPORT = os.path.join("scripts", "analysis", "_export")
SEED = 42

BLOCCHI = {
    "tiri": ["shots_on_goal", "total_shots", "shots_insidebox", "shots_off_goal", "shots_outsidebox", "blocked_shots"],
    "attacco/xg": ["expected_goals", "goals_prevented", "goalkeeper_saves"],
    "possesso": ["ball_possession", "passes", "passes_accurate", "total_passes"],
    "disciplina": ["yellow_cards", "red_cards", "fouls"],
    "altro": ["corner_kicks", "offsides"],
}


def quote_per_linea(linea: str) -> list[str]:
    """Le sei colonne di quota tenute su Under/Over 1.5, riparametrizzate.

    Volutamente NON si includono `implied_prob_*` ne' `odds_min/max`: la EDA
    le ha mostrate identiche (|corr| > 0.99) a colonne gia' presenti, e dare
    al modello la stessa informazione due volte non aggiunge segnale.
    """
    return [
        f"prob_norm_over_{linea}",
        f"odds_mean_over_{linea}",
        f"odds_mean_under_{linea}",
        "odds_count",
        f"odds_std_over_{linea}",
        "overround",
    ]


def colonne(grandezze: list[str], df: pd.DataFrame) -> list[str]:
    return [
        f"mean_{g}_{lato}_stat"
        for g in grandezze
        for lato in ("home", "away", "diff")
        if f"mean_{g}_{lato}_stat" in df.columns
    ]


def modello() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=SEED, n_jobs=-1)),
        ]
    )


def oof(df: pd.DataFrame, feature: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frame = df.sort_values("prediction_at").reset_index(drop=True)
    splits = _filter_valid_splits(frame["y"], _build_temporal_cv(frame) or [])
    y_all, p_all = [], []
    for tr, va in splits:
        m = modello()
        m.fit(frame.loc[tr, feature], frame.loc[tr, "y"])
        p_all.extend(m.predict_proba(frame.loc[va, feature])[:, 1])
        y_all.extend(frame.loc[va, "y"])
    return np.array(y_all), np.array(p_all)


def oof_casuale(df: pd.DataFrame, feature: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frame = df.reset_index(drop=True)
    y_all, p_all = [], []
    for tr, va in KFold(n_splits=5, shuffle=True, random_state=SEED).split(frame):
        m = modello()
        m.fit(frame.loc[tr, feature], frame.loc[tr, "y"])
        p_all.extend(m.predict_proba(frame.loc[va, feature])[:, 1])
        y_all.extend(frame.loc[va, "y"])
    return np.array(y_all), np.array(p_all)


def righe_sospette(df: pd.DataFrame, linea: str) -> pd.Series:
    """Quote impossibili da un bookmaker reale.

    `overround < 1` significa che la somma delle probabilita' implicite sta
    sotto il 100%: nessun bookmaker quota in perdita, quindi o una quota e'
    gonfiata o le quote mediate non sono dello stesso evento. La soglia su
    `odds_max` prende i casi eclatanti (un "Over 2.5" a 23.00) anche quando
    l'overround resta sopra 1 perche' l'altro esito compensa.
    """
    return (
        (df["overround"] < 1.0)
        | (df[f"odds_max_over_{linea}"] > 6.0)
        | (df[f"odds_max_under_{linea}"] > 8.0)
    )


def curva(y: np.ndarray, p: np.ndarray, etichetta: str) -> None:
    base = y.mean()
    print(f"\n  {etichetta}")
    print(f"     {'soglia':>7} {'precisione':>11} {'partite':>9} {'% tot':>7} {'vs base':>9}")
    print("     " + "-" * 46)
    for soglia in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        scelte = p >= soglia
        n = int(scelte.sum())
        if n < 30:
            print(f"     {soglia:7.2f} {'-':>11} {n:9} {'poche':>7}")
            continue
        prec = y[scelte].mean()
        print(f"     {soglia:7.2f} {prec:10.1%} {n:9,} {n/len(y):6.1%} {prec-base:+8.1%}")


def curva_negativa(y: np.ndarray, p: np.ndarray, etichetta: str) -> None:
    """Stessa curva sulla classe 0 (Under): si scommette quando la
    probabilita' di Over e' BASSA, e la precisione e' sugli Under azzeccati."""
    base = 1 - y.mean()
    print(f"\n  {etichetta}")
    print(f"     {'soglia':>7} {'precisione':>11} {'partite':>9} {'% tot':>7} {'vs base':>9}")
    print("     " + "-" * 46)
    for soglia in (0.50, 0.45, 0.40, 0.35, 0.30, 0.25):
        scelte = p <= soglia
        n = int(scelte.sum())
        if n < 30:
            print(f"     {soglia:7.2f} {'-':>11} {n:9} {'poche':>7}")
            continue
        prec = 1 - y[scelte].mean()
        print(f"     {soglia:7.2f} {prec:10.1%} {n:9,} {n/len(y):6.1%} {prec-base:+8.1%}")


def main() -> int:
    if len(sys.argv) < 3:
        print("Uso: python scripts/analysis/rifacimento_mercato.py <mercato> <linea>")
        print("  es: python scripts/analysis/rifacimento_mercato.py under_over_2_5 2_5")
        return 1
    mercato, linea = sys.argv[1], sys.argv[2]
    percorso = os.path.join(EXPORT, f"{mercato}_raw.csv")
    if not os.path.exists(percorso):
        print(f"Manca il dataset grezzo: {percorso}")
        return 1

    df = pd.read_csv(percorso)
    QUOTE = quote_per_linea(linea)
    mancanti = [c for c in QUOTE if c not in df.columns]
    if mancanti:
        print(f"Colonne di quota assenti: {mancanti}")
        return 1

    df = df[df[QUOTE].notna().all(axis=1)].copy()
    sospette = righe_sospette(df, linea)
    df_pulito = df[~sospette].copy()

    print(f"mercato: {mercato}   linea: {linea}")
    print(f"righe con quote: {len(df):,}   base rate y=1: {df['y'].mean():.4f}")
    print(f"righe sospette : {sospette.sum():,} ({sospette.mean():.2%})   base rate su queste: {df.loc[sospette,'y'].mean():.4f}")
    print(f"righe pulite   : {len(df_pulito):,}   base rate: {df_pulito['y'].mean():.4f}")

    tutte = [c for b in BLOCCHI.values() for c in colonne(b, df)]

    print("\n" + "=" * 74)
    print("PASSO 3-bis. SCARTARE LE QUOTE CONTAMINATE CAMBIA QUALCOSA?")
    print("=" * 74)
    print("\nStesso modello, stesse feature (quote + tutte le statistiche):\n")
    print(f"  {'dataset':28} {'righe':>8} {'AUC':>8} {'logloss':>9} {'brier':>8}")
    print("  " + "-" * 65)
    for nome, frame in (("con le righe sospette", df), ("senza le righe sospette", df_pulito)):
        y, p = oof(frame, QUOTE + tutte)
        print(f"  {nome:28} {len(frame):8,} {roc_auc_score(y,p):8.4f} {log_loss(y,p):9.4f} {brier_score_loss(y,p):8.4f}")

    print("\n" + "=" * 74)
    print("PASSO 4a. QUANTO AGGIUNGE OGNI FAMIGLIA DI STATISTICHE")
    print("=" * 74)
    y, p = oof(df_pulito, QUOTE)
    base_auc = roc_auc_score(y, p)
    print(f"\n  solo quote ({len(QUOTE)} feature): AUC {base_auc:.4f}\n")
    print(f"  {'aggiungendo':16} {'feature':>8} {'AUC':>8} {'delta':>9}")
    print("  " + "-" * 45)
    for nome, blocco in BLOCCHI.items():
        cols = colonne(blocco, df_pulito)
        y, p = oof(df_pulito, QUOTE + cols)
        auc = roc_auc_score(y, p)
        print(f"  {nome:16} {len(cols):8} {auc:8.4f} {auc-base_auc:+9.4f}")
    y, p = oof(df_pulito, QUOTE + tutte)
    auc_tutte = roc_auc_score(y, p)
    print(f"\n  {'TUTTE insieme':16} {len(tutte):8} {auc_tutte:8.4f} {auc_tutte-base_auc:+9.4f}")

    print("\n" + "=" * 74)
    print("PASSO 4b. QUANTO INCIDE ORDINARE PER DATA")
    print("=" * 74)
    print(f"\n  {'configurazione':22} {'temporale':>11} {'casuale':>11} {'illusione':>11}")
    print("  " + "-" * 58)
    for nome, feature in (("solo quote", QUOTE), ("quote + tutte stat", QUOTE + tutte)):
        yt, pt = oof(df_pulito, feature)
        yc, pc = oof_casuale(df_pulito, feature)
        t, c = roc_auc_score(yt, pt), roc_auc_score(yc, pc)
        print(f"  {nome:22} {t:11.4f} {c:11.4f} {c-t:+11.4f}")

    print("\n" + "=" * 74)
    print("PASSO 5. CONFRONTO DEI SET CANDIDATI (sul dataset pulito)")
    print("=" * 74)
    configurazioni = {
        "solo quota media": [f"odds_mean_over_{linea}"],
        "quote": QUOTE,
        "quote + tiri": QUOTE + colonne(BLOCCHI["tiri"], df_pulito),
        "quote + tiri + disciplina": QUOTE + colonne(BLOCCHI["tiri"] + BLOCCHI["disciplina"], df_pulito),
        "quote + tutte le stat": QUOTE + tutte,
    }
    print(f"\n{'configurazione':28} {'feat':>5} {'AUC':>8} {'logloss':>9} {'brier':>8}")
    print("-" * 62)
    risultati = {}
    for nome, feature in configurazioni.items():
        y, p = oof(df_pulito, feature)
        risultati[nome] = (y, p)
        print(f"{nome:28} {len(feature):5} {roc_auc_score(y,p):8.4f} {log_loss(y,p):9.4f} {brier_score_loss(y,p):8.4f}")

    base_over = df_pulito["y"].mean()
    print("\n" + "=" * 74)
    print("PRECISIONE / VOLUME — LE DUE DIREZIONI")
    print("=" * 74)
    print(f"\nDire SEMPRE Over: precisione {base_over:.1%}.  Dire SEMPRE Under: {1-base_over:.1%}.")
    print("Su un mercato bilanciato nessuna delle due e' una scorciatoia: contano entrambe.")
    for nome, (y, p) in risultati.items():
        if nome == "solo quota media":
            continue
        print(f"\n{nome}:")
        curva(y, p, "puntando OVER (soglia = probabilita' minima di Over)")
        curva_negativa(y, p, "puntando UNDER (soglia = probabilita' massima di Over)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
