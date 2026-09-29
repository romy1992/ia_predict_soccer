"""EDA di un mercato, sul dataset GREZZO (valori mancanti preservati).

Richiesta esplicita dell'operatore (2026-09-13): prima di riaddestrare
qualunque cosa, capire per ogni mercato quanti dati ci sono, quanti mancano,
come sono correlati e quali feature sono ridondanti - "perche' se mi dici
7000 valori mancanti allora non ha senso usare un SimpleImputer con mediana".

Va eseguito sul CSV prodotto da `build_dataset(fill_missing=False)`: sul CSV
normale i mancanti sono gia' stati azzerati e l'analisi non direbbe nulla.

Uso:
    python scripts/analysis/eda_market.py under_over_1_5
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

EXPORT_DIR = os.path.join("scripts", "analysis", "_export")
META = {"y", "market", "id_fixture", "season", "league", "prediction_at"}


def blocco(titolo: str) -> None:
    print(f"\n{'=' * 78}\n{titolo}\n{'=' * 78}")


def famiglia(colonna: str) -> str:
    if colonna.endswith("_stat"):
        return colonna.rsplit("_", 2)[0].replace("mean_", "")
    return "QUOTE"


def main() -> int:
    if len(sys.argv) < 2:
        print("Uso: python scripts/analysis/eda_market.py <mercato>")
        return 1
    mercato = sys.argv[1]
    percorso = os.path.join(EXPORT_DIR, f"{mercato}_raw.csv")
    if not os.path.exists(percorso):
        print(f"Manca il dataset grezzo: {percorso}")
        print("Va prodotto con build_dataset(fill_missing=False) sulla macchina che vede il DB.")
        return 1

    df = pd.read_csv(percorso)

    blocco(f"1. STRUTTURA  —  {mercato}")
    print(f"righe: {len(df):,}   colonne: {df.shape[1]}   memoria: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
    print(f"periodo: {df['prediction_at'].min()[:10]} -> {df['prediction_at'].max()[:10]}")
    print(f"leghe distinte: {df['league'].nunique()}   stagioni: {sorted(df['season'].dropna().unique().tolist())}")
    print("\ntarget y (value_counts):")
    vc = df["y"].value_counts(dropna=False).sort_index()
    for valore, quante in vc.items():
        print(f"   y={valore}: {quante:6,}  ({quante/len(df):6.2%})")

    blocco("2. VALORI MANCANTI (sul dato grezzo, prima di qualunque riempimento)")
    mancanti = df.isna().sum()
    mancanti = mancanti[mancanti > 0].sort_values(ascending=False)
    if mancanti.empty:
        print("Nessun valore mancante: sospetto, verificare di aver usato fill_missing=False.")
    else:
        print(f"colonne con almeno un mancante: {len(mancanti)} su {df.shape[1]}")
        print(f"celle mancanti in totale: {int(df.isna().sum().sum()):,}\n")
        print(f"{'colonna':42} {'mancanti':>9} {'%':>8}")
        print("-" * 62)
        for col, quanti in mancanti.items():
            print(f"{col:42} {quanti:9,} {quanti/len(df):8.1%}")

    blocco("3. MANCANTI AGGREGATI PER GRANDEZZA (home+away+diff insieme)")
    stat_cols = [c for c in df.columns if c.endswith("_stat")]
    per_fam = {}
    for c in stat_cols:
        per_fam.setdefault(famiglia(c), []).append(c)
    righe = []
    for fam, cols in per_fam.items():
        quota_mancante = df[cols].isna().any(axis=1).mean()
        righe.append((fam, quota_mancante, df[cols[0]].notna().sum()))
    righe.sort(key=lambda r: -r[1])
    print(f"{'grandezza':24} {'righe con almeno un buco':>26} {'righe utilizzabili':>20}")
    print("-" * 74)
    for fam, quota, utili in righe:
        print(f"{fam:24} {quota:25.1%} {utili:20,}")

    blocco("4. CORRELAZIONE CON IL TARGET (Spearman, solo righe valorizzate)")
    numeriche = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    corr = {}
    for c in numeriche:
        valide = df[[c, "y"]].dropna()
        if len(valide) > 200 and valide[c].nunique() > 1:
            corr[c] = valide[c].corr(valide["y"], method="spearman")
    serie = pd.Series(corr).dropna().sort_values(key=np.abs, ascending=False)
    print("Le 20 piu' correlate (in valore assoluto):\n")
    print(f"{'colonna':42} {'corr':>8} {'n':>9}")
    print("-" * 62)
    for col, valore in serie.head(20).items():
        print(f"{col:42} {valore:8.4f} {df[[col,'y']].dropna().shape[0]:9,}")
    print(f"\ncorrelazione |r| > 0.05: {(serie.abs() > 0.05).sum()} colonne su {len(serie)}")
    print(f"correlazione |r| > 0.02: {(serie.abs() > 0.02).sum()} colonne su {len(serie)}")

    blocco("5. RIDONDANZA FRA FEATURE (coppie con |corr| > 0.95)")
    base = [c for c in numeriche if df[c].notna().sum() > len(df) * 0.5]
    matrice = df[base].corr(method="spearman").abs()
    viste = set()
    coppie = []
    for i, a in enumerate(base):
        for b in base[i + 1:]:
            valore = matrice.loc[a, b]
            if pd.notna(valore) and valore > 0.95:
                coppie.append((a, b, valore))
    coppie.sort(key=lambda t: -t[2])
    if not coppie:
        print("Nessuna coppia sopra 0.95.")
    else:
        print(f"{len(coppie)} coppie quasi identiche (una delle due e' ridondante):\n")
        for a, b, valore in coppie[:25]:
            print(f"  {valore:.4f}   {a:38} <-> {b}")
        if len(coppie) > 25:
            print(f"  ... e altre {len(coppie)-25}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
