"""EDA + misura feature per goal_no_goal (esiti goal/no_goal, non over/under).

Riusa BLOCCHI/_build_temporal_cv di rifacimento_mercato.py ma con le colonne
quote proprie del mercato (prob_norm_goal, odds_mean_goal/no_goal, ecc.),
perche' quote_per_linea() li' e' scritta per over_/under_.

Uso: python scripts/analysis/eda_goal_no_goal.py
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.analysis.rifacimento_mercato import BLOCCHI, colonne
from src.service_ia.training.train_multi_market import _build_temporal_cv, _filter_valid_splits

CSV = "scripts/analysis/_export/goal_no_goal_raw.csv"
QUOTE = ["prob_norm_goal", "odds_mean_goal", "odds_mean_no_goal", "odds_count", "odds_std_goal", "overround"]
SEED = 42

def righe_sospette(df):
    r_g = df["odds_max_goal"] / df["odds_mean_goal"].replace(0, np.nan)
    r_n = df["odds_max_no_goal"] / df["odds_mean_no_goal"].replace(0, np.nan)
    return ((df["overround"] < 1.0) | (r_g > 3.0) | (r_n > 3.0)).fillna(False)

def modello():
    return Pipeline([("i", SimpleImputer(strategy="median")),
                      ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=SEED, n_jobs=-1))])

def oof(df, feat):
    frame = df.sort_values("prediction_at").reset_index(drop=True)
    splits = _filter_valid_splits(frame["y"], _build_temporal_cv(frame) or [])
    y_all, p_all = [], []
    for tr, va in splits:
        m = modello()
        m.fit(frame.loc[tr, feat], frame.loc[tr, "y"])
        p_all.extend(m.predict_proba(frame.loc[va, feat])[:, 1])
        y_all.extend(frame.loc[va, "y"])
    return np.array(y_all), np.array(p_all)

def main():
    df = pd.read_csv(CSV)
    df = df[df[QUOTE].notna().all(axis=1)].copy()
    sosp = righe_sospette(df)
    print(f"righe con quote: {len(df):,}   base rate: {df['y'].mean():.4f}")
    print(f"righe sospette: {int(sosp.sum())} ({sosp.mean():.2%})")
    df = df[~sosp].copy()

    print("\n--- Passo 4a: blocchi statistiche ---")
    y, p = oof(df, QUOTE)
    base = roc_auc_score(y, p)
    print(f"solo quote (6 feat): AUC {base:.4f}")
    for nome, blocco in BLOCCHI.items():
        cols = colonne(blocco, df)
        y, p = oof(df, QUOTE + cols)
        print(f"  +{nome:14} {len(cols):3} feat  AUC {roc_auc_score(y,p):.4f}  delta {roc_auc_score(y,p)-base:+.4f}")
    tutte = [c for b in BLOCCHI.values() for c in colonne(b, df)]
    y, p = oof(df, QUOTE + tutte)
    print(f"  TUTTE            {len(tutte):3} feat  AUC {roc_auc_score(y,p):.4f}  delta {roc_auc_score(y,p)-base:+.4f}")

    print("\n--- Passo 5: confronto set (AUC/logloss/brier) ---")
    TIRI = ["shots_on_goal","total_shots","shots_insidebox","shots_off_goal","shots_outsidebox","blocked_shots"]
    DISC = ["yellow_cards","red_cards","fouls"]
    for nome, feat in [("quote", QUOTE),
                        ("quote+tiri", QUOTE+colonne(TIRI,df)),
                        ("quote+tiri+disciplina", QUOTE+colonne(TIRI+DISC,df)),
                        ("quote+tutte", QUOTE+tutte)]:
        y, p = oof(df, feat)
        print(f"  {nome:24} {len(feat):3} feat  AUC {roc_auc_score(y,p):.4f}  logloss {log_loss(y,p):.4f}  brier {brier_score_loss(y,p):.4f}")

if __name__ == "__main__":
    main()
