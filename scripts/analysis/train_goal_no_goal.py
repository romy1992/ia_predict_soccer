"""Training goal_no_goal: solo quote (6 feat), uniche a battere le statistiche.
Righe corrotte pre-fix 2026-09-08 (bug Yes/No, vedi report) scartate.
"""
import sys, os, json
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.train_multi_market import train_market

CSV = "scripts/analysis/_export/goal_no_goal_raw.csv"
QUOTE = ["prob_norm_goal", "odds_mean_goal", "odds_mean_no_goal", "odds_count", "odds_std_goal", "overround"]

def main():
    df = pd.read_csv(CSV)
    df = df[df[QUOTE].notna().all(axis=1)].copy()
    r_g = df["odds_max_goal"] / df["odds_mean_goal"].replace(0, np.nan)
    r_n = df["odds_max_no_goal"] / df["odds_mean_no_goal"].replace(0, np.nan)
    sosp = ((df["overround"] < 1.0) | (r_g > 3.0) | (r_n > 3.0)).fillna(False)
    df = df[~sosp].copy()
    print(f"righe: {len(df):,}  base rate: {df['y'].mean():.4f}  feature: {len(QUOTE)}")

    with patch.object(FilterMarketService, "build_dataset", return_value=df):
        r = train_market(market="goal_no_goal", selection_method="kbest", save_model=True, feature_columns=QUOTE)

    print(f"\nchampion: {r.champion}  score: {r.details['champion_selection_score']:.4f}")
    for k, v in r.details["models"].items():
        m = v.get("probability_metrics") or {}
        f = lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x)
        print(f"  {k:22} score {f(v.get('selection_score'))}  auc {f(m.get('auc'))}  logloss {f(m.get('log_loss'))}  brier {f(m.get('brier'))}  ece {f(m.get('ece'))}")
    c = r.details.get("calibration") or {}
    if c.get("enabled"):
        print(f"\ncalibrazione {c['method']}: ece {c['pre_metrics']['ece']:.4f} -> {c['post_metrics']['ece']:.4f}")

    dest = "scripts/analysis/_export/train_goal_no_goal_selected.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump({"market": "goal_no_goal", "rows": len(df), "feature_columns": QUOTE,
                    "status": r.status, "champion": r.champion, "details": r.details}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nreport: {dest}")

if __name__ == "__main__":
    main()
