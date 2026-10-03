"""Addestra e promuove goal_no_goal (solo quote, 6 feat) sotto best_models/goal_no_goal/.

Segue la stessa forma di promuovi_mercato.py, ma:
- feature fisse (solo quote: sono le uniche a battere le statistiche su
  questo mercato, misurato in eda_goal_no_goal.py);
- righe corrotte dal bug Yes/No pre-fix 2026-09-08 scartate (vedi
  report_goal_no_goal_passo1.md);
- salva sotto destination_subdir("goal_no_goal") = "goal_no_goal/", non nella
  radice di best_models (convenzione model_paths.py, MERCATI_CARTELLA_EVENTO).

Post-migrazione al Bucket (2026-10): `train_market(save_model=True)` scrive
gia' alla chiave corretta (`destination_subdir` e' applicato internamente),
quindi un solo giro basta - niente piu' doppio training + spostamento
manuale del file + riscrittura a mano della riga di registry.

Uso: python scripts/analysis/promuovi_goal_no_goal.py
"""
import sys, os
import numpy as np
from unittest.mock import patch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.train_multi_market import train_market
from src.service_ia.training.model_registry import ModelRegistry

MERCATO = "goal_no_goal"
QUOTE = ["prob_norm_goal", "odds_mean_goal", "odds_mean_no_goal", "odds_count", "odds_std_goal", "overround"]
CSV = "scripts/analysis/_export/goal_no_goal_raw.csv"
SUFFISSO = "20260916"


def main():
    df = pd.read_csv(CSV)
    df = df[df[QUOTE].notna().all(axis=1)].copy()
    r_g = df["odds_max_goal"] / df["odds_mean_goal"].replace(0, np.nan)
    r_n = df["odds_max_no_goal"] / df["odds_mean_no_goal"].replace(0, np.nan)
    sosp = ((df["overround"] < 1.0) | (r_g > 3.0) | (r_n > 3.0)).fillna(False)
    df = df[~sosp].copy()
    print(f"righe: {len(df):,}  base rate: {df['y'].mean():.4f}")

    with patch.object(FilterMarketService, "build_dataset", return_value=df):
        r = train_market(market=MERCATO, selection_method="kbest", save_model=True, feature_columns=QUOTE)

    print(f"champion: {r.champion}  score: {r.details['champion_selection_score']:.4f}")

    registry = ModelRegistry()
    ultimo = registry.get_latest(market=MERCATO)
    print(f"run registrato: {ultimo['run_id']}")
    print(f"   chiave bucket: {ultimo.get('model_path')}")

    esito = registry.promote_with_policy(
        run_id=ultimo["run_id"], to_stage="production",
        reason="goal_no_goal rifatto: solo quote (6 feat, uniche a battere le statistiche), "
               "righe corrotte dal bug Yes/No pre-fix scartate. Salvato sotto best_models/goal_no_goal/.",
        actor="operator_request",
    )
    print(f"\nesito promozione: allowed={esito.get('promoted') if esito else esito}")
    if esito:
        ev = esito.get("evaluation", {})
        print(f"gate passed: {ev.get('gate',{}).get('passed')}  comparison: {ev.get('comparison',{})}")
    return esito


if __name__ == "__main__":
    main()
