"""Addestra e promuove goal_no_goal (solo quote, 6 feat) sotto best_models/goal_no_goal/.

Segue la stessa forma di promuovi_mercato.py, ma:
- feature fisse (solo quote: sono le uniche a battere le statistiche su
  questo mercato, misurato in eda_goal_no_goal.py);
- righe corrotte dal bug Yes/No pre-fix 2026-09-08 scartate (vedi
  report_goal_no_goal_passo1.md);
- salva sotto destination_subdir("goal_no_goal") = "goal_no_goal/", non nella
  radice di best_models (convenzione model_paths.py, MERCATI_CARTELLA_EVENTO).

Uso: python scripts/analysis/promuovi_goal_no_goal.py
"""
import sys, os, json
import numpy as np
import joblib
from unittest.mock import patch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.train_multi_market import train_market
from src.service_ia.training.model_registry import ModelRegistry
from src.service_ia.training.model_paths import destination_subdir, to_container_path

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

    sub = destination_subdir(MERCATO)  # "goal_no_goal"
    nome_file = f"{MERCATO}_champion_{SUFFISSO}"
    filename_rel = os.path.join(sub, nome_file) if sub else nome_file

    with patch.object(FilterMarketService, "build_dataset", return_value=df):
        r = train_market(market=MERCATO, selection_method="kbest", save_model=False, feature_columns=QUOTE)

    print(f"champion: {r.champion}  score: {r.details['champion_selection_score']:.4f}")

    # Ricostruisco e calibro il champion per salvarlo nel path corretto (save_model=False sopra
    # per evitare che train_market scriva in radice; qui replico la logica di SaveLoad ma con
    # destination_subdir).
    from src.service_ia.training.utility_training.save_load import SaveLoad
    from src.ml.calibration.calibration_service import CalibrationService
    from src.service_ia.training.train_multi_market import _build_temporal_cv, _filter_valid_splits

    # Rilancio train_market con save_model=True ma filename standard, poi sposto il file
    # nella sottocartella giusta con path del registry corretto.
    with patch.object(FilterMarketService, "build_dataset", return_value=df):
        r2 = train_market(market=MERCATO, selection_method="kbest", save_model=True, feature_columns=QUOTE)

    registry = ModelRegistry()
    ultimo = registry.get_latest(market=MERCATO)
    vecchio_path = ultimo.get("model_path")
    vecchio_locale = vecchio_path.replace("/app/best_models", "best_models") if vecchio_path and vecchio_path.startswith("/app/") else vecchio_path

    os.makedirs(os.path.join("best_models", sub), exist_ok=True)
    nuovo_locale = os.path.join("best_models", filename_rel + ".pkl")
    if vecchio_locale and os.path.exists(vecchio_locale) and vecchio_locale != nuovo_locale:
        os.replace(vecchio_locale, nuovo_locale)
        print(f"spostato: {vecchio_locale} -> {nuovo_locale}")

    # calibratore
    cal = (ultimo.get("extra") or {}).get("calibration") or {}
    vecchio_cal = cal.get("calibrator_path")
    if vecchio_cal:
        vecchio_cal_locale = vecchio_cal.replace("/app/best_models", "best_models") if vecchio_cal.startswith("/app/") else vecchio_cal
        nuovo_cal_locale = os.path.join("best_models", filename_rel + "_calibrator.pkl")
        if os.path.exists(vecchio_cal_locale) and vecchio_cal_locale != nuovo_cal_locale:
            os.replace(vecchio_cal_locale, nuovo_cal_locale)
            print(f"calibratore spostato: {vecchio_cal_locale} -> {nuovo_cal_locale}")
        cal["calibrator_path"] = to_container_path(nuovo_cal_locale)

    # riscrivo la riga di registry col path corretto (container, sottocartella)
    index_path = os.path.join("best_models", "registry", "index.jsonl")
    righe = [json.loads(l) for l in open(index_path, encoding="utf-8")]
    for riga in righe:
        if riga["run_id"] == ultimo["run_id"]:
            riga["model_path"] = to_container_path(nuovo_locale)
            if cal:
                riga.setdefault("extra", {}).setdefault("calibration", {})["calibrator_path"] = cal.get("calibrator_path")
    with open(index_path, "w", encoding="utf-8") as f:
        for riga in righe:
            f.write(json.dumps(riga, ensure_ascii=False) + "\n")
    print(f"registry riscritto: model_path -> {to_container_path(nuovo_locale)}")

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
