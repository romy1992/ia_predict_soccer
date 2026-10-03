"""Addestra, registra e promuove i champion h2h e dc in `best_models/<mercato>/`.

COSA FA
Ricostruisce il dataset come phase6b (quote per esito, EDA, random search,
voting/stacking), salva i due campioni sul bucket:

    best_models/h2h/h2h_champion_<data>.pkl
    best_models/dc/dc_champion_<data>.pkl

poi li promuove a production. I modelli vecchi NON vengono piu' spostati in
un `archivio/<mercato>/` locale (post-migrazione al Bucket, 2026-10): il
registry vive sul bucket ed e' condiviso da `api`/`scheduler`/locale senza
bisogno di riorganizzare file - restano semplicemente alla loro chiave
originale, nessun danno (storage bucket economico, nessun impatto sul
serving: solo il run in stage `production` viene servito).

Uso:
    python scripts/maintenance/promuovi_h2h_dc.py            # simulazione
    python scripts/maintenance/promuovi_h2h_dc.py --apply    # esegue
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from typing import Any
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd

from scripts.analysis.phase6_h2h_dc_retrain_and_cascade import (
    DC_REQUIRED_ODDS,
    H2H_REQUIRED_ODDS,
    _prepare_frame,
)
from scripts.analysis.phase6b_full_pipeline import (
    RANDOM_SEARCH_ITER,
    eda_report,
    features_after_eda,
)
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.model_paths import destination_subdir, resolve_model_key
from src.service_ia.training.model_registry import ModelRegistry
from src.service_ia.training.train_multi_market import train_market
from src.service_ia.training.utility_training.save_load import SaveLoad
from src.storage import bucket_store

BEST = "best_models"
SUFFISSO = date.today().strftime("%Y%m%d")
MERCATI = ("h2h", "dc")
MOTIVO = (
    "Promozione esplicita dell'operatore: h2h/dc rifatti sul dataset pieno con "
    "quote bookmaker per esito, EDA, RandomizedSearchCV, voting/stacking e "
    "calibrazione isotonica."
)


def prepara_frame(mercato: str) -> tuple[pd.DataFrame, list[str]]:
    service = FilterMarketService()
    raw = service.build_dataset(market=mercato, fill_missing=False)
    required = H2H_REQUIRED_ODDS if mercato == "h2h" else DC_REQUIRED_ODDS
    frame = _prepare_frame(raw, required)
    eda = eda_report(frame, f"{mercato}_dopo_filtro_quote")
    features = features_after_eda(frame, mercato, eda)
    frame = frame.fillna(0)
    print(f"   {mercato}: {len(frame)} righe, {len(features)} feature dopo EDA")
    return frame, features


def addestra_e_salva(mercato: str, frame: pd.DataFrame, features: list[str], applica: bool) -> dict[str, Any]:
    print(f"\n=== train_market({mercato}) random search iter={RANDOM_SEARCH_ITER} ===")
    if not applica:
        print("   (simulazione: training non eseguito)")
        return {"status": "dry_run", "market": mercato, "n_features": len(features), "n_rows": len(frame)}

    with patch.object(FilterMarketService, "build_dataset", return_value=frame):
        result = train_market(
            market=mercato,
            selection_method="kbest",
            save_model=False,
            feature_columns=features,
            search_strategy="random",
            random_search_iter=RANDOM_SEARCH_ITER,
        )
    if result.status != "trained" or result.estimator is None:
        raise RuntimeError(f"{mercato}: training fallito ({result.status})")

    sotto = destination_subdir(mercato)
    nome_modello = os.path.join(sotto, f"{mercato}_champion_{SUFFISSO}")
    nome_cal = os.path.join(sotto, f"{mercato}_champion_calibrator_{SUFFISSO}.pkl")
    chiave_cal = f"{BEST}/{nome_cal}".replace("\\", "/")
    bucket_store.put_joblib(chiave_cal, result.estimator)

    cal = (result.details or {}).get("calibration") or {}
    cal = dict(cal)
    cal["calibrator_path"] = chiave_cal
    champion_payload = ((result.details or {}).get("models") or {}).get(result.champion) or {}
    prob = champion_payload.get("probability_metrics") or {}
    post = cal.get("post_metrics") or {}

    saver = SaveLoad(
        save_pkl=True,
        filename=nome_modello,
        market_name=mercato,
        feature_names=features,
        metrics={
            "selection_metric": "composite_probability_score",
            "selection_score": (result.details or {}).get("champion_selection_score"),
            "best_cv_f1": result.best_cv_f1,
            "log_loss": post.get("log_loss", prob.get("log_loss")),
            "brier": post.get("brier", prob.get("brier")),
            "ece": post.get("ece", prob.get("ece")),
            "auc": post.get("auc", prob.get("auc")),
            "calibration_enabled": cal.get("enabled"),
            "calibration_method": cal.get("method"),
            "model_family": result.champion,
            "rows": result.rows,
            "selected_features_count": len(result.selected_features or []),
        },
        registry_enabled=True,
    )
    saver.save_model(
        estimator=result.estimator,
        model_name=result.champion,
        params=champion_payload.get("best_params"),
        extra={
            "selected_features": result.selected_features,
            "selection_method": "kbest",
            "selection_in_pipeline": True,
            "search_strategy": "random",
            "random_search_iter": RANDOM_SEARCH_ITER,
            "calibration": cal,
        },
    )
    print(f"   champion={result.champion} salvato in {BEST}/{nome_modello}.pkl")
    return {
        "status": result.status,
        "champion": result.champion,
        "rows": result.rows,
        "n_features": len(features),
        "model_rel": f"{nome_modello}.pkl".replace("\\", "/"),
        "calibrator_rel": nome_cal.replace("\\", "/"),
        "selection_score": (result.details or {}).get("champion_selection_score"),
        "best_params": champion_payload.get("best_params"),
    }


def promuovi(registry: ModelRegistry, mercato: str, applica: bool) -> dict[str, Any]:
    if not applica:
        print(f"   {mercato}: (simulazione) promozione saltata")
        return {"promoted": False, "dry_run": True}
    ultimo = registry.get_latest(market=mercato)
    if not ultimo or SUFFISSO not in str(ultimo.get("model_path") or ""):
        raise RuntimeError(f"{mercato}: l'ultimo run non e' quello appena salvato: {ultimo}")
    esito = registry.promote_with_policy(
        run_id=ultimo["run_id"],
        to_stage="production",
        reason=MOTIVO,
        actor="operator_request",
    )
    if not esito.get("promoted"):
        print(f"   {mercato}: policy ha rifiutato, forzo su richiesta dell'operatore")
        print(f"      evaluation={esito.get('evaluation')}")
        esito = registry.promote_with_policy(
            run_id=ultimo["run_id"],
            to_stage="production",
            reason=MOTIVO + " Policy rifiutata; forzatura esplicita.",
            actor="operator_request",
            force=True,
        )
    print(f"   {mercato}: promoted={esito.get('promoted')} run={esito.get('run_id')}")
    return esito


def verifica(registry: ModelRegistry) -> None:
    print("\nverifica finale")
    for mercato in MERCATI:
        prod = registry.get_production(market=mercato)
        if not prod:
            print(f"   {mercato}: NESSUNA production")
            continue

        chiave = resolve_model_key(prod.get("model_path"))
        if not chiave:
            print(f"   {mercato}: chiave non trovata sul bucket {prod.get('model_path')}")
            continue
        modello = bucket_store.get_joblib(chiave)
        n_feat = len(prod.get("feature_names") or [])
        import numpy as np

        X = pd.DataFrame(np.zeros((3, n_feat)), columns=prod["feature_names"])
        proba = modello.predict_proba(X)[:, 1]
        print(
            f"   {mercato}: {chiave}  "
            f"feat={n_feat}  p={list(map(float, proba.round(3)))}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="esegue davvero (default: simulazione)")
    args = parser.parse_args()
    applica = args.apply

    print(f"suffisso file: {SUFFISSO}   apply={applica}")
    print("destinazioni:")
    for mercato in MERCATI:
        print(f"   {mercato} -> best_models/{destination_subdir(mercato)}/")

    esiti = {}
    for mercato in MERCATI:
        frame, features = prepara_frame(mercato)
        esiti[mercato] = addestra_e_salva(mercato, frame, features, applica)

    if applica:
        registry = ModelRegistry()
        for mercato in MERCATI:
            promuovi(registry, mercato, applica)
        verifica(ModelRegistry())
    else:
        print("\nSimulazione conclusa. Rilancia con --apply per eseguire.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
