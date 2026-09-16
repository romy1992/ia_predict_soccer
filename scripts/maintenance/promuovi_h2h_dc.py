"""Addestra, registra e promuove i champion h2h e dc in `best_models/<mercato>/`.

COSA FA
Ricostruisce il dataset come phase6b (quote per esito, EDA, random search,
voting/stacking), salva i due campioni in:

    best_models/h2h/h2h_champion_<data>.pkl
    best_models/dc/dc_champion_<data>.pkl

poi li promuove a production. I modelli VECCHI non si cancellano: se non sono
gia' in `archivio/<mercato>/` vengono SPOSTATI li', e le righe di registry che
li puntavano vengono aggiornate. Un file gia' presente in archivio non viene
sovrascritto.

I .pkl non viaggiano per git (`best_models/` e' in .gitignore). Lanciare
questo script sulla macchina dove sta il volume montato sui container
(`./best_models:/app/best_models`) e' quello che allinea l'app. Una sessione
cloud puo' comunque produrre i file locali.

Uso:
    python scripts/maintenance/promuovi_h2h_dc.py            # simulazione
    python scripts/maintenance/promuovi_h2h_dc.py --apply    # esegue
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import date
from typing import Any, Optional
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import joblib
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
from src.service_ia.training.model_paths import (
    ARCHIVIO_DIRNAME,
    CONTAINER_BEST_MODELS,
    destination_subdir,
    relative_to_best_models,
    to_container_path,
)
from src.service_ia.training.model_registry import ModelRegistry
from src.service_ia.training.train_multi_market import train_market
from src.service_ia.training.utility_training.save_load import SaveLoad

BEST = "best_models"
REGISTRY = os.path.join(BEST, "registry")
ARCHIVIO = os.path.join(BEST, ARCHIVIO_DIRNAME)
SUFFISSO = date.today().strftime("%Y%m%d")
MERCATI = ("h2h", "dc")
MOTIVO = (
    "Promozione esplicita dell'operatore: h2h/dc rifatti sul dataset pieno con "
    "quote bookmaker per esito, EDA, RandomizedSearchCV, voting/stacking e "
    "calibrazione isotonica. I champion da 400 righe (69 feat slot mescolate) "
    "restano in archivio, non cancellati."
)


def sposta_senza_cancellare(
    radice: str, sorgente_rel: str, dest_rel: str, applica: bool
) -> tuple[str, str]:
    """Sposta un pkl in archivio. Mai cancella, mai sovrascrive.

    Ritorna (relativo_finale, esito) dove esito e' `moved` / `already_there` /
    `kept_source_collision` / `missing`.
    """
    src = os.path.join(radice, *sorgente_rel.split("/"))
    dst = os.path.join(radice, *dest_rel.split("/"))
    if os.path.abspath(src) == os.path.abspath(dst):
        return dest_rel, "already_there"
    if not os.path.exists(src):
        return dest_rel, "missing"
    if os.path.exists(dst):
        return sorgente_rel, "kept_source_collision"
    if applica:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
    return dest_rel, "moved"


def destinazione_archivio(relativo: str, mercato: str) -> str:
    return f"{ARCHIVIO_DIRNAME}/{mercato}/{os.path.basename(relativo)}"


def campi_percorso(riga: dict[str, Any]) -> list[tuple[str, Optional[str]]]:
    cal = (riga.get("extra") or {}).get("calibration") or {}
    return [
        ("model_path", riga.get("model_path")),
        ("metadata_path", riga.get("metadata_path")),
        ("calibrator_path", cal.get("calibrator_path")),
    ]


def imposta_percorso(riga: dict[str, Any], campo: str, valore: str) -> None:
    if campo == "calibrator_path":
        extra = riga.setdefault("extra", {})
        cal = extra.setdefault("calibration", {})
        cal["calibrator_path"] = valore
        return
    riga[campo] = valore


def archivia_vecchi(registry: ModelRegistry, mercati: tuple[str, ...], applica: bool) -> list[dict[str, str]]:
    """Sposta i pkl h2h/dc che non sono la production corrente.

    Non tocca altri mercati. Non cancella niente in `archivio/`.
    """
    index = os.path.join(REGISTRY, "index.jsonl")
    if not os.path.exists(index):
        print("   registry assente: niente da archiviare")
        return []

    righe = [json.loads(l) for l in open(index, encoding="utf-8") if l.strip()]
    in_produzione: set[str] = set()
    for mercato in mercati:
        prod = registry.get_production(market=mercato)
        if not prod:
            continue
        for campo, valore in campi_percorso(prod):
            relativo = relative_to_best_models(valore or "")
            if relativo:
                in_produzione.add(relativo)

    report: list[dict[str, str]] = []
    aggiornate = 0
    for riga in righe:
        if riga.get("market") not in mercati:
            continue
        for campo, valore in campi_percorso(riga):
            relativo = relative_to_best_models(valore or "")
            if not relativo or relativo in in_produzione:
                continue
            if relativo.startswith(f"{ARCHIVIO_DIRNAME}/"):
                continue
            dest = destinazione_archivio(relativo, riga["market"])
            finale, esito = sposta_senza_cancellare(BEST, relativo, dest, applica)
            report.append({"from": relativo, "to": finale, "esito": esito, "campo": campo})
            if esito in {"moved", "missing", "already_there"}:
                nuovo = f"{CONTAINER_BEST_MODELS}/{finale}"
                if valore != nuovo:
                    imposta_percorso(riga, campo, nuovo)
                    aggiornate += 1

    print(f"   file considerati: {len(report)}   righe aggiornate: {aggiornate}")
    for voce in report:
        print(f"      {voce['esito']:22} {voce['from']} -> {voce['to']}")
    if applica and aggiornate:
        with open(index, "w", encoding="utf-8") as f:
            for riga in righe:
                f.write(json.dumps(riga, ensure_ascii=False) + "\n")
        print(f"   {index} riscritto")
    elif not applica:
        print("   (simulazione: niente spostato, niente riscritto)")
    return report


def riscrivi_percorsi_container() -> int:
    index = os.path.join(REGISTRY, "index.jsonl")
    if not os.path.exists(index):
        return 0
    righe = [json.loads(l) for l in open(index, encoding="utf-8") if l.strip()]
    corrette = 0
    for riga in righe:
        for campo, valore in campi_percorso(riga):
            if not valore:
                continue
            nuovo = to_container_path(valore)
            if nuovo != valore:
                imposta_percorso(riga, campo, nuovo)
                corrette += 1
    if corrette:
        with open(index, "w", encoding="utf-8") as f:
            for riga in righe:
                f.write(json.dumps(riga, ensure_ascii=False) + "\n")
    return corrette


def backup_registry(applica: bool) -> str:
    dest = os.path.join(REGISTRY, f"_backup_{date.today():%Y%m%d}_h2h_dc")
    print(f"\nbackup del registry in {dest}")
    if not applica:
        print("   (simulazione: non copiato)")
        return dest
    os.makedirs(dest, exist_ok=True)
    for nome in ("index.jsonl", "promotion_history.jsonl"):
        src = os.path.join(REGISTRY, nome)
        if not os.path.exists(src):
            continue
        shutil.copy2(src, os.path.join(dest, nome))
        print(f"   {nome}: copiato")
    return dest


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
    percorso_cal = os.path.abspath(os.path.join(BEST, nome_cal))
    os.makedirs(os.path.dirname(percorso_cal), exist_ok=True)
    joblib.dump(result.estimator, percorso_cal)

    cal = (result.details or {}).get("calibration") or {}
    cal = dict(cal)
    cal["calibrator_path"] = percorso_cal
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
        from src.service_ia.training.model_paths import resolve_model_path

        locale = resolve_model_path(prod.get("model_path"))
        if not locale or not os.path.exists(locale):
            print(f"   {mercato}: file mancante {prod.get('model_path')}")
            continue
        modello = joblib.load(locale)
        n_feat = len(prod.get("feature_names") or [])
        import numpy as np

        X = pd.DataFrame(np.zeros((3, n_feat)), columns=prod["feature_names"])
        proba = modello.predict_proba(X)[:, 1]
        print(
            f"   {mercato}: {os.path.relpath(locale, BEST)}  "
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

    backup_registry(applica)
    esiti = {}
    for mercato in MERCATI:
        frame, features = prepara_frame(mercato)
        esiti[mercato] = addestra_e_salva(mercato, frame, features, applica)

    if applica:
        n = riscrivi_percorsi_container()
        print(f"\npercorsi riscritti in forma container: {n}")
        registry = ModelRegistry()
        for mercato in MERCATI:
            promuovi(registry, mercato, applica)
        # Archivia DOPO la promozione: la production nuova resta dove e',
        # i pkl superati vanno in archivio/<mercato>/ senza cancellarli.
        print("\narchiviazione dei modelli h2h/dc non piu' in produzione")
        archivia_vecchi(ModelRegistry(), MERCATI, applica)
        n = riscrivi_percorsi_container()
        print(f"percorsi container dopo archivio: {n}")
        verifica(ModelRegistry())
    else:
        print("\nSimulazione conclusa. Rilancia con --apply per eseguire.")
        print("Archivio: nessun file verrebbe cancellato; i vecchi pkl verrebbero spostati in archivio/<mercato>/. ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
