"""Esegue `train_market()` (produzione, INVARIATA) usando i CSV gia'
esportati da `export_datasets_for_cloud_training.py` invece di interrogare il
DB - per ambienti senza accesso di rete diretto al Postgres remoto (vedi
CURRENT_TASK.md, blocco ambiente documentato).

Nessuna logica di training modificata: si sostituisce SOLO la sorgente dati
(`FilterMarketService.build_dataset`, monkeypatchata per la durata della
singola chiamata) con il CSV gia' esportato per quel mercato - stesso schema
di colonne che l'implementazione DB produrrebbe. Una soglia alla volta, in
isolamento (come richiesto), risultati riassunti a fine esecuzione.
"""
from __future__ import annotations

import json
import os
import sys
import time
from unittest.mock import patch

import pandas as pd

from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.train_multi_market import train_market

EXPORT_DIR = os.path.abspath(os.path.join("scripts", "analysis", "_export"))
MARKETS = ["under_over_1_5", "under_over_2_5", "under_over_3_5", "under_over_4_5"]


def main() -> None:
    results = {}
    for market in MARKETS:
        csv_path = os.path.join(EXPORT_DIR, f"{market}.csv")
        df = pd.read_csv(csv_path)
        print(f"\n=== {market}: {len(df)} righe caricate da {csv_path} ===", flush=True)

        t0 = time.time()
        try:
            with patch.object(FilterMarketService, "build_dataset", return_value=df):
                result = train_market(market=market, selection_method="kbest", save_model=True)
            elapsed = time.time() - t0
            print(
                f"{market}: status={result.status} champion={result.champion} "
                f"selection_score={result.details.get('champion_selection_score')} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            for model_name, payload in (result.details.get("models") or {}).items():
                print(f"  {model_name}: selection_score={payload.get('selection_score')}", flush=True)
            results[market] = {
                "status": result.status,
                "champion": result.champion,
                "selection_score": result.details.get("champion_selection_score"),
                "elapsed_seconds": round(elapsed, 1),
                "models": {k: v.get("selection_score") for k, v in (result.details.get("models") or {}).items()},
            }
        except Exception as exc:  # noqa: BLE001 - vogliamo l'errore esatto per soglia, senza abortire le altre
            elapsed = time.time() - t0
            print(f"{market}: FALLITO dopo {elapsed:.1f}s - {type(exc).__name__}: {exc}", flush=True)
            results[market] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": round(elapsed, 1)}

    output_path = os.path.abspath(os.path.join("best_models", "train_from_export_summary.json"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nRiepilogo salvato in {output_path}")

    if any(r.get("status") == "failed" for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
