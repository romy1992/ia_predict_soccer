"""Esegue `train_market()` (produzione, INVARIATA) per il mercato
"goal_no_goal" usando il CSV gia' esportato da
`export_goal_no_goal_for_cloud_training.py` invece di interrogare il DB -
stesso pattern di `train_from_export.py` gia' usato per Under/Over.

Nessuna logica di training modificata: si sostituisce SOLO la sorgente dati
(`FilterMarketService.build_dataset`, monkeypatchata per la durata della
singola chiamata) con il CSV gia' esportato - stesso schema di colonne che
l'implementazione DB produrrebbe.
"""
from __future__ import annotations

import json
import os
import time
from unittest.mock import patch

import pandas as pd

from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.train_multi_market import train_market

EXPORT_DIR = os.path.abspath(os.path.join("scripts", "analysis", "_export"))
MARKET = "goal_no_goal"


def main() -> None:
    csv_path = os.path.join(EXPORT_DIR, f"{MARKET}.csv")
    df = pd.read_csv(csv_path)
    print(f"\n=== {MARKET}: {len(df)} righe caricate da {csv_path} ===", flush=True)

    t0 = time.time()
    with patch.object(FilterMarketService, "build_dataset", return_value=df):
        result = train_market(market=MARKET, selection_method="kbest", save_model=True)
    elapsed = time.time() - t0

    print(
        f"{MARKET}: status={result.status} champion={result.champion} "
        f"selection_score={result.details.get('champion_selection_score')} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    for model_name, payload in (result.details.get("models") or {}).items():
        print(f"  {model_name}: selection_score={payload.get('selection_score')}", flush=True)

    summary = {
        "status": result.status,
        "champion": result.champion,
        "selection_score": result.details.get("champion_selection_score"),
        "elapsed_seconds": round(elapsed, 1),
        "models": {k: v.get("selection_score") for k, v in (result.details.get("models") or {}).items()},
    }
    output_path = os.path.abspath(os.path.join("best_models", "train_goal_no_goal_summary.json"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nRiepilogo salvato in {output_path}")


if __name__ == "__main__":
    main()
