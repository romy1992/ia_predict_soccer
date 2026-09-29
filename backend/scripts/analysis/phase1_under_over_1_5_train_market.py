"""FASE 1 (parte B): train_market('under_over_1_5') con selection_method
sia 'kbest' che 'rfe' - mercato pilota, dati REALI da DB (riusa
`train_multi_market.train_market` cosi' com'e', nessuna nuova pipeline).

Confronta: selection_score/log_loss/brier/ece/auc dei 2 modelli base
(logistic/random_forest) + voting + stacking, per ciascun selection_method.

Output: stampa il confronto e salva in
`best_models/phase1_under_over_1_5_train_market_result.json`.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logging.basicConfig(level=logging.WARNING)

from src.service_ia.training.train_multi_market import train_market

OUTPUT_PATH = os.path.abspath(os.path.join("best_models", "phase1_under_over_1_5_train_market_result.json"))
MARKET = "under_over_1_5"


def _print_result(selection_method: str, result) -> None:
    print(f"\n=== train_market(market='{MARKET}', selection_method='{selection_method}') ===")
    print(f"status={result.status} | rows={result.rows} | champion={result.champion} "
          f"| champion_selection_score={result.details.get('champion_selection_score')}")
    for model_name, payload in (result.details.get("models") or {}).items():
        pm = payload.get("probability_metrics") or {}
        print(
            f"  {model_name:>14s}: selection_score={payload.get('selection_score'):.4f} "
            f"| log_loss={pm.get('log_loss'):.4f} | brier={pm.get('brier'):.4f} "
            f"| ece={pm.get('ece'):.4f} | auc={pm.get('auc')} | best_cv_f1={payload.get('best_cv_f1'):.4f}"
        )
    calibration = result.details.get("calibration") or {}
    print(f"  calibration: enabled={calibration.get('enabled')} method={calibration.get('method')} "
          f"pre_log_loss={calibration.get('pre_metrics', {}).get('log_loss') if calibration.get('pre_metrics') else None} "
          f"post_log_loss={calibration.get('post_metrics', {}).get('log_loss') if calibration.get('post_metrics') else None}")


def main() -> None:
    results: dict[str, Any] = {}

    for selection_method in ("kbest", "rfe"):
        t0 = time.time()
        result = train_market(market=MARKET, seasons=None, selection_method=selection_method, save_model=True)
        elapsed = time.time() - t0
        _print_result(selection_method, result)
        print(f"  elapsed={elapsed:.1f}s")
        results[selection_method] = {
            "market": result.market,
            "rows": result.rows,
            "status": result.status,
            "champion": result.champion,
            "best_cv_f1": result.best_cv_f1,
            "selected_features": result.selected_features,
            "details": result.details,
            "elapsed_seconds": round(elapsed, 2),
        }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSalvato in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

