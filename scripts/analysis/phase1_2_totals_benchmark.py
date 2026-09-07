"""FASE 1+2 (parte A): run_totals_benchmark_from_db su TUTTE le 4 soglie.

Riusa direttamente `run_totals_benchmark_from_db` (nessuna nuova pipeline):
confronta binary_independent / hierarchical / goal_distribution sullo
STESSO walk-forward per le 4 soglie insieme, salva il vincitore come
'candidate' in ModelRegistry (mai 'production' automatica).

Output: stampa il report e lo salva in
`best_models/phase1_2_totals_benchmark_result.json`.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import time

logging.basicConfig(level=logging.WARNING)

from src.ml.markets.totals.totals_market import run_totals_benchmark_from_db

OUTPUT_PATH = os.path.abspath(os.path.join("best_models", "phase1_2_totals_benchmark_result.json"))


def main() -> None:
    t0 = time.time()
    print("Avvio run_totals_benchmark_from_db(seasons=None, save_model=True) ...")
    result = run_totals_benchmark_from_db(seasons=None, save_model=True)
    elapsed = time.time() - t0

    print(f"status={result.status} | rows={result.rows} | best_approach={result.best_approach} | elapsed={elapsed:.1f}s")
    if result.details.get("approach_aggregate_scores"):
        for approach, score in result.details["approach_aggregate_scores"].items():
            print(f"  approach_aggregate_score[{approach}] = {score:.4f}")
    if result.details.get("best_approach_by_threshold"):
        print("  best_approach_by_threshold:")
        for label, approach in result.details["best_approach_by_threshold"].items():
            print(f"    {label}: {approach}")
    if result.details.get("monotonicity_violations_before_projection"):
        print("  monotonicity_violations_before_projection:", result.details["monotonicity_violations_before_projection"])
    if result.details.get("run"):
        print("  registered run_id:", result.details["run"].get("run_id"))

    payload = {
        "market": result.market,
        "rows": result.rows,
        "status": result.status,
        "best_approach": result.best_approach,
        "elapsed_seconds": round(elapsed, 2),
        "details": result.details,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"Salvato in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

