"""Esegue `run_sequential_cascade` (produzione, INVARIATA, vedi
`phase3b_sequential_stacking_thresholds.py`) usando il frame gia' esportato
da `export_datasets_for_cloud_training.py` invece di interrogare il DB - per
ambienti senza accesso di rete diretto al Postgres remoto (vedi
CURRENT_TASK.md, blocco ambiente documentato).

Nessuna logica di cascata/anti-leakage modificata: si sostituisce SOLO la
sorgente dati (il frame + feature_columns, normalmente costruiti da
`build_totals_evaluation_frame(matches=...)` con `matches` dal DB) con il CSV
gia' esportato per lo stesso identico schema di colonne.
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd

from src.ml.validation.temporal_split import expanding_window_splits
from scripts.analysis.phase3b_sequential_stacking_thresholds import (
    GOAL_DISTRIBUTION_AGGREGATE_SCORE_REFERENCE,
    BINARY_INDEPENDENT_OVER_2_5_SCORE_REFERENCE,
    OUTPUT_PATH,
    run_sequential_cascade,
)
from src.ml.markets.totals.totals_market import THRESHOLD_LABELS

EXPORT_DIR = os.path.abspath(os.path.join("scripts", "analysis", "_export"))


def main() -> None:
    t0 = time.time()
    frame = pd.read_csv(os.path.join(EXPORT_DIR, "totals_evaluation_frame.csv"))
    with open(os.path.join(EXPORT_DIR, "totals_feature_columns.json"), "r", encoding="utf-8") as f:
        feature_columns = json.load(f)
    print(f"rows={len(frame)} | feature_columns={len(feature_columns)}")

    min_train = max(30, int(len(frame) * 0.45))
    min_valid = max(10, int(len(frame) * 0.1))
    cv_splits = expanding_window_splits(frame=frame, time_col="prediction_at", n_splits=5, min_train_size=min_train, min_valid_size=min_valid)
    if not cv_splits:
        print("CV insufficiente, esperimento non eseguibile.")
        return

    t1 = time.time()
    result = run_sequential_cascade(frame=frame, feature_columns=feature_columns, cv_splits=cv_splits)
    print(f"cascata sequenziale calcolata ({time.time() - t1:.1f}s)")

    print(f"\n=== Stacking sequenziale a cascata tra soglie (n_oof={result['n_oof']}) ===")
    for label in THRESHOLD_LABELS:
        payload = result["per_threshold"][label]
        cascade = payload["cascade"]
        line = (
            f"{label:>10s}: n_eval_rows={payload['n_eval_rows']} selection_score={cascade['selection_score']:.4f} "
            f"log_loss={cascade['log_loss']:.4f} brier={cascade['brier']:.4f}"
        )
        if payload["baseline_same_rows"] is not None:
            line += (
                f" | baseline(no-cascade, same rows)={payload['baseline_same_rows']['selection_score']:.4f}"
                f" | delta={payload['delta_selection_score']:+.4f}"
            )
        print(line)

    print(f"\naggregate_cascade_score (media 4 soglie) = {result['aggregate_cascade_score']:.4f}")
    print(f"riferimento goal_distribution aggregato    = {GOAL_DISTRIBUTION_AGGREGATE_SCORE_REFERENCE:.4f}")
    print(f"riferimento binary_independent over_2_5    = {BINARY_INDEPENDENT_OVER_2_5_SCORE_REFERENCE:.4f}")
    print(f"\nVERDETTO: {result['verdict']}")

    payload = {**result, "rows": len(frame), "elapsed_seconds": round(time.time() - t0, 2), "source": "export_csv"}
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSalvato in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
