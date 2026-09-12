"""Esegue il training reale (walk-forward OOF + calibrazione, INVARIATO) di
Corners/Cards (MARKET-05/06) usando i CSV gia' esportati da
`export_corners_cards_for_cloud_training.py`, invece di interrogare il DB -
stesso identico pattern di `train_from_export.py` (Under/Over 1.5-4.5): il
training pesante gira qui, in un ambiente cloud che non dipende dalla
stabilita' della connessione di rete della macchina locale dell'operatore
(causa di ripetuti fallimenti di training diretto su bridge in passato, vedi
IMPLEMENTATION_LOG.md 2026-09-07/08).

Nessuna logica di training modificata: `run_corners_benchmark`/
`run_cards_benchmark` (produzione, INVARIATE a parte il nuovo parametro
additivo `frame=`, vedi corners_market.py/cards_market.py) vengono chiamate
passando DIRETTAMENTE il DataFrame gia' esportato invece di ricostruirlo da
`matches` - stesso schema di colonne che l'interrogazione DB reale
produrrebbe (`build_corners_frame_from_records`/`build_cards_frame_from_records`
sono state chiamate identiche in fase di export)."""

from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd

from src.ml.markets.cards.cards_market import DEFAULT_LINES as CARDS_LINES
from src.ml.markets.cards.cards_market import run_cards_benchmark
from src.ml.markets.corners.corners_market import DEFAULT_LINES as CORNERS_LINES
from src.ml.markets.corners.corners_market import run_corners_benchmark

EXPORT_DIR = os.path.abspath(os.path.join("scripts", "analysis", "_export"))
_RUNNERS = {
    "corners": (run_corners_benchmark, CORNERS_LINES),
    "cards": (run_cards_benchmark, CARDS_LINES),
}


def _load_frame(csv_path: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    # `expanding_window_splits`/`ensure_temporal_order` ricoercono comunque
    # `prediction_at` a datetime internamente - nessun re-parsing manuale
    # necessario qui, stesso comportamento del CSV di export Under/Over.
    return frame


def main() -> None:
    save_model = "--dry-run" not in sys.argv
    results: dict[str, dict] = {}

    for market, (runner, lines) in _RUNNERS.items():
        csv_path = os.path.join(EXPORT_DIR, f"{market}.csv")
        frame = _load_frame(csv_path)
        print(f"\n=== {market}: {len(frame)} righe caricate da {csv_path} ===", flush=True)

        t0 = time.time()
        try:
            result = runner(lines=lines, save_model=save_model, frame=frame)
            elapsed = time.time() - t0
            print(f"{market}: status={result.status} linee_addestrate={result.lines_trained} elapsed={elapsed:.1f}s", flush=True)
            results[market] = {
                "status": result.status,
                "rows": result.rows,
                "lines_trained": result.lines_trained,
                "elapsed_seconds": round(elapsed, 1),
                "details": result.details,
            }
        except Exception as exc:  # noqa: BLE001 - vogliamo l'errore esatto per mercato, senza abortire l'altro
            elapsed = time.time() - t0
            print(f"{market}: FALLITO dopo {elapsed:.1f}s - {type(exc).__name__}: {exc}", flush=True)
            results[market] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": round(elapsed, 1)}

    output_path = os.path.abspath(os.path.join("best_models", "corners_cards_from_export_summary.json"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nRiepilogo salvato in {output_path}")
    if not save_model:
        print("--dry-run attivo: nessun modello registrato.")

    if any(r.get("status") == "failed" for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
