"""Metriche di classificazione dettagliate (precision/recall/F1 per classe,
confusion matrix, curva ROC+AUC) per i modelli 'champion' gia' registrati.

CLI wrapper attorno a `src.ml.evaluation.model_diagnostics_service` (2026-09-12,
riusato SENZA duplicare dall'endpoint `GET /models/diagnostics` - stesso
principio "una sola funzione, riusata da manuale e API" gia' seguito ovunque
nel progetto). Legge dal DB reale via `FilterMarketService.build_dataset`
(non piu' dai CSV di `export_datasets_for_cloud_training.py`, rimossi dal
repo dopo l'uso una tantum che li aveva generati).
"""
from __future__ import annotations

import argparse
import json
import os

from src.ml.evaluation.model_diagnostics_service import evaluate_market_diagnostics, list_diagnosable_markets
from src.service_ia.training.model_registry import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--markets",
        default=None,
        help="Lista mercati separati da virgola (default: tutti quelli diagnosticabili - vedi list_diagnosable_markets)",
    )
    parser.add_argument("--seasons", default=None, help="Lista stagioni separate da virgola (default: tutte)")
    parser.add_argument("--output", default=os.path.join("best_models", "champions_detailed_metrics.json"))
    parser.add_argument(
        "--merge", action="store_true", help="Unisce ai risultati gia' presenti in --output invece di sovrascriverli"
    )
    args = parser.parse_args()

    registry = ModelRegistry()
    markets = [m.strip() for m in args.markets.split(",") if m.strip()] if args.markets else list_diagnosable_markets(registry=registry)
    seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()] if args.seasons else None

    results = {}
    if args.merge and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            results = json.load(f)

    for market in markets:
        print(f"=== {market} ===", flush=True)
        result = evaluate_market_diagnostics(market=market, seasons=seasons, registry=registry)
        results[market] = result.to_dict()

        if result.status != "ok":
            print(f"SALTATO ({result.status})\n", flush=True)
            continue

        cm = result.cm
        print(f"n_oof={result.n_oof} | champion={result.champion} | stage={result.stage} | accuracy={result.accuracy:.4f} | auc={result.auc:.4f}")
        print(f"confusion_matrix: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
        print(f"class 0: precision={result.class0['precision']:.4f} recall={result.class0['recall']:.4f} f1={result.class0['f1']:.4f}")
        print(f"class 1: precision={result.class1['precision']:.4f} recall={result.class1['recall']:.4f} f1={result.class1['f1']:.4f}")
        print(f"weighted avg: precision={result.weighted['precision']:.4f} recall={result.weighted['recall']:.4f} f1={result.weighted['f1']:.4f}\n", flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Salvato in {args.output}")


if __name__ == "__main__":
    main()
