"""Training reale su DB per Corners/Cards a linea configurabile (MARKET-05/06,
2026-09-12) - richiesto esplicitamente dall'operatore ("procediamo con
l'implementazione... vorrei avere tutte le metriche possibili per ogni
contesto").

CLI wrapper attorno a `run_corners_benchmark_from_db`/`run_cards_benchmark_from_db`
(gia' esistenti, MAI eseguiti prima d'ora contro dati reali - stesso principio
"una sola funzione, riusata da manuale/job/API, mai duplicata" gia' seguito
ovunque nel progetto): per ciascuna linea configurata, addestra+calibra il
modello, stampa il report di classificazione COMPLETO (log loss/Brier/ECE/AUC
pre/post calibrazione, accuracy, confusion matrix, precision/recall/F1 per
classe e pesati, ROC/PR + AUC, soglia ottimale di Youden) e - salvo
`--dry-run` - registra ciascuna linea come modello 'candidate' separato nel
registry (mai promosso automaticamente a production: quella e' sempre una
decisione esplicita dell'operatore via `ModelRegistry.promote_with_policy`).

NOTA: richiede un accesso DB reale (Postgres Railway) - non eseguibile dalla
sessione cloud di sviluppo (nessun accesso di rete diretto, stesso blocco gia'
documentato per le migration in IMPLEMENTATION_LOG.md). Da lanciare
dall'operatore in locale o da una sessione con accesso DB reale.
"""

from __future__ import annotations

import argparse
import json
import os

from src.ml.markets.cards.cards_market import DEFAULT_LINES as CARDS_DEFAULT_LINES
from src.ml.markets.cards.cards_market import run_cards_benchmark_from_db
from src.ml.markets.corners.corners_market import DEFAULT_LINES as CORNERS_DEFAULT_LINES
from src.ml.markets.corners.corners_market import run_corners_benchmark_from_db

_RUNNERS = {
    "corners": (run_corners_benchmark_from_db, CORNERS_DEFAULT_LINES),
    "cards": (run_cards_benchmark_from_db, CARDS_DEFAULT_LINES),
}


def _parse_lines(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _print_line_report(market: str, label: str, entry: dict) -> None:
    print(f"--- {market} / {label} (linea {entry['line']}) ---", flush=True)
    print(f"sample_size={entry['sample_size']} | calibrazione={entry['calibration_method']}")

    pre, post = entry["pre_metrics"], entry["post_metrics"]
    print(
        f"log_loss  pre={pre.get('log_loss'):.4f}  post={post.get('log_loss'):.4f}\n"
        f"brier     pre={pre.get('brier'):.4f}  post={post.get('brier'):.4f}\n"
        f"ece       pre={pre.get('ece'):.4f}  post={post.get('ece'):.4f}\n"
        f"auc       pre={pre.get('auc')}  post={post.get('auc')}"
    )

    classification = entry["classification_report_post"]
    if classification["status"] != "ok":
        print(f"classificazione (post): {classification['status']} - metriche a soglia non disponibili\n", flush=True)
        return

    cm = classification["confusion_matrix"]
    optimal = classification["optimal_threshold"]
    print(
        f"accuracy@0.5={classification['accuracy']:.4f} | weighted_f1@0.5={classification['weighted']['f1']:.4f} | "
        f"pr_auc={classification['pr_curve']['average_precision']:.4f}"
    )
    print(f"confusion_matrix@0.5: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    print(
        f"soglia ottimale (Youden J)={optimal['threshold']:.4f} | "
        f"accuracy={optimal['accuracy']:.4f} | weighted_f1={optimal['weighted']['f1']:.4f}\n",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", default="corners,cards", help="Sottoinsieme di 'corners,cards' (default: entrambi)")
    parser.add_argument("--seasons", default=None, help="Lista stagioni separate da virgola (default: tutte)")
    parser.add_argument("--corners-lines", default=None, help=f"Linee corners separate da virgola (default: {CORNERS_DEFAULT_LINES})")
    parser.add_argument("--cards-lines", default=None, help=f"Linee cards separate da virgola (default: {CARDS_DEFAULT_LINES})")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Addestra e stampa il report SENZA registrare i modelli (nessuna scrittura su best_models/registry)",
    )
    parser.add_argument("--output", default=os.path.join("best_models", "corners_cards_benchmark_report.json"))
    args = parser.parse_args()

    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    unknown = set(markets) - set(_RUNNERS)
    if unknown:
        raise SystemExit(f"Mercati sconosciuti: {sorted(unknown)} (validi: {sorted(_RUNNERS)})")

    seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()] if args.seasons else None
    line_overrides = {"corners": args.corners_lines, "cards": args.cards_lines}

    results: dict[str, dict] = {}
    for market in markets:
        runner, default_lines = _RUNNERS[market]
        lines = _parse_lines(line_overrides[market]) if line_overrides[market] else default_lines

        print(f"=== {market} (linee: {lines}) ===", flush=True)
        result = runner(seasons=seasons, lines=lines, save_model=not args.dry_run)
        results[market] = {
            "status": result.status,
            "rows": result.rows,
            "lines_trained": result.lines_trained,
            "details": result.details,
        }

        if result.status != "benchmarked":
            print(f"SALTATO ({result.status}): {result.details}\n", flush=True)
            continue

        metrics_summary = result.details.get("metrics_summary", {})
        for label, entry in metrics_summary.items():
            _print_line_report(market=market, label=label, entry=entry)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Salvato in {args.output}")
    if args.dry_run:
        print("--dry-run attivo: nessun modello registrato (ri-eseguire senza --dry-run per candidare i modelli).")


if __name__ == "__main__":
    main()
