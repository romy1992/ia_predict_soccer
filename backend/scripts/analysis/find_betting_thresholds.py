"""Soglie di decisione orientate alla PRECISIONE sulla classe "Over", per un
uso di betting (l'operatore vuole puntare quando il modello dice "Over": un
falso positivo costa direttamente, quindi conta piu' la precisione del
recall - a differenza di `find_optimal_thresholds.py`, che bilancia Youden
tra le due classi).

Per ciascuna soglia (1.5/2.5/3.5/4.5), riporta la precisione raggiungibile a
diversi livelli minimi di recall (quanti Over reali il modello riesce ancora
a intercettare) dalla curva precision-recall gia' salvata in
`champions_detailed_metrics.json` (nessun nuovo OOF). Recall troppo basso
=> soglia altissima => pochissime predizioni "Over", statisticamente non
affidabile anche se la precisione sembra perfetta: per questo si riporta
sempre anche il numero assoluto di predizioni attese (support), non solo le
percentuali.
"""
from __future__ import annotations

import json
import os

import numpy as np

INPUT_PATH = os.path.join("best_models", "champions_detailed_metrics.json")
OUTPUT_PATH = os.path.join("best_models", "betting_thresholds.json")

RECALL_FLOORS = [0.10, 0.20, 0.30, 0.40, 0.50]


def find_precision_at_recall_floor(pr_precision: np.ndarray, pr_recall: np.ndarray, pr_th: np.ndarray, recall_floor: float) -> dict:
    """Tra tutti i punti della curva PR con recall >= recall_floor, prende
    quello a precisione piu' alta (equivale al threshold piu' alto compatibile
    col vincolo, dato che precision cresce - non monotonicamente ma in
    tendenza - al crescere della soglia)."""
    mask = pr_recall[:-1] >= recall_floor  # ultimo punto (recall=0) non ha soglia associata
    if not mask.any():
        return None
    candidate_idx = np.where(mask)[0]
    best_idx = candidate_idx[np.argmax(pr_precision[candidate_idx])]
    return {
        "threshold": round(float(pr_th[best_idx]), 4),
        "precision": round(float(pr_precision[best_idx]), 4),
        "recall": round(float(pr_recall[best_idx]), 4),
    }


def main() -> None:
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    for market, r in data.items():
        pr_precision = np.array(r["pr_curve"]["precision"])
        pr_recall = np.array(r["pr_curve"]["recall"])
        pr_th = np.array(r["pr_curve"]["thresholds"])
        n_over = r["per_class"]["class_1_over"]["support"]
        base_rate = n_over / r["n_oof"]

        print(f"=== {market} === (base rate Over = {base_rate:.1%}, n_oof={r['n_oof']})")
        market_results = {"base_rate_over": round(base_rate, 4), "n_oof": r["n_oof"], "by_recall_floor": {}}
        for floor in RECALL_FLOORS:
            point = find_precision_at_recall_floor(pr_precision, pr_recall, pr_th, floor)
            if point is None:
                print(f"  recall>={floor:.0%}: non raggiungibile")
                continue
            expected_predictions = round(point["recall"] * n_over)
            point["expected_over_predictions"] = expected_predictions
            market_results["by_recall_floor"][f"{floor:.0%}"] = point
            print(f"  recall>={floor:.0%}: soglia={point['threshold']:.3f} precisione={point['precision']:.1%} "
                  f"(recall reale={point['recall']:.1%}, ~{expected_predictions} predizioni 'Over' attese su {r['n_oof']} partite)")
        results[market] = market_results
        print()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Salvato in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
