"""Trova la soglia di decisione ottimale per ciascuno dei 4 champion Under/Over,
in alternativa al cutoff fisso p >= 0.5 usato di default (che, come mostrato in
`evaluate_champions_detailed.py`, lascia 3 modelli su 4 praticamente ciechi
sulla classe minoritaria: la P(Over)/P(Under) prevista raramente attraversa
0.5 anche quando il modello, guardando l'AUC, distingue le due classi).

Criterio: statistica J di Youden (max(tpr - fpr) sulla curva ROC) - la soglia
che massimizza il bilanciamento tra le due classi, non solo l'accuratezza
complessiva (che con classi sbilanciate premia banalmente prevedere sempre la
classe maggioritaria). Ricalcola la confusion matrix e precision/recall/F1 per
classe A QUELLA soglia dai punti gia' presenti in
`champions_detailed_metrics.json` (nessun nuovo giro di training/OOF).

Non modifica alcuna soglia di default nel codice di produzione: e' un
risultato di analisi, la soglia operativa va scelta esplicitamente (qui o in
serving) quando/se questi modelli verranno promossi.
"""
from __future__ import annotations

import json
import os

import numpy as np

INPUT_PATH = os.path.join("best_models", "champions_detailed_metrics.json")
OUTPUT_PATH = os.path.join("best_models", "optimal_thresholds.json")


def find_youden_threshold(market: str, r: dict) -> dict:
    fpr = np.array(r["roc_curve"]["fpr"])
    tpr = np.array(r["roc_curve"]["tpr"])
    roc_th = np.array(r["roc_curve"]["thresholds"])
    j = tpr - fpr
    j_idx = int(np.argmax(j))
    threshold = float(roc_th[j_idx])

    n_under = r["per_class"]["class_0_under"]["support"]
    n_over = r["per_class"]["class_1_over"]["support"]
    tp = tpr[j_idx] * n_over
    fn = n_over - tp
    fp = fpr[j_idx] * n_under
    tn = n_under - fp

    prec_under = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_under = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_under = 2 * prec_under * rec_under / (prec_under + rec_under + 1e-12)

    prec_over = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_over = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_over = 2 * prec_over * rec_over / (prec_over + rec_over + 1e-12)

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "market": market,
        "threshold": round(threshold, 4),
        "youden_j": round(float(j[j_idx]), 4),
        "accuracy": round(float(accuracy), 4),
        "confusion_matrix": {"tn": round(tn), "fp": round(fp), "fn": round(fn), "tp": round(tp)},
        "class_0_under": {"precision": round(prec_under, 4), "recall": round(rec_under, 4), "f1": round(f1_under, 4)},
        "class_1_over": {"precision": round(prec_over, 4), "recall": round(rec_over, 4), "f1": round(f1_over, 4)},
        "baseline_at_0_5": {
            "accuracy": r["accuracy"],
            "confusion_matrix": r["confusion_matrix"],
            "class_0_under": {k: r["per_class"]["class_0_under"][k] for k in ("precision", "recall", "f1")},
            "class_1_over": {k: r["per_class"]["class_1_over"][k] for k in ("precision", "recall", "f1")},
        },
    }


def main() -> None:
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    for market, r in data.items():
        result = find_youden_threshold(market, r)
        results[market] = result
        print(f"=== {market} ===")
        print(f"soglia ottimale (Youden J) = {result['threshold']:.3f}  (default 0.5)")
        print(f"accuracy: {result['accuracy']:.3f} (era {result['baseline_at_0_5']['accuracy']:.3f} a 0.5)")
        print(f"under: precision={result['class_0_under']['precision']:.3f} recall={result['class_0_under']['recall']:.3f} "
              f"(era recall={result['baseline_at_0_5']['class_0_under']['recall']:.3f} a 0.5)")
        print(f"over : precision={result['class_1_over']['precision']:.3f} recall={result['class_1_over']['recall']:.3f} "
              f"(era recall={result['baseline_at_0_5']['class_1_over']['recall']:.3f} a 0.5)")
        print()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Salvato in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
