"""Riscrive `model_path` in `best_models/registry/index.jsonl` per puntare
alla cartella `best_models/` LOCALE (calcolata sulla macchina che esegue
questo script), non a quella della sessione cloud dove i modelli sono
stati addestrati/registrati in origine.

Senza questo fix, `os.path.exists(model_path)` fallisce sempre sulla
macchina dell'operatore (percorso Linux della sessione cloud, es.
`/home/user/ia_predict_soccer/best_models/...`, inesistente su Windows) e
`DashboardService`/`DirectMarketExpert` non caricano mai nessun modello -
sintomo: "Nessuna previsione"/N-D per ogni fixture nonostante il registry
elenchi correttamente 5 mercati con modello.

DA ESEGUIRE UNA VOLTA IN LOCALE, dopo aver ricomposto i `.pkl` in
`best_models/` (`reassemble_model_files.py`) e copiato
`index.jsonl`/`promotion_history.jsonl` in `best_models/registry/`.
Nessun accesso DB, nessun addestramento - solo riscrittura di un file di
testo, idempotente (rieseguirlo non fa danni)."""
from __future__ import annotations

import json
import os

INDEX_PATH = os.path.join("best_models", "registry", "index.jsonl")
MODELS_DIR = os.path.abspath("best_models")


def main() -> None:
    if not os.path.exists(INDEX_PATH):
        raise SystemExit(
            f"Non trovato: {INDEX_PATH} - esegui questo script dalla radice del repo, "
            "dopo aver copiato index.jsonl in best_models/registry/."
        )

    rows: list[dict] = []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    fixed = 0
    missing_files = []
    for row in rows:
        old_path = row.get("model_path")
        if not old_path:
            continue
        filename = os.path.basename(old_path)
        new_path = os.path.join(MODELS_DIR, filename)
        if new_path != old_path:
            row["model_path"] = new_path
            fixed += 1
        if not os.path.exists(new_path):
            missing_files.append(filename)

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Aggiornati {fixed}/{len(rows)} model_path -> {MODELS_DIR}")

    if missing_files:
        print("\nATTENZIONE: questi file NON esistono ancora in best_models/ "
              "(il path e' stato comunque corretto, ma il file .pkl manca davvero):")
        for name in sorted(set(missing_files)):
            print(f"  - {name}")
    else:
        print("Tutti i file .pkl referenziati esistono in best_models/.")


if __name__ == "__main__":
    main()
