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
testo, idempotente (rieseguirlo non fa danni).

Setup Docker (docker-compose.yml): il servizio `api` monta `./best_models`
su `/app/best_models` DENTRO il container - e' il container, non l'host
Windows/Linux, a servire le predizioni, quindi il path corretto in
QUELL'ambiente e' sempre `/app/best_models/...` (fisso, indipendente dalla
macchina host). Passa `--target-dir /app/best_models` in quel caso, invece
di lasciare il default (che assume lo script esegua nello stesso ambiente
che serve i modelli, es. un run diretto senza Docker)."""
from __future__ import annotations

import argparse
import json
import os

INDEX_PATH = os.path.join("best_models", "registry", "index.jsonl")
LOCAL_CHECK_DIR = os.path.abspath("best_models")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-dir",
        default=LOCAL_CHECK_DIR,
        help="Cartella best_models/ COSI' COME LA VEDE il processo che serve i modelli "
             "(es. /app/best_models se l'api gira in un container Docker con quel volume montato). "
             "Il controllo 'il file esiste?' resta invece SEMPRE sul filesystem locale "
             "(best_models/ accanto a questo script), dato che e' li' che il file va posizionato "
             "davvero anche quando --target-dir e' un percorso dentro un container.",
    )
    args = parser.parse_args()
    # Sempre stile posix (/) per il valore scritto nel JSON: e' quello che
    # serve dentro un container Linux, e funziona comunque anche per
    # os.path.exists() su Windows (accetta '/' come separatore).
    models_dir = args.target_dir.rstrip("/\\").replace("\\", "/")

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
        filename = os.path.basename(old_path.replace("\\", "/"))
        new_path = f"{models_dir}/{filename}"
        if new_path != old_path:
            row["model_path"] = new_path
            fixed += 1
        if not os.path.exists(os.path.join(LOCAL_CHECK_DIR, filename)):
            missing_files.append(filename)

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Aggiornati {fixed}/{len(rows)} model_path -> {models_dir}")

    if missing_files:
        print(f"\nATTENZIONE: questi file NON esistono in {LOCAL_CHECK_DIR} "
              "(il path nel registry e' stato comunque scritto, ma il file .pkl manca davvero):")
        for name in sorted(set(missing_files)):
            print(f"  - {name}")
    else:
        print(f"Tutti i file .pkl referenziati esistono in {LOCAL_CHECK_DIR}.")


if __name__ == "__main__":
    main()
