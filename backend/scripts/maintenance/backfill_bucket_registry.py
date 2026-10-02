"""Backfill UNA TANTUM: migra il vecchio registry locale (`best_models/
registry/index.jsonl` + `promotion_history.jsonl`, letti dal Volume ancora
montato) e i relativi file `.pkl` al nuovo schema a oggetti sul Bucket
(`ModelRegistry` post-migrazione: un oggetto per run/evento invece di due
file ad append - vedi `src/service_ia/training/model_registry.py`).

Va eseguito UNA VOLTA, da un container che ha ANCORA sia il Volume montato
(sorgente, dati legacy) sia il bucket collegato (destinazione, codice gia'
deployato) - cioe' il servizio `api`, PRIMA che il Volume venga rimosso
(Fase 5 della migrazione). Sola lettura sul Volume, additivo sul bucket
(idempotente: una chiave gia' presente viene saltata, rieseguibile senza
duplicare nulla). Non tocca `archivio/` (migrato a parte in precedenza).

I calibratori salvati come file SEPARATO (`extra.calibration.
calibrator_path`) non vengono migrati: sono una copia ridondante dello
stesso estimator gia' calibrato contenuto nel file champion principale
(mai risolti/caricati a parte in fase di serving - vedi
`train_multi_market.py::train_market`), quindi non servono per il
funzionamento, solo come riferimento informativo in `extra`."""

from __future__ import annotations

import json
import logging
import os

from src.service_ia.training.model_paths import relative_to_best_models, resolve_model_path
from src.storage import bucket_store

logging.basicConfig(level=logging.INFO)

LOCAL_ROOT = "/app/best_models"
INDEX_PATH = os.path.join(LOCAL_ROOT, "registry", "index.jsonl")
PROMOTIONS_PATH = os.path.join(LOCAL_ROOT, "registry", "promotion_history.jsonl")


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    if not os.path.exists(path):
        logging.warning("File non trovato, salto: %s", path)
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _to_bucket_key(path: str | None) -> str | None:
    relative = relative_to_best_models(path) if path else None
    if relative is None:
        return None
    return f"best_models/{relative}"


def migrate_runs() -> None:
    rows = _read_jsonl(INDEX_PATH)
    migrated = 0
    skipped_no_file = 0
    skipped_duplicate = 0

    for row in rows:
        run_id = row.get("run_id")
        if not run_id:
            continue
        run_key = f"best_models/registry/runs/{run_id}.json"
        if bucket_store.exists(run_key):
            skipped_duplicate += 1
            continue

        old_model_path = row.get("model_path")
        local_resolved = resolve_model_path(old_model_path)
        new_model_key = _to_bucket_key(old_model_path)

        payload = dict(row)
        if local_resolved and new_model_key:
            with open(local_resolved, "rb") as f:
                data = f.read()
            bucket_store.put_bytes(new_model_key, data)
            payload["model_path"] = new_model_key
            payload["metadata_path"] = run_key
        else:
            skipped_no_file += 1
            logging.warning(
                "Run %s: file modello non trovato localmente (%s) - migro solo i metadati",
                run_id,
                old_model_path,
            )

        bucket_store.put_json(run_key, payload)
        migrated += 1

    logging.info(
        "Run migrati: %d (file mancante: %d, gia' presenti sul bucket: %d)",
        migrated,
        skipped_no_file,
        skipped_duplicate,
    )


def migrate_promotions() -> None:
    events = _read_jsonl(PROMOTIONS_PATH)
    migrated = 0
    skipped = 0

    for event in events:
        event_id = event.get("event_id")
        if not event_id:
            continue
        key = f"best_models/registry/promotions/{event_id}.json"
        if bucket_store.exists(key):
            skipped += 1
            continue
        bucket_store.put_json(key, event)
        migrated += 1

    logging.info("Eventi di promozione migrati: %d (gia' presenti sul bucket: %d)", migrated, skipped)


if __name__ == "__main__":
    migrate_runs()
    migrate_promotions()
