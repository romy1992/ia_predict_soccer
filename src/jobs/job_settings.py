from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

# Job registrati da `build_scheduler` (src/jobs/scheduler.py) - le chiavi
# DEVONO combaciare esattamente con i `job_id` passati a `_add_job`, cosi'
# il toggle da questa pagina Impostazioni disattiva/riattiva ESATTAMENTE il
# job schedulato corrispondente (nessun mapping duplicato altrove).
#
# `data_sync_live` di default DISABILITATO (richiesto esplicitamente
# 2026-09-05): e' il job piu' "costoso" in termini di chiamate API-Sports
# (polling ogni `live_sync_interval_seconds`, default 90s, quindi centinaia
# di chiamate/giorno) e non tutti gli utenti hanno bisogno del centro live
# in tempo reale - meglio farlo attivare esplicitamente da chi lo vuole.
JOB_DEFINITIONS: dict[str, dict[str, Any]] = {
    "data_daily_refresh": {
        "label": "Aggiorna tutto (ieri + prossimi giorni)",
        "description": (
            "Job schedulato equivalente al bottone 'Aggiorna tutto' della sidebar: in un colpo solo "
            "importa risultati/statistiche/quote delle partite di IERI per tutti i campionati censiti "
            "e sincronizza il calendario prossimo (aggiornando anche le partite gia' presenti a DB). "
            "Si sovrappone in parte a 'Sync calendario prossimo': se entrambi attivi il calendario "
            "prossimo viene aggiornato due volte al giorno (doppio consumo quota API-Sports)."
        ),
        "default_enabled": True,
    },
    "data_sync_today": {
        "label": "Sync partite di oggi",
        "description": "Aggiorna stato/punteggi/quote delle fixture odierne (NS/live/final).",
        "default_enabled": True,
    },
    "data_settlement": {
        "label": "Settlement partite concluse",
        "description": "Riconcilia le partite concluse per il settlement (dashboard/paper betting).",
        "default_enabled": True,
    },
    "data_future_sync": {
        "label": "Sync calendario prossimo",
        "description": "Importa le fixture future (solo status NS) nella finestra di giorni configurata.",
        "default_enabled": True,
    },
    "ml_training": {
        "label": "Retrain modelli ML",
        "description": "Riaddestra tutti i mercati con i dati piu' recenti (giornaliero).",
        "default_enabled": True,
    },
    "data_sync_live": {
        "label": "Sync live (polling ogni pochi secondi)",
        "description": (
            "Polling molto frequente del dataset LIVE per il centro live in tempo reale. "
            "Consuma molte chiamate API-Sports: disattivato di default, attivalo solo se "
            "ti serve davvero il live."
        ),
        "default_enabled": False,
    },
}

DEFAULT_JOB_SETTINGS: dict[str, bool] = {
    job_id: definition["default_enabled"] for job_id, definition in JOB_DEFINITIONS.items()
}


def _settings_path() -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(project_root, "best_models", "job_settings.json")


def get_job_settings() -> dict[str, bool]:
    """Legge i flag enabled/disabled correnti, sempre con TUTTE le chiavi
    note (anche se il file su disco e' vuoto/parziale/mancante): un job
    nuovo aggiunto in futuro a `JOB_DEFINITIONS` risultera' comunque
    presente con il suo default, mai un KeyError lato scheduler/frontend."""
    path = _settings_path()
    stored: dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    stored = loaded
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("job_settings.json illeggibile (%s): uso i default", exc)

    merged = dict(DEFAULT_JOB_SETTINGS)
    for job_id in JOB_DEFINITIONS:
        if job_id in stored:
            merged[job_id] = bool(stored[job_id])
    return merged


def update_job_settings(updates: dict[str, bool]) -> dict[str, bool]:
    """Aggiorna (merge parziale) i flag e persiste su disco (scrittura
    atomica via file temporaneo + `os.replace`, cosi' un container che
    legge nel mezzo di uno scritto da un altro non trova mai JSON tronco).
    Chiavi non riconosciute sollevano `ValueError` esplicito (mai un job
    fantasma scritto su file per un typo del chiamante)."""
    unknown = [job_id for job_id in updates if job_id not in JOB_DEFINITIONS]
    if unknown:
        raise ValueError(f"Job id non riconosciuti: {unknown}")

    with _LOCK:
        current = get_job_settings()
        current.update({job_id: bool(value) for job_id, value in updates.items()})

        path = _settings_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    return current


def is_job_enabled(job_id: str) -> bool:
    """Usato dallo scheduler PRIMA di ogni esecuzione (non solo alla
    registrazione): un toggle da questa pagina Impostazioni ha quindi
    effetto immediato, senza richiedere il restart del container
    `scheduler`."""
    return bool(get_job_settings().get(job_id, DEFAULT_JOB_SETTINGS.get(job_id, True)))


def list_job_definitions() -> list[dict[str, Any]]:
    """Vista arricchita (label/description/enabled) per il frontend."""
    settings = get_job_settings()
    rows = []
    for job_id, definition in JOB_DEFINITIONS.items():
        rows.append(
            {
                "job_id": job_id,
                "label": definition["label"],
                "description": definition["description"],
                "enabled": settings.get(job_id, definition["default_enabled"]),
            }
        )
    return rows


