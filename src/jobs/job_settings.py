from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from src.jobs.api_quota_state import get_quota_snapshot, is_quota_exhausted_today

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_QUOTA_PAUSE_LOCK = threading.Lock()

# Job registrati da `build_scheduler` (src/jobs/scheduler.py) - le chiavi
# DEVONO combaciare esattamente con i `job_id` passati a `_add_job`, cosi'
# il toggle da questa pagina Impostazioni disattiva/riattiva ESATTAMENTE il
# job schedulato corrispondente (nessun mapping duplicato altrove).
#
# `calls_api_sports` distingue i job che chiamano DAVVERO il provider
# esterno (consumano quota) da quelli che lavorano SOLO sui dati gia' a DB
# (`data_settlement`: riconciliazione locale; `ml_training`: training sui
# dati gia' scaricati) - usato da `sync_job_settings_with_quota` per capire
# QUALI job auto-mettere in pausa quando la quota e' esaurita: mettere in
# pausa anche settlement/training non avrebbe alcun senso (non consumano
# quota, quindi possono continuare a lavorare sui dati gia' presenti anche
# a quota esaurita) - stessa distinzione gia' applicata lato frontend per
# decidere quali bottoni disabilitare (Data Center/ML Lab).
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
        "calls_api_sports": True,
    },
    "data_sync_today": {
        "label": "Sync partite di oggi",
        "description": "Aggiorna stato/punteggi/quote delle fixture odierne (NS/live/final).",
        "default_enabled": True,
        "calls_api_sports": True,
    },
    "data_settlement": {
        "label": "Settlement partite concluse",
        "description": "Riconcilia le partite concluse per il settlement (dashboard/paper betting).",
        "default_enabled": True,
        "calls_api_sports": False,
    },
    "data_future_sync": {
        "label": "Sync calendario prossimo",
        "description": "Importa le fixture future (solo status NS) nella finestra di giorni configurata.",
        "default_enabled": True,
        "calls_api_sports": True,
    },
    "ml_training": {
        "label": "Retrain modelli ML",
        "description": "Riaddestra tutti i mercati con i dati piu' recenti (giornaliero).",
        "default_enabled": True,
        "calls_api_sports": False,
    },
    "data_sync_live": {
        "label": "Sync live (polling ogni pochi secondi)",
        "description": (
            "Polling molto frequente del dataset LIVE per il centro live in tempo reale. "
            "Consuma molte chiamate API-Sports: disattivato di default, attivalo solo se "
            "ti serve davvero il live."
        ),
        "default_enabled": False,
        "calls_api_sports": True,
    },
}

# Sottoinsieme di `JOB_DEFINITIONS` che consuma DAVVERO quota API-Sports -
# calcolato una volta sola qui (single source of truth: aggiungere un job
# futuro con `calls_api_sports=True` lo include automaticamente, senza
# toccare `sync_job_settings_with_quota`).
QUOTA_SENSITIVE_JOB_IDS: frozenset[str] = frozenset(
    job_id for job_id, definition in JOB_DEFINITIONS.items() if definition.get("calls_api_sports")
)

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
    """Vista arricchita (label/description/enabled/calls_api_sports) per il
    frontend - `calls_api_sports` permette alla pagina Impostazioni di
    segnalare quali job vengono coinvolti dall'auto-pausa per quota
    esaurita (vedi `sync_job_settings_with_quota`)."""
    settings = get_job_settings()
    rows = []
    for job_id, definition in JOB_DEFINITIONS.items():
        rows.append(
            {
                "job_id": job_id,
                "label": definition["label"],
                "description": definition["description"],
                "enabled": settings.get(job_id, definition["default_enabled"]),
                "calls_api_sports": bool(definition.get("calls_api_sports")),
            }
        )
    return rows



def _quota_pause_path() -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(project_root, "best_models", "quota_pause_state.json")


def _read_quota_pause_state() -> dict[str, Any]:
    path = _quota_pause_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_quota_pause_state(state: dict[str, Any]) -> None:
    path = _quota_pause_path()
    try:
        with _QUOTA_PAUSE_LOCK:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
    except OSError as exc:
        logger.warning("Impossibile persistere quota_pause_state.json: %s", exc)


def get_quota_pause_state() -> dict[str, Any]:
    """Stato corrente dell'auto-pausa per quota esaurita, per il frontend
    (pagina Impostazioni): `{"paused_date": "2026-09-06", "previous_settings":
    {...}}` se la pausa e' attiva, `{}` se non in pausa."""
    return _read_quota_pause_state()


def sync_job_settings_with_quota(quota_snapshot: Optional[dict[str, Any]] = None) -> bool:
    """Auto-pausa SOLO i job che chiamano davvero API-Sports
    (`QUOTA_SENSITIVE_JOB_IDS` - MAI `data_settlement`/`ml_training`, che
    lavorano solo su dati gia' a DB e possono continuare tranquillamente)
    quando la quota giornaliera risulta esaurita da un check AUTORITATIVO
    di OGGI (`is_quota_exhausted_today`, MAI dalla sola stima passiva -
    vedi `api_quota_state.py`), e li riattiva automaticamente al reset del
    giorno successivo (o non appena non e' piu' segnalata esaurita per
    oggi) ripristinando ESATTAMENTE lo stato enabled/disabled che avevano
    PRIMA della pausa (es. `data_sync_live`, di default disattivato, resta
    disattivato se lo era gia').

    A differenza di una singola "botta" alla prima rilevazione, la
    disattivazione viene RI-APPLICATA ad ogni chiamata finche' la pausa e'
    attiva: un eventuale toggle manuale fatto durante la pausa (es. un
    utente che riaccende "Sync partite di oggi" sperando funzioni) viene
    corretto di nuovo al controllo successivo (scheduler: prima di ogni
    esecuzione - Impostazioni: al prossimo GET/POST) invece di restare
    "acceso" fino al giorno dopo, inutilmente destinato a fallire di nuovo
    contro un provider che risponderebbe comunque "quota esaurita".

    Chiamata ad OGNI tick dello scheduler (vedi
    `src/jobs/scheduler.py::_run_if_enabled`) e ad ogni lettura/scrittura
    delle Impostazioni (`src/api/main.py`) - mai un'eccezione propagata: un
    fallimento qui non deve MAI bloccare l'esecuzione normale dei job.
    Ritorna True se una pausa e' (rimasta) attiva."""
    try:
        snapshot = quota_snapshot if quota_snapshot is not None else get_quota_snapshot()
        exhausted_today = is_quota_exhausted_today(snapshot)
        pause_state = _read_quota_pause_state()
        paused_date = pause_state.get("paused_date")
        today_iso = datetime.now(timezone.utc).date().isoformat()

        if exhausted_today:
            current = get_job_settings()
            if paused_date != today_iso:
                # Prima rilevazione di oggi: salva la baseline (SOLO i job
                # quota-sensitive) PRIMA di disattivarli, cosi' da poterla
                # ripristinare esattamente al reset della quota.
                previous = {job_id: current[job_id] for job_id in QUOTA_SENSITIVE_JOB_IDS}
                _write_quota_pause_state({"paused_date": today_iso, "previous_settings": previous})
                logger.warning(
                    "Quota API-Sports esaurita (100%%): job che chiamano il provider esterno (%s) "
                    "disattivati fino al reset di domani (UTC). data_settlement/ml_training restano attivi.",
                    ", ".join(sorted(QUOTA_SENSITIVE_JOB_IDS)),
                )
            # Ri-applicata ad OGNI check (prima rilevazione o successiva):
            # solo se serve davvero (evita scritture su disco a vuoto
            # quando e' gia' tutto disattivato).
            needs_disable = any(current.get(job_id, True) for job_id in QUOTA_SENSITIVE_JOB_IDS)
            if needs_disable:
                update_job_settings({job_id: False for job_id in QUOTA_SENSITIVE_JOB_IDS})
            return True

        if paused_date:
            previous = pause_state.get("previous_settings") or {}
            if previous:
                update_job_settings(previous)
            _write_quota_pause_state({})
            logger.info("Quota API-Sports non piu' segnalata esaurita: job ripristinati allo stato pre-pausa.")
        return False
    except Exception as exc:  # pragma: no cover - mai bloccare lo scheduler/le Impostazioni
        logger.warning("sync_job_settings_with_quota fallita (ignorata): %s", exc)
        return False


