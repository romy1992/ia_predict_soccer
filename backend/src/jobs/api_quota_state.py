from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


def _state_path() -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(project_root, "best_models", "api_quota_state.json")


def _read_state() -> dict[str, Any]:
    path = _state_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_state(partial_update: dict[str, Any]) -> None:
    """Merge (mai overwrite totale) di `partial_update` nello stato gia'
    persistito su disco, poi scrittura atomica (file temp + `os.replace`,
    per evitare letture di JSON troncato in caso di scritture concorrenti).

    Scritto su un file condiviso tra i container `api` e `scheduler`
    (entrambi montano `./best_models`, vedi docker-compose.yml) perche' le
    chiamate reali avvengono in PROCESSI SEPARATI: senza persistenza su
    disco, l'endpoint `/settings/quota` (letto dal container `api`) non
    vedrebbe mai gli aggiornamenti fatti dalle chiamate del `scheduler`.
    Il merge (invece di un overwrite) e' necessario perche' due fonti
    diverse scrivono sullo STESSO file in momenti diversi: lo snapshot
    passivo `daily_remaining/minute_*` (osservato su OGNI chiamata dati,
    vedi `record_quota_snapshot`) e lo snapshot autoritativo `status_*`
    (solo quando l'utente clicca "Aggiorna", vedi `record_status_snapshot`)
    - un overwrite totale farebbe perdere l'uno scrivendo l'altro."""
    path = _state_path()
    try:
        with _LOCK:
            state = _read_state()
            state.update(partial_update)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
    except OSError as exc:
        logger.warning("Impossibile persistere api_quota_state.json: %s", exc)


def record_quota_snapshot(
    *,
    daily_remaining: int,
    minute_remaining: int,
    minute_limit: int,
    daily_limit: Optional[int] = None,
) -> None:
    """Persiste l'ultimo snapshot di quota osservato PASSIVAMENTE negli
    header HTTP di API-Sports (`x-ratelimit-*`, vedi
    `ApiSportsProvider._handle_quota_headers`) durante una chiamata dati
    qualsiasi (fixtures/odds/statistics/...).

    ATTENZIONE - `daily_remaining` NON e' affidabile come segnale di
    "quota esaurita o no" (diagnosticato 2026-09-05: un `/status` con
    HTTP 200 e body `errors: {"requests": "...limite giornaliero
    raggiunto..."}` e' arrivato ASSIEME a `x-ratelimit-requests-remaining:
    7499` sulla STESSA risposta - il provider mantiene evidentemente un
    contatore/bucket per l'header diverso da quello che poi enforcea
    davvero il blocco). Va quindi trattato SOLO come stima di replacement
    quando non esiste ancora nessun check live (`status_*`, vedi
    `record_status_snapshot`) - l'unica fonte davvero autoritativa e' il
    campo `errors` del body (qui o su `/status`).
    `daily_limit`, se presente nell'header `x-ratelimit-requests-limit`,
    viene comunque salvato: e' il valore REALE del piano, piu' affidabile
    del fallback hardcoded `API_SPORTS_DAILY_LIMIT`."""
    update: dict[str, Any] = {
        "daily_remaining": daily_remaining,
        "minute_remaining": minute_remaining,
        "minute_limit": minute_limit,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if daily_limit:
        update["daily_limit"] = daily_limit
    _write_state(update)


def record_status_snapshot(
    *,
    requests_current: Optional[int],
    requests_limit_day: Optional[int],
    plan: Optional[str] = None,
    active: Optional[bool] = None,
    error_message: Optional[str] = None,
) -> None:
    """Persiste il risultato di un'interrogazione REALE e autoritativa:
    o l'endpoint ufficiale `GET /status` di API-Sports (vedi
    `ApiSportsProvider.get_status`), o la rilevazione di un campo `errors`
    (quota esaurita) su QUALSIASI risposta - vedi
    `ApiSportsProvider._mark_quota_exhausted`. A differenza di
    `record_quota_snapshot` (dedotto passivamente dagli header, dimostrato
    inaffidabile per capire se la quota e' esaurita), questo e' il valore
    ESATTO mostrato sulla dashboard account di api-sports.io (oppure,
    quando `error_message` e' valorizzato, la certezza assoluta che la
    quota e' al 100% - il messaggio testuale del provider stesso).
    Usata su richiesta esplicita (bottone "Aggiorna") o automaticamente
    non appena una QUALSIASI chiamata dati incontra `errors` nel body."""
    _write_state(
        {
            "status_requests_current": requests_current,
            "status_requests_limit_day": requests_limit_day,
            "status_plan": plan,
            "status_active": active,
            "status_error_message": error_message,
            "status_updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def get_quota_snapshot() -> Optional[dict[str, Any]]:
    """Ultimo snapshot noto (merge di eventuale check live `status_*` e/o
    stima passiva `daily_remaining`), oppure `None` se non e' ancora stata
    fatta nessuna chiamata API-Sports da quando esiste questo tracking."""
    state = _read_state()
    return state or None


def is_quota_exhausted_today(snapshot: Optional[dict[str, Any]] = None) -> bool:
    """True SOLO se l'ultimo check AUTORITATIVO (`status_error_message`,
    popolato esclusivamente da un vero errore di quota GIORNALIERA - vedi
    `record_status_snapshot`/`ApiSportsProvider._mark_quota_exhausted`, MAI
    dalla sola stima passiva `daily_remaining` dimostrata inaffidabile da
    sola) e' riferito alla giornata UTC CORRENTE: la quota si resetta a
    mezzanotte UTC, quindi uno snapshot "esaurita" di ieri non e' piu' un
    segnale valido oggi (senza questo controllo di data, un `error_message`
    rimasto in cache mostrerebbe un falso allarme anche dopo il reset).

    Usata per l'auto-pausa dei job schedulati quando la quota e' al 100%
    (vedi `src/jobs/job_settings.py::sync_job_settings_with_quota`) e, lato
    frontend, per disabilitare i bottoni che richiamano il provider esterno
    (vedi `ApiQuotaResponse.daily_used_percentage`/`src/api/main.py`)."""
    state = snapshot if snapshot is not None else get_quota_snapshot()
    if not state:
        return False
    error_message = state.get("status_error_message")
    updated_at = state.get("status_updated_at")
    if not error_message or not updated_at:
        return False
    try:
        dt = datetime.fromisoformat(updated_at)
    except ValueError:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).date() == datetime.now(timezone.utc).date()
