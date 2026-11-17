from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from src.jobs.api_quota_state import is_quota_exhausted_today, read_quota_snapshot_strict
from src.service_ia.config.app_config import AppConfig, load_app_config
from src.storage import bucket_store

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_QUOTA_PAUSE_LOCK = threading.Lock()
# Serializza, DENTRO il processo, l'intero ciclo "leggi quota + leggi pausa +
# decidi + scrivi" di `sync_job_settings_with_quota` (e l'aggiornamento della
# baseline fatto da `remember_operator_toggles_during_pause`). Lo scheduler
# valuta i job su piu' thread nello stesso tick e l'API serve richieste in
# parallelo: senza, due thread che rilevano insieme la quota esaurita possono
# salvare come baseline lo stato GIA' disattivato dall'altro (e al reset non
# si riaccende piu' nulla). Rientrante: non e' mai tenuto insieme a `_LOCK` o
# `_QUOTA_PAUSE_LOCK` in ordine inverso. Fra processi diversi (`api` e
# `scheduler`) non esiste un lock: resta una finestra di pochi istanti.
_QUOTA_SYNC_LOCK = threading.RLock()
_SCHEDULE_LOCK = threading.Lock()

# Job registrati da `build_scheduler` (src/jobs/scheduler.py) - le chiavi
# DEVONO combaciare esattamente con i `job_id` passati a `_add_job`, cosi'
# il toggle da questa pagina Impostazioni disattiva/riattiva ESATTAMENTE il
# job schedulato corrispondente (nessun mapping duplicato altrove).
#
# `calls_api_sports` distingue i job che NON possono lavorare senza il
# provider esterno da quelli che possono continuare sui dati gia' a DB
# (`ml_training`: training sui dati gia' scaricati) - usato da
# `sync_job_settings_with_quota` per capire QUALI job auto-mettere in pausa
# quando la quota e' esaurita - stessa distinzione gia' applicata lato
# frontend per decidere quali bottoni disabilitare (Data Center/ML Lab).
#
# `data_settlement` e' un caso a parte e resta `False` di proposito: in
# condizioni normali INIZIA con un import da API-Sports delle partite
# concluse nella finestra recente (`SettlementService.run_settlement` ->
# `download_import_matches`, quindi consuma quota), ma la riconciliazione
# vera e propria (completezza, ledger, schedine) lavora sul DB. Metterlo in
# pausa a quota esaurita fermerebbe anche quella parte: invece resta attivo
# e, quando `is_quota_exhausted_today()` e' vero, salta il solo import dal
# provider (vedi `SettlementService.run_settlement`).
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
        "schedule_kind": "daily",
        "job_history_type": "daily_refresh",
        "cfg_fields": {"hour": "daily_refresh_hour", "minute": "daily_refresh_minute"},
        "run_all_priority": 10,
        "run_all_included": True,
    },
    "data_sync_today": {
        "label": "Sync partite di oggi",
        "description": "Aggiorna stato/punteggi/quote delle fixture odierne (NS/live/final).",
        "default_enabled": True,
        "calls_api_sports": True,
        "schedule_kind": "interval_minutes",
        "job_history_type": "today_update",
        "cfg_fields": {"interval_minutes": "data_sync_interval_minutes"},
        "run_all_priority": 10,
        "run_all_included": True,
    },
    "data_settlement": {
        "label": "Settlement partite concluse",
        "description": (
            "Riconcilia le partite concluse per il settlement (dashboard/paper betting). "
            "Prima aggiorna da API-Sports i risultati degli ultimi giorni; a quota esaurita "
            "salta l'aggiornamento e lavora solo sui dati gia' presenti."
        ),
        "default_enabled": True,
        "calls_api_sports": False,
        # Additivo: il job consuma quota API-Sports solo nel passo iniziale di
        # import (saltato a quota esaurita); non e' in pausa automatica.
        "partially_calls_api_sports": True,
        "schedule_kind": "interval_minutes",
        "job_history_type": "settlement",
        "cfg_fields": {"interval_minutes": "settlement_interval_minutes"},
        "run_all_priority": 20,
        "run_all_included": True,
    },
    "data_future_sync": {
        "label": "Sync calendario prossimo",
        "description": "Importa le fixture future (solo status NS) nella finestra di giorni configurata.",
        "default_enabled": True,
        "calls_api_sports": True,
        "schedule_kind": "daily",
        "job_history_type": "future_sync",
        "cfg_fields": {"hour": "future_sync_hour", "minute": "future_sync_minute"},
        "run_all_priority": 10,
        "run_all_included": True,
    },
    "ml_training": {
        "label": "Retrain modelli ML",
        "description": "Riaddestra tutti i mercati con i dati piu' recenti (giornaliero).",
        "default_enabled": True,
        "calls_api_sports": False,
        "schedule_kind": "daily",
        "job_history_type": "retrain",
        "cfg_fields": {"hour": "training_hour", "minute": "training_minute"},
        # Escluso da "Aggiorna tutto manualmente" (2026-09-26): e' l'unico
        # job davvero lento (GridSearchCV multi-mercato, minuti), farebbe
        # sembrare il giro bloccato. Resta eseguibile singolarmente con
        # "Esegui ora" come sempre.
        "run_all_priority": 999,
        "run_all_included": False,
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
        "schedule_kind": "interval_seconds",
        "job_history_type": "live_sync",
        "cfg_fields": {"interval_seconds": "live_sync_interval_seconds"},
        # Escluso (2026-09-26): e' un polling continuo pensato per girare
        # schedulato, non un "refresh singolo" - fuori posto in una
        # sequenza "aggiorna tutto una volta".
        "run_all_priority": 999,
        "run_all_included": False,
    },
    "data_quality_report": {
        "label": "Aggiorna report Data Quality",
        "description": (
            "Ricalcola periodicamente il report di qualita' dati (coverage odds, anomalie, "
            "duplicati, distribuzione) e lo registra nello storico job - stessa funzione "
            "del bottone 'Aggiorna report' della pagina Data Quality. Lavora solo su dati "
            "gia' a DB: non chiama API-Sports."
        ),
        "default_enabled": True,
        "calls_api_sports": False,
        "schedule_kind": "interval_minutes",
        "job_history_type": "data_quality_report",
        "cfg_fields": {"interval_minutes": "data_quality_interval_minutes"},
        "run_all_priority": 70,
        "run_all_included": True,
    },
    "prediction_snapshot_refresh": {
        "label": "Aggiorna predizioni salvate",
        "description": (
            "Ricalcola in background le predizioni delle partite non ancora disputate la cui "
            "fingerprint di feature (quote/statistiche) e' cambiata dall'ultimo giro, E chiude "
            "automaticamente il buco di copertura per le partite APPENA concluse (ultimi giorni) "
            "che non hanno ancora nessuna riga salvata - popolando la banca dati "
            "'match_prediction_snapshot' PRIMA che qualcuno apra la Dashboard, cosi' la vista "
            "storica legge sempre una riga gia' calcolata invece di rifare l'inferenza ML ad "
            "ogni richiesta. Lavora solo su dati gia' a DB: non chiama API-Sports."
        ),
        "default_enabled": True,
        "calls_api_sports": False,
        "schedule_kind": "interval_minutes",
        "job_history_type": "prediction_snapshot_refresh",
        "cfg_fields": {"interval_minutes": "prediction_snapshot_interval_minutes"},
        "run_all_priority": 50,
        "run_all_included": True,
    },
    "team_rating_refresh": {
        "label": "Aggiorna rating squadre",
        "description": (
            "Ricalcola in background lo stato corrente (attacco/difesa/forma) di OGNI squadra "
            "(EXP-01), popolando la tabella 'team_rating' PRIMA che qualcuno apra l'Oracle Match "
            "Detail, cosi' quella vista legge sempre una riga gia' calcolata invece di rifare "
            "query+ricalcolo sull'intero storico squadra ad ogni apertura. Lavora solo su dati "
            "gia' a DB: non chiama API-Sports."
        ),
        "default_enabled": True,
        "calls_api_sports": False,
        "schedule_kind": "interval_minutes",
        "job_history_type": "team_rating_refresh",
        "cfg_fields": {"interval_minutes": "team_rating_refresh_interval_minutes"},
        "run_all_priority": 30,
        "run_all_included": True,
    },
    "model_consensus_refresh": {
        "label": "Aggiorna Model Consensus",
        "description": (
            "Ricalcola in background il Model Consensus (Direct Expert + Market/Odds Expert + "
            "eventuale meta-model) per le partite non ancora disputate nella finestra configurata "
            "E chiude il buco di copertura per le partite appena concluse, popolando la tabella "
            "'model_consensus_snapshot' PRIMA che qualcuno apra l'Oracle Match Detail, cosi' quella "
            "vista legge sempre una riga gia' calcolata invece di ricaricare il modello da disco per "
            "OGNI mercato ad ogni apertura. Lavora solo su dati gia' a DB: non chiama API-Sports."
        ),
        "default_enabled": True,
        "calls_api_sports": False,
        "schedule_kind": "interval_minutes",
        "job_history_type": "model_consensus_refresh",
        "cfg_fields": {"interval_minutes": "model_consensus_refresh_interval_minutes"},
        "run_all_priority": 40,
        "run_all_included": True,
    },
    "official_prediction_capture": {
        "label": "Cattura PLAY ufficiali",
        "description": (
            "Registra nel Prediction Ledger le sole decisioni PLAY con modello production "
            "e quota snapshot valida, prima del kickoff. È idempotente e non dipende dalla Dashboard."
        ),
        "default_enabled": True,
        "calls_api_sports": False,
        "schedule_kind": "interval_minutes",
        "job_history_type": "official_prediction_capture",
        "cfg_fields": {"interval_minutes": "official_capture_interval_minutes"},
        "run_all_priority": 60,
        "run_all_included": True,
    },
    "scorer_predictions": {
        "label": "Predizioni marcatori",
        "description": (
            "Per le partite delle prime 5 leghe in partenza a breve scarica la formazione ufficiale "
            "(poche chiamate API-Sports) e calcola la probabilita' che ogni titolare segni, salvando "
            "una revisione append-only prima del kickoff. Usa il modello in stage 'production' del "
            "mercato 'player_anytime_goalscorer': se nessuno e' stato promosso non predice."
        ),
        "default_enabled": True,
        "calls_api_sports": True,
        "schedule_kind": "interval_minutes",
        "job_history_type": "scorer_predictions",
        "cfg_fields": {"interval_minutes": "scorer_predictions_interval_minutes"},
        "run_all_priority": 85,
        "run_all_included": True,
    },
    "scorer_training": {
        "label": "Retrain modello marcatori",
        "description": (
            "Riaddestra il modello 'il giocatore segna' sui dati a DB (prime 5 leghe): confronta logistica, "
            "gradient boosting, random forest e voting su un holdout temporale (log loss, Brier, ECE, AUC), "
            "registra la migliore come CANDIDATE e la promuove a production solo se batte la baseline, e' "
            "calibrata e supera gate metriche + confronto con l'attuale production (mai forzato, tutto "
            "nell'audit). Lavora solo su dati gia' a DB."
        ),
        "default_enabled": True,
        "calls_api_sports": False,
        "schedule_kind": "daily",
        "job_history_type": "scorer_training",
        "cfg_fields": {"hour": "scorer_training_hour", "minute": "scorer_training_minute"},
        # Prima delle predizioni (85): il retrain puo' promuovere un modello nuovo che le predizioni usano.
        # Dura 2-3 minuti, molto meno del retrain dei mercati di partita (che resta escluso).
        "run_all_priority": 84,
        "run_all_included": True,
    },
    "player_cards_predictions": {
        "label": "Predizioni cartellini giocatori",
        "description": (
            "Per le partite in partenza a breve (stesse competizioni dei marcatori) usa la formazione "
            "ufficiale (gia' scaricata dal job marcatori, altrimenti la scarica: poche chiamate API-Sports) "
            "e calcola la probabilita' che ogni titolare prenda un giallo, salvando una revisione "
            "append-only prima del kickoff. Usa il modello in stage 'production' del mercato "
            "'player_yellow_card': se nessuno e' stato promosso non predice."
        ),
        "default_enabled": True,
        "calls_api_sports": True,
        "schedule_kind": "interval_minutes",
        "job_history_type": "player_cards_predictions",
        "cfg_fields": {"interval_minutes": "player_cards_predictions_interval_minutes"},
        "run_all_priority": 87,
        "run_all_included": True,
    },
    "player_cards_training": {
        "label": "Retrain modello cartellini giocatori",
        "description": (
            "Riaddestra il modello 'il giocatore prende un giallo' sui dati a DB: confronta logistica, "
            "gradient boosting, random forest e voting su un holdout temporale (solo titolari), registra la "
            "migliore come CANDIDATE e la promuove a production solo se batte la baseline, e' calibrata e "
            "supera gate metriche + confronto con l'attuale production (mai forzato, tutto nell'audit). "
            "Lavora solo su dati gia' a DB."
        ),
        "default_enabled": True,
        "calls_api_sports": False,
        "schedule_kind": "daily",
        "job_history_type": "player_cards_training",
        "cfg_fields": {"hour": "player_cards_training_hour", "minute": "player_cards_training_minute"},
        "run_all_priority": 86,
        "run_all_included": True,
    },
    "fotmob_predicted_lineups": {
        "label": "Formazioni previste FotMob",
        "description": (
            "Una volta al giorno recupera da FotMob le probabili formazioni delle partite delle prossime 72 ore "
            "(prime 5 leghe e coppe europee gia' a DB), le valida e le salva come snapshot. Solo se i controlli "
            "passano calcola una previsione iniziale di marcatori e ammoniti con il modello gia' in production. "
            "Non sostituisce le formazioni ufficiali API-Sports e non cambia i pesi del modello. "
            "Richiede RAPIDAPI_KEY (RapidAPI, non la quota API-Sports)."
        ),
        "default_enabled": True,
        "calls_api_sports": False,
        "schedule_kind": "daily",
        "job_history_type": "fotmob_predicted_lineups",
        "cfg_fields": {"hour": "fotmob_lineup_hour", "minute": "fotmob_lineup_minute"},
        "run_all_priority": 999,
        "run_all_included": False,
    },
}

# Limiti di validazione per ciascun `schedule_kind` (Impostazioni, editor
# orario/intervallo): valori troppo piccoli per gli intervalli in secondi/
# minuti rischierebbero di bombardare API-Sports (rate limit) o il DB -
# stesso principio di cautela gia' applicato altrove nel progetto (pool
# limitato per le chiamate parallele, TTL cache minimi, ecc.).
_SCHEDULE_FIELD_BOUNDS: dict[str, tuple[int, int]] = {
    "hour": (0, 23),
    "minute": (0, 59),
    "interval_minutes": (5, 1440),
    "interval_seconds": (30, 3600),
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



_SETTINGS_KEY = "best_models/job_settings.json"


def get_job_settings() -> dict[str, bool]:
    """Legge i flag enabled/disabled correnti, sempre con TUTTE le chiavi
    note (anche se l'oggetto sul bucket e' vuoto/parziale/mancante): un job
    nuovo aggiunto in futuro a `JOB_DEFINITIONS` risultera' comunque
    presente con il suo default, mai un KeyError lato scheduler/frontend.
    Chiamata PRIMA di ogni esecuzione schedulata (`scheduler._run_if_due`): un
    fallimento di rete verso il bucket non deve mai bloccare il tick dello
    scheduler, uso i default come fallback."""
    try:
        return _read_job_settings_strict()
    except Exception as exc:
        logger.warning("job_settings.json illeggibile dal bucket (%s): uso i default", exc)
        return dict(DEFAULT_JOB_SETTINGS)


def _read_job_settings_strict() -> dict[str, bool]:
    """Come `get_job_settings`, ma un errore di LETTURA del bucket viene
    propagato. Usata da chi poi RISCRIVE l'oggetto (`update_job_settings`,
    `sync_job_settings_with_quota`): ripiegare sui default e salvarli
    cancellerebbe i toggle dell'operatore per un errore transitorio di rete.
    Oggetto assente o non valido: default, come sempre."""
    stored = bucket_store.get_json(_SETTINGS_KEY, default={})
    if not isinstance(stored, dict):
        stored = {}

    merged = dict(DEFAULT_JOB_SETTINGS)
    for job_id in JOB_DEFINITIONS:
        if job_id in stored:
            merged[job_id] = bool(stored[job_id])
    return merged


def update_job_settings(updates: dict[str, bool]) -> dict[str, bool]:
    """Aggiorna (merge parziale) i flag e persiste sul bucket (PUT
    dell'intero oggetto - stesso principio dell'atomicita' che la vecchia
    scrittura via file temporaneo + `os.replace` garantiva, qui per
    costruzione: S3 non ha scritture parziali). Chiavi non riconosciute
    sollevano `ValueError` esplicito (mai un job fantasma scritto per un
    typo del chiamante).

    Se lo stato corrente NON e' leggibile (errore del bucket) l'eccezione
    viene propagata e non si scrive nulla: prima si ripiegava sui default e
    li si salvava, azzerando in silenzio tutti gli altri toggle."""
    unknown = [job_id for job_id in updates if job_id not in JOB_DEFINITIONS]
    if unknown:
        raise ValueError(f"Job id non riconosciuti: {unknown}")

    with _LOCK:
        current = _read_job_settings_strict()
        current.update({job_id: bool(value) for job_id, value in updates.items()})
        bucket_store.put_json(_SETTINGS_KEY, current)

    return current


def is_job_enabled(job_id: str) -> bool:
    """Usato dallo scheduler PRIMA di ogni esecuzione (non solo alla
    registrazione): un toggle da questa pagina Impostazioni ha quindi
    effetto immediato, senza richiedere il restart del container
    `scheduler`."""
    return bool(get_job_settings().get(job_id, DEFAULT_JOB_SETTINGS.get(job_id, True)))


def list_job_definitions(cfg: Optional[AppConfig] = None) -> list[dict[str, Any]]:
    """Vista arricchita (label/description/enabled/calls_api_sports/orario)
    per il frontend - `calls_api_sports` permette alla pagina Impostazioni
    di segnalare quali job vengono coinvolti dall'auto-pausa per quota
    esaurita (vedi `sync_job_settings_with_quota`). `schedule` e' l'orario
    EFFETTIVO (override salvato se presente, altrimenti quello da
    `AppConfig`/variabili d'ambiente - vedi `resolve_job_schedule`)."""
    settings = get_job_settings()
    cfg = cfg or load_app_config()
    # Job in corso adesso (anche lanciati da fuori, es. "Aggiorna tutto" dell'app), per tipo di cronologia.
    try:
        from src.jobs.job_history import JobHistory

        active = JobHistory().active_runs()
    except Exception:  # noqa: BLE001 - la lista job non deve dipendere dalla cronologia
        active = {}
    rows = []
    for job_id, definition in JOB_DEFINITIONS.items():
        run = active.get(definition.get("job_history_type") or "")
        rows.append(
            {
                "job_id": job_id,
                "label": definition["label"],
                "description": definition["description"],
                "enabled": settings.get(job_id, definition["default_enabled"]),
                "calls_api_sports": bool(definition.get("calls_api_sports")),
                "partially_calls_api_sports": bool(definition.get("partially_calls_api_sports")),
                "schedule_kind": definition["schedule_kind"],
                "schedule": resolve_job_schedule(job_id, cfg=cfg),
                "schedule_is_default": job_id not in get_job_schedule_overrides(),
                # Usati da "Aggiorna tutto manualmente" (Impostazioni, 2026-09-26)
                # per decidere ORDINE e SOTTOINSIEME dei job da lanciare in
                # sequenza - single source of truth qui, mai duplicati lato
                # frontend (stesso principio di `calls_api_sports`).
                "run_all_priority": definition.get("run_all_priority", 999),
                "run_all_included": bool(definition.get("run_all_included", True)),
                "active_run": (
                    {
                        "job_id": run.get("job_id"),
                        "status": run.get("status"),
                        "summary": run.get("summary") or {},
                        "started_at": run.get("started_at"),
                    }
                    if run
                    else None
                ),
            }
        )
    return rows


# ------------------------------------------------------------------
# Orario/intervallo editabile da Impostazioni (2026-09-09, richiesto
# esplicitamente dall'operatore): stesso principio del toggle
# enabled/disabled sopra (file JSON riletto dallo scheduler PRIMA di ogni
# esecuzione, MAI un valore fissato una volta sola all'avvio - vedi
# `src/jobs/scheduler.py::_is_job_due`, che sostituisce i vecchi
# `IntervalTrigger`/`CronTrigger` con orario fisso). Se non esiste un
# override salvato, il valore effettivo resta quello di `AppConfig`
# (variabili d'ambiente) - stesso "default" di sempre, invariato per chi
# non tocca mai questa pagina.
_SCHEDULE_KEY = "best_models/job_schedule.json"


def _cfg_default_schedule(job_id: str, cfg: AppConfig) -> dict[str, int]:
    cfg_fields = JOB_DEFINITIONS[job_id]["cfg_fields"]
    return {key: getattr(cfg, attr_name) for key, attr_name in cfg_fields.items()}


def get_job_schedule_overrides() -> dict[str, dict[str, int]]:
    """Legge gli override di orario/intervallo salvati - SOLO i job con un
    override esplicito compaiono qui (un job mai toccato dall'operatore
    non ha una chiave, e resta sul default `AppConfig`)."""
    try:
        loaded = bucket_store.get_json(_SCHEDULE_KEY, default={})
    except Exception as exc:
        logger.warning("job_schedule.json illeggibile dal bucket (%s): uso i default AppConfig", exc)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _validate_schedule(job_id: str, schedule: dict[str, Any]) -> dict[str, int]:
    if job_id not in JOB_DEFINITIONS:
        raise ValueError(f"Job id non riconosciuto: {job_id}")

    expected_keys = set(JOB_DEFINITIONS[job_id]["cfg_fields"].keys())
    provided_keys = set(schedule.keys())
    if provided_keys != expected_keys:
        raise ValueError(
            f"Schedule per '{job_id}' deve avere ESATTAMENTE le chiavi {sorted(expected_keys)}, "
            f"ricevute {sorted(provided_keys)}"
        )

    validated: dict[str, int] = {}
    for key, raw_value in schedule.items():
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' per '{job_id}' deve essere un intero, ricevuto {raw_value!r}")
        low, high = _SCHEDULE_FIELD_BOUNDS[key]
        if not (low <= value <= high):
            raise ValueError(f"'{key}' per '{job_id}' deve essere tra {low} e {high}, ricevuto {value}")
        validated[key] = value
    return validated


def update_job_schedule(job_id: str, schedule: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Valida (chiavi attese per lo `schedule_kind` di quel job, range
    sensati) e salva un override di orario/intervallo, persistito su disco
    (scrittura atomica via file temporaneo + `os.replace`, stesso principio
    di `update_job_settings`). Solleva `ValueError` esplicito su chiavi/
    valori non validi (mai un override silenziosamente scartato)."""
    validated = _validate_schedule(job_id, schedule)

    with _SCHEDULE_LOCK:
        current = get_job_schedule_overrides()
        current[job_id] = validated
        bucket_store.put_json(_SCHEDULE_KEY, current)

    return current


def reset_job_schedule(job_id: str) -> dict[str, dict[str, int]]:
    """Rimuove l'override per `job_id` (torna al default `AppConfig`)."""
    if job_id not in JOB_DEFINITIONS:
        raise ValueError(f"Job id non riconosciuto: {job_id}")

    with _SCHEDULE_LOCK:
        current = get_job_schedule_overrides()
        current.pop(job_id, None)
        bucket_store.put_json(_SCHEDULE_KEY, current)

    return current


def resolve_job_schedule(job_id: str, cfg: Optional[AppConfig] = None) -> dict[str, int]:
    """Schedule EFFETTIVO per `job_id`: override salvato se presente,
    altrimenti il default calcolato da `AppConfig` (variabili d'ambiente) -
    usato SIA da `list_job_definitions` (per mostrarlo in Impostazioni) SIA
    da `src/jobs/scheduler.py::_is_job_due` (per decidere se e' ora di
    eseguire il job), cosi' i due punti non possono mai disallinearsi."""
    if job_id not in JOB_DEFINITIONS:
        raise ValueError(f"Job id non riconosciuto: {job_id}")

    overrides = get_job_schedule_overrides()
    if job_id in overrides:
        return dict(overrides[job_id])

    cfg = cfg or load_app_config()
    return _cfg_default_schedule(job_id, cfg)



_QUOTA_PAUSE_KEY = "best_models/quota_pause_state.json"


def _read_quota_pause_state() -> dict[str, Any]:
    try:
        payload = bucket_store.get_json(_QUOTA_PAUSE_KEY, default={})
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_quota_pause_state_strict() -> dict[str, Any]:
    """Come `_read_quota_pause_state`, ma un errore di LETTURA del bucket
    viene propagato: `sync_job_settings_with_quota` non deve scambiare "non
    riesco a leggere la pausa" per "nessuna pausa attiva" (salverebbe come
    baseline lo stato gia' disattivato dalla pausa in corso)."""
    payload = bucket_store.get_json(_QUOTA_PAUSE_KEY, default={})
    return payload if isinstance(payload, dict) else {}


def _write_quota_pause_state(state: dict[str, Any], *, strict: bool = False) -> None:
    """`strict=True`: un errore di scrittura viene propagato (usato quando la
    scrittura e' la PRECONDIZIONE di un passo successivo, es. salvare la
    baseline prima di disattivare i job)."""
    try:
        with _QUOTA_PAUSE_LOCK:
            bucket_store.put_json(_QUOTA_PAUSE_KEY, state)
    except Exception as exc:
        if strict:
            raise
        logger.warning("Impossibile persistere quota_pause_state.json sul bucket: %s", exc)


def get_quota_pause_state() -> dict[str, Any]:
    """Stato corrente dell'auto-pausa per quota esaurita, per il frontend
    (pagina Impostazioni): `{"paused_date": "2026-09-06", "previous_settings":
    {...}}` se la pausa e' attiva, `{}` se non in pausa."""
    return _read_quota_pause_state()


def remember_operator_toggles_during_pause(updates: dict[str, bool]) -> bool:
    """Se l'auto-pausa per quota e' attiva, memorizza nella baseline di
    ripristino i toggle ESPLICITI dell'operatore sui job quota-sensitive.

    Senza questo, un job spento a mano durante la pausa verrebbe riacceso
    di nascosto al reset (si ripristina lo stato pre-pausa) e uno acceso a
    mano verrebbe dimenticato. Ritorna True se la baseline e' stata
    aggiornata. Mai un'eccezione propagata."""
    try:
        touched = {j: bool(v) for j, v in updates.items() if j in QUOTA_SENSITIVE_JOB_IDS}
        if not touched:
            return False
        with _QUOTA_SYNC_LOCK:
            state = _read_quota_pause_state_strict()
            previous = state.get("previous_settings")
            if not state.get("paused_date") or not isinstance(previous, dict):
                return False
            previous = {**previous, **touched}
            _write_quota_pause_state({**state, "previous_settings": previous}, strict=True)
        return True
    except Exception as exc:
        logger.warning("remember_operator_toggles_during_pause fallita (ignorata): %s", exc)
        return False


def sync_job_settings_with_quota(quota_snapshot: Optional[dict[str, Any]] = None) -> bool:
    """Auto-pausa SOLO i job che non possono lavorare senza API-Sports
    (`QUOTA_SENSITIVE_JOB_IDS` - MAI `ml_training`, che lavora solo su dati
    gia' a DB, ne' `data_settlement`, che a quota esaurita salta da solo
    l'import dal provider e continua con la riconciliazione locale)
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

    Chiamata dallo scheduler prima di valutare un job che chiama API-Sports
    (vedi `src/jobs/scheduler.py::_run_if_due` ->
    `_sync_quota_pause_if_needed`, al massimo una volta al minuto per
    processo) e ad ogni lettura/scrittura delle Impostazioni
    (`src/api/main.py`) - mai un'eccezione propagata: un fallimento qui non
    deve MAI bloccare l'esecuzione normale dei job.
    Ritorna True se una pausa e' (rimasta) attiva."""
    try:
        with _QUOTA_SYNC_LOCK:
            return _sync_job_settings_with_quota_locked(quota_snapshot)
    except Exception as exc:  # mai bloccare lo scheduler/le Impostazioni
        logger.warning("sync_job_settings_with_quota fallita (ignorata): %s", exc)
        return False


def _sync_job_settings_with_quota_locked(quota_snapshot: Optional[dict[str, Any]]) -> bool:
    """Corpo di `sync_job_settings_with_quota`, eseguito sotto
    `_QUOTA_SYNC_LOCK`.

    Tutte le letture sono STRETTE (un errore del bucket viene propagato e il
    chiamante lo ignora fino al controllo successivo): ognuna delle tre, se
    ripiegasse su un valore vuoto, porterebbe a una scrittura sbagliata -
    quota illeggibile = "non esaurita" (pausa tolta), pausa illeggibile =
    "nessuna pausa" (baseline sovrascritta con lo stato gia' disattivato),
    Impostazioni illeggibili = default (toggle dell'operatore azzerati)."""
    snapshot = quota_snapshot if quota_snapshot is not None else read_quota_snapshot_strict()
    exhausted_today = is_quota_exhausted_today(snapshot)
    pause_state = _read_quota_pause_state_strict()
    paused_date = pause_state.get("paused_date")
    today_iso = datetime.now(timezone.utc).date().isoformat()

    if exhausted_today:
        current = _read_job_settings_strict()
        if paused_date != today_iso:
            # Prima rilevazione di oggi: salva la baseline (SOLO i job
            # quota-sensitive) PRIMA di disattivarli, cosi' da poterla
            # ripristinare esattamente al reset della quota.
            previous = {job_id: current[job_id] for job_id in QUOTA_SENSITIVE_JOB_IDS}
            if paused_date and pause_state.get("previous_settings"):
                # Pausa di un giorno precedente mai ripristinata (es. quota
                # di nuovo esaurita prima del primo tick dopo il reset):
                # `current` e' gia' tutto disattivato dalla pausa, NON e'
                # la baseline. Si conserva quella originale.
                previous = dict(pause_state["previous_settings"])
            # Scrittura STRETTA: se la baseline non viene salvata i job non
            # vanno disattivati, altrimenti al controllo successivo la
            # "baseline" diventerebbe lo stato gia' spento.
            _write_quota_pause_state({"paused_date": today_iso, "previous_settings": previous}, strict=True)
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
        # Solo job ancora esistenti: una baseline salvata prima di un deploy
        # che ha rimosso un job farebbe fallire `update_job_settings` ad ogni
        # controllo, lasciando gli altri job in pausa per sempre.
        restorable = {
            job_id: bool(value)
            for job_id, value in previous.items()
            if job_id in JOB_DEFINITIONS
        } if isinstance(previous, dict) else {}
        if restorable:
            update_job_settings(restorable)
        _write_quota_pause_state({})
        logger.info("Quota API-Sports non piu' segnalata esaurita: job ripristinati allo stato pre-pausa.")
    return False


