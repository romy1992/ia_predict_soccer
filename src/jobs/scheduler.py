from __future__ import annotations

import functools
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.data.live.live_sync_job import run_manual_live_sync
from src.jobs.job_history import JobHistory
from src.jobs.job_settings import is_job_enabled
from src.service_ia.config.app_config import AppConfig, load_app_config
from src.service_ia.pre_processing.download_match_service import calculate_mean, download_import_matches
from src.service_ia.pre_processing.settlement_service import SettlementService
from src.service_ia.training.train_multi_market import train_all_markets

logging.basicConfig(level=logging.INFO)


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def run_manual_import(
    seasons: Optional[list[int]] = None,
    leagues: Optional[list[int]] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    fixture_date: Optional[str] = None,
    statuses: Optional[str] = None,
    days_ahead: Optional[int] = None,
    is_next: bool = False,
    job_id: Optional[str] = None,
    job_type: str = "import",
) -> dict:
    cfg = load_app_config()
    seasons = seasons or cfg.seasons
    leagues = leagues or cfg.leagues
    history = JobHistory()
    params = {
        "seasons": seasons,
        "leagues": leagues,
        "from_date": from_date,
        "to_date": to_date,
        "fixture_date": fixture_date,
        "statuses": statuses,
        "days_ahead": days_ahead,
        "is_next": is_next,
    }
    if job_id:
        history.mark_running(job_id=job_id, params=params)
    else:
        started = history.create_job(job_type=job_type, status="running", params=params, started_at=JobHistory._now_iso())
        job_id = started["job_id"]

    start = time.perf_counter()
    try:
        report = download_import_matches(
            seasons=seasons,
            leagues=leagues,
            is_next=is_next,
            from_date=from_date,
            to_date=to_date,
            fixture_date=fixture_date,
            statuses=statuses,
            days_ahead=days_ahead,
        )
        for season in seasons:
            calculate_mean(with_season=season)
        summary = {
            "report": report,
            "duration_seconds": time.perf_counter() - start,
        }
        history.mark_success(job_id=job_id, summary=summary)
        report["job_id"] = job_id
        return report
    except Exception as exc:
        history.mark_failed(
            job_id=job_id,
            error={
                "message": str(exc),
                "duration_seconds": time.perf_counter() - start,
                "params": params,
            },
        )
        raise


def run_manual_retrain(
    markets: Optional[list[str]] = None,
    seasons: Optional[list[int]] = None,
    selection_method: str = "kbest",
    job_id: Optional[str] = None,
) -> dict:
    cfg = load_app_config()
    seasons = seasons or cfg.seasons
    history = JobHistory()
    params = {
        "markets": markets,
        "seasons": seasons,
        "selection_method": selection_method,
    }
    if job_id:
        history.mark_running(job_id=job_id, params=params)
    else:
        started = history.create_job(job_type="retrain", status="running", params=params, started_at=JobHistory._now_iso())
        job_id = started["job_id"]

    start = time.perf_counter()
    try:
        results = train_all_markets(
            markets=markets,
            seasons=seasons,
            selection_method=selection_method,
            save_model=True,
        )
        summary = {
            "markets": markets,
            "seasons": seasons,
            "selection_method": selection_method,
            "results": [r.__dict__ for r in results],
            "duration_seconds": time.perf_counter() - start,
        }
        history.mark_success(job_id=job_id, summary=summary)
        return {"job_id": job_id, **summary}
    except Exception as exc:
        history.mark_failed(
            job_id=job_id,
            error={
                "message": str(exc),
                "duration_seconds": time.perf_counter() - start,
                "params": params,
            },
        )
        raise


def run_daily_pipeline() -> None:
    """DEPRECATO (OPS-01): incatenava update+settlement+retrain in un unico
    job, causando un retrain automatico ad OGNI ciclo di import - esattamente
    l'anti-pattern che OPS-01 rimuove dallo scheduler (vedi `build_scheduler`).
    Non piu' registrata come job; lasciata SOLO per compatibilita' di eventuali
    chiamate manuali/script esterni gia' esistenti (mai una funzione rimossa
    silenziosamente se ancora invocabile)."""
    logging.warning(
        "run_daily_pipeline e' deprecata (OPS-01): usa i job separati "
        "(data_sync_today/data_settlement/data_future_sync/ml_training) "
        "registrati da build_scheduler()."
    )
    run_manual_today_update()
    run_manual_settlement()
    run_manual_retrain()


def run_manual_today_update(
    target_date: Optional[str] = None,
    seasons: Optional[list[int]] = None,
    leagues: Optional[list[int]] = None,
    job_id: Optional[str] = None,
) -> dict:
    """Sincronizza fixture odierne (NS/live/final) con persistenza idempotente."""
    date_value = target_date or _today_iso()
    return run_manual_import(
        seasons=seasons,
        leagues=leagues,
        fixture_date=date_value,
        statuses="NS-1H-HT-2H-FT-AET-PEN-ABD",
        is_next=False,
        job_id=job_id,
        job_type="today_update",
    )


def run_manual_future_sync(
    days_ahead: int = 7,
    seasons: Optional[list[int]] = None,
    leagues: Optional[list[int]] = None,
    job_id: Optional[str] = None,
) -> dict:
    """Importa fixture future in una finestra configurabile, mantenendo solo status NS."""
    if days_ahead < 1:
        raise ValueError("days_ahead deve essere >= 1")

    from_day = datetime.now(timezone.utc).date().isoformat()
    to_day = (datetime.now(timezone.utc).date() + timedelta(days=days_ahead)).isoformat()
    return run_manual_import(
        seasons=seasons,
        leagues=leagues,
        from_date=from_day,
        to_date=to_day,
        statuses="NS",
        days_ahead=days_ahead,
        is_next=True,
        job_id=job_id,
        job_type="future_sync",
    )


def run_daily_refresh(
    seasons: Optional[list[int]] = None,
    leagues: Optional[list[int]] = None,
    days_ahead: int = 7,
    job_id: Optional[str] = None,
) -> dict:
    """Azione combinata per il bottone "Aggiorna tutto" del Data Center:

    1) importa i risultati (con statistiche+quote) delle partite giocate
       IERI per TUTTI i campionati censiti (`cfg.leagues` se non filtrati
       esplicitamente da `leagues`);
    2) sincronizza il calendario delle partite PROSSIME in una finestra di
       `days_ahead` giorni (default 7), che si arricchira' via via di quote
       man mano che le partite si avvicinano (le quote dell'API sports
       durano solo una settimana - vedi `download_match_service.py`).

    Le due sotto-fasi restano loggate anche singolarmente in
    `best_models/jobs_history.jsonl` (job_type `daily_refresh_played` /
    `future_sync`, chiamando le funzioni manuali gia' esistenti) cosi' da
    non perdere granularita' di debug in caso di fallimento parziale, oltre
    alla riga "ombrello" `daily_refresh` con il riepilogo di entrambe.
    """
    cfg = load_app_config()
    seasons = seasons or cfg.seasons
    leagues = leagues or cfg.leagues
    if days_ahead < 1:
        raise ValueError("days_ahead deve essere >= 1")

    history = JobHistory()
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    params = {
        "seasons": seasons,
        "leagues": leagues,
        "days_ahead": days_ahead,
        "played_date": yesterday,
    }
    if job_id:
        history.mark_running(job_id=job_id, params=params)
    else:
        started = history.create_job(job_type="daily_refresh", status="running", params=params, started_at=JobHistory._now_iso())
        job_id = started["job_id"]

    start = time.perf_counter()
    try:
        played_report = run_manual_import(
            seasons=seasons,
            leagues=leagues,
            fixture_date=yesterday,
            statuses="FT-AET-PEN-ABD",
            is_next=False,
            job_type="daily_refresh_played",
        )
        upcoming_report = run_manual_future_sync(
            days_ahead=days_ahead,
            seasons=seasons,
            leagues=leagues,
        )
        summary = {
            "played_date": yesterday,
            "played_matches": played_report,
            "upcoming_matches": upcoming_report,
            "duration_seconds": time.perf_counter() - start,
        }
        history.mark_success(job_id=job_id, summary=summary)
        return {"job_id": job_id, **summary}
    except Exception as exc:
        history.mark_failed(
            job_id=job_id,
            error={
                "message": str(exc),
                "duration_seconds": time.perf_counter() - start,
                "params": params,
            },
        )
        raise


def run_manual_settlement(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    seasons: Optional[list[int]] = None,
    leagues: Optional[list[int]] = None,
    job_id: Optional[str] = None,
) -> dict:
    """Riconcilia partite finali e marca completezza dati per settlement prediction."""
    history = JobHistory()
    params = {
        "from_date": from_date,
        "to_date": to_date,
        "seasons": seasons,
        "leagues": leagues,
    }
    if job_id:
        history.mark_running(job_id=job_id, params=params)
    else:
        started = history.create_job(job_type="settlement", status="running", params=params, started_at=JobHistory._now_iso())
        job_id = started["job_id"]

    start = time.perf_counter()
    service = SettlementService()

    try:
        report = service.run_settlement(
            from_date=from_date,
            to_date=to_date,
            seasons=seasons,
            leagues=leagues,
        )
        report["duration_seconds"] = time.perf_counter() - start
        history.mark_success(job_id=job_id, summary=report)
        report["job_id"] = job_id
        return report
    except Exception as exc:
        history.mark_failed(
            job_id=job_id,
            error={
                "message": str(exc),
                "duration_seconds": time.perf_counter() - start,
                "params": params,
            },
        )
        raise


def run_manual_refresh_for_next_round() -> None:
    """Compatibilità legacy: alias del nuovo future sync."""
    cfg = load_app_config()
    run_manual_future_sync(days_ahead=7, seasons=cfg.seasons, leagues=cfg.leagues)


def _add_job(
    scheduler: BlockingScheduler,
    func,
    trigger,
    job_id: str,
    misfire_grace_time: int,
) -> None:
    """Registra un job con `max_instances=1`/`coalesce=True` SEMPRE
    espliciti (acceptance criteria "max_instances/coalesce corretti"),
    centralizzati qui una volta sola invece che ripetuti (e potenzialmente
    disallineati) ad ogni chiamata di `add_job` - stesso principio "niente
    duplicazione" gia' seguito nel resto del progetto.

    - `max_instances=1`: mai due esecuzioni sovrapposte dello STESSO job
      (es. un data sync che dura piu' del previsto non ne fa partire un
      secondo in parallelo).
    - `coalesce=True`: se il trigger "perde" piu' esecuzioni per qualunque
      motivo (container fermo, scheduler in pausa), ne recupera SOLO UNA
      al riavvio - mai un burst di esecuzioni accumulate in coda.
    """
    scheduler.add_job(
        func,
        trigger=trigger,
        id=job_id,
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=misfire_grace_time,
    )


def _run_if_enabled(job_id: str, func, **kwargs) -> Optional[dict]:
    """Esegue `func` SOLO se il job e' abilitato in `job_settings.json`
    (pagina Impostazioni), controllato ad OGNI tick e non solo alla
    registrazione: un toggle da frontend ha quindi effetto immediato,
    senza richiedere il restart del container `scheduler`."""
    if not is_job_enabled(job_id):
        logging.info("Job '%s' disabilitato da Impostazioni: skip esecuzione.", job_id)
        return None
    return func(**kwargs)


def build_scheduler(cfg: Optional[AppConfig] = None) -> BlockingScheduler:
    """Costruisce lo scheduler con TUTTI i job registrati, SENZA avviarlo
    (`.start()` bloccherebbe il processo): separato da `start_scheduler`
    apposta per restare ispezionabile/testabile (`scheduler.get_jobs()`)
    senza dover mockare un loop bloccante.

    OPS-01 (REFACTOR) — "Data jobs frequenti" + "Training job indipendente"
    + "No retrain automatico ad ogni import": quattro job COMPLETAMENTE
    separati e indipendenti, ciascuno col proprio trigger/orario
    configurabile via env (`AppConfig` — mai un valore hardcoded inline):

    - `data_sync_today` (frequente, `IntervalTrigger` ogni
      `cfg.data_sync_interval_minutes` minuti): sincronizza le fixture
      odierne (NS/live/final) — l'unico job che deve girare spesso, per
      tenere aggiornati punteggi/stati in tempo quasi reale.
    - `data_settlement` (frequente, `IntervalTrigger` ogni
      `cfg.settlement_interval_minutes` minuti): riconcilia le partite
      concluse per il settlement (BET-06/dashboard), indipendente dal
      training.
    - `data_future_sync` (giornaliero, `CronTrigger`): importa le fixture
      future in una finestra di N giorni — non richiede la frequenza dei
      due job precedenti (le partite future non cambiano stato spesso).
    - `data_daily_refresh` (giornaliero, `CronTrigger`, orario configurabile
      via `cfg.daily_refresh_hour`/`minute`): STESSO job invocato dal
      bottone "Aggiorna tutto" della Sidebar (sempre visibile, in ogni
      pagina) - chiama `run_daily_refresh` per importare le partite di IERI
      (tutti i campionati censiti) + sincronizzare il calendario prossimo
      (`cfg.daily_refresh_days_ahead` giorni, default 7). Si sovrappone
      volutamente alla finestra futura di `data_future_sync`: se entrambi
      abilitati il calendario prossimo viene risincronizzato due volte al
      giorno (quote piu' fresche, ma doppio consumo quota API-Sports) - chi
      preferisce un solo giro puo' disattivare uno dei due da Impostazioni.
    - `ml_training` (giornaliero, `CronTrigger`, orario INDIPENDENTE dai
      data job): l'UNICO job che fa retrain — MAI innescato dal
      completamento di un data job, gira col proprio orario/frequenza
      configurabile separatamente (acceptance criteria "No retrain
      automatico ad ogni import").

    LIVE-01 aggiunge un job aggiuntivo, anch'esso indipendente:
    - `data_sync_live` (molto frequente, `IntervalTrigger` ogni
      `cfg.live_sync_interval_seconds` SECONDI): sincronizza il dataset
      LIVE distinto (`src/data/live/`, tabelle `live_*`) - MAI le tabelle
      pre-match `match`/`statistics`/`odds` (nessun impatto sul training).
    """
    cfg = cfg or load_app_config()
    scheduler = BlockingScheduler(timezone="Europe/Rome")

    # --- Data jobs (frequenti, indipendenti dal training) ---
    _add_job(
        scheduler,
        functools.partial(_run_if_enabled, "data_sync_today", run_manual_today_update),
        trigger=IntervalTrigger(minutes=cfg.data_sync_interval_minutes),
        job_id="data_sync_today",
        misfire_grace_time=max(60, cfg.data_sync_interval_minutes * 60),
    )
    _add_job(
        scheduler,
        functools.partial(_run_if_enabled, "data_settlement", run_manual_settlement),
        trigger=IntervalTrigger(minutes=cfg.settlement_interval_minutes),
        job_id="data_settlement",
        misfire_grace_time=max(60, cfg.settlement_interval_minutes * 60),
    )
    _add_job(
        scheduler,
        functools.partial(_run_if_enabled, "data_future_sync", run_manual_future_sync),
        trigger=CronTrigger(hour=cfg.future_sync_hour, minute=cfg.future_sync_minute),
        job_id="data_future_sync",
        misfire_grace_time=3600,
    )
    _add_job(
        scheduler,
        functools.partial(
            _run_if_enabled,
            "data_daily_refresh",
            run_daily_refresh,
            days_ahead=cfg.daily_refresh_days_ahead,
        ),
        trigger=CronTrigger(hour=cfg.daily_refresh_hour, minute=cfg.daily_refresh_minute),
        job_id="data_daily_refresh",
        misfire_grace_time=3600,
    )

    # --- Training job (indipendente, orario SEPARATO dai data job) ---
    _add_job(
        scheduler,
        functools.partial(_run_if_enabled, "ml_training", run_manual_retrain),
        trigger=CronTrigger(hour=cfg.training_hour, minute=cfg.training_minute),
        job_id="ml_training",
        misfire_grace_time=3600,
    )

    # --- Live job (LIVE-01, dataset SEPARATO, polling frequente in secondi) ---
    _add_job(
        scheduler,
        functools.partial(_run_if_enabled, "data_sync_live", run_manual_live_sync),
        trigger=IntervalTrigger(seconds=cfg.live_sync_interval_seconds),
        job_id="data_sync_live",
        misfire_grace_time=max(30, cfg.live_sync_interval_seconds * 2),
    )

    return scheduler


def start_scheduler() -> None:
    cfg = load_app_config()
    scheduler = build_scheduler(cfg)

    logging.info(
        "Scheduler started: data_sync_today ogni %d min, data_settlement ogni %d min, "
        "data_future_sync alle %02d:%02d, data_daily_refresh (ieri+%dgg, come 'Aggiorna tutto') "
        "alle %02d:%02d, ml_training (indipendente) alle %02d:%02d, "
        "data_sync_live (LIVE-01) ogni %d sec",
        cfg.data_sync_interval_minutes,
        cfg.settlement_interval_minutes,
        cfg.future_sync_hour,
        cfg.future_sync_minute,
        cfg.daily_refresh_days_ahead,
        cfg.daily_refresh_hour,
        cfg.daily_refresh_minute,
        cfg.training_hour,
        cfg.training_minute,
        cfg.live_sync_interval_seconds,
    )
    scheduler.start()


if __name__ == "__main__":
    start_scheduler()





