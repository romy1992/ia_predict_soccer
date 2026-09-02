from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.jobs.job_history import JobHistory
from src.service_ia.config.app_config import load_app_config
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
    logging.info("Start daily pipeline")
    run_manual_today_update()
    run_manual_settlement()
    run_manual_retrain()
    logging.info("End daily pipeline")


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


def start_scheduler() -> None:
    cfg = load_app_config()
    scheduler = BlockingScheduler(timezone="Europe/Rome")
    scheduler.add_job(
        run_daily_pipeline,
        trigger=CronTrigger(hour=cfg.scheduler_hour, minute=cfg.scheduler_minute),
        id="daily_pipeline_2300",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )

    logging.info("Scheduler started: daily pipeline at %02d:%02d", cfg.scheduler_hour, cfg.scheduler_minute)
    scheduler.start()


if __name__ == "__main__":
    start_scheduler()






