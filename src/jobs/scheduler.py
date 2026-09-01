from __future__ import annotations

import logging
import time
from typing import Optional

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.jobs.job_history import JobHistory
from src.service_ia.config.app_config import load_app_config
from src.service_ia.pre_processing.download_match_service import calculate_mean, download_import_matches
from src.service_ia.training.train_multi_market import train_all_markets

logging.basicConfig(level=logging.INFO)


def run_manual_import(seasons: Optional[list[int]] = None, leagues: Optional[list[int]] = None) -> None:
    cfg = load_app_config()
    seasons = seasons or cfg.seasons
    leagues = leagues or cfg.leagues
    history = JobHistory()
    start = time.perf_counter()
    try:
        download_import_matches(seasons=seasons, leagues=leagues, is_next=False)
        for season in seasons:
            calculate_mean(with_season=season)
        history.append(
            job_type="import",
            status="success",
            duration_seconds=time.perf_counter() - start,
            details={"seasons": seasons, "leagues": leagues},
        )
    except Exception as exc:
        history.append(
            job_type="import",
            status="failed",
            duration_seconds=time.perf_counter() - start,
            details={"error": str(exc), "seasons": seasons, "leagues": leagues},
        )
        raise


def run_manual_retrain(
    markets: Optional[list[str]] = None,
    seasons: Optional[list[int]] = None,
    selection_method: str = "kbest",
) -> None:
    cfg = load_app_config()
    seasons = seasons or cfg.seasons
    history = JobHistory()
    start = time.perf_counter()
    try:
        results = train_all_markets(
            markets=markets,
            seasons=seasons,
            selection_method=selection_method,
            save_model=True,
        )
        history.append(
            job_type="retrain",
            status="success",
            duration_seconds=time.perf_counter() - start,
            details={
                "markets": markets,
                "seasons": seasons,
                "selection_method": selection_method,
                "results": [r.__dict__ for r in results],
            },
        )
    except Exception as exc:
        history.append(
            job_type="retrain",
            status="failed",
            duration_seconds=time.perf_counter() - start,
            details={
                "error": str(exc),
                "markets": markets,
                "seasons": seasons,
                "selection_method": selection_method,
            },
        )
        raise


def run_daily_pipeline() -> None:
    logging.info("Start daily pipeline")
    run_manual_import()
    run_manual_retrain()
    logging.info("End daily pipeline")


def run_manual_refresh_for_next_round() -> None:
    """Optional helper for manually downloading upcoming fixtures."""
    cfg = load_app_config()
    download_import_matches(seasons=cfg.seasons, leagues=cfg.leagues, is_next=True)
    for season in cfg.seasons:
        calculate_mean(with_season=season)


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


