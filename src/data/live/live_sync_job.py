"""Job "sync live" (LIVE-01) - stesso pattern try/except + `JobHistory`
gia' usato per gli altri job manuali (`run_manual_today_update`/
`run_manual_settlement` in `src/jobs/scheduler.py`), separato in un modulo
dedicato invece che dentro `scheduler.py` per restare coerenti con
l'indicazione del task ("File/aree da ispezionare: src/data/live/")."""

from __future__ import annotations

import time
from typing import Optional

from src.data.live.live_data_service import LiveDataService
from src.jobs.job_history import JobHistory
from src.service_ia.config.app_config import load_app_config


def run_manual_live_sync(
    leagues: Optional[list[int]] = None,
    include_events: bool = True,
    include_statistics: bool = True,
    job_id: Optional[str] = None,
) -> dict:
    """Sincronizza il dataset LIVE (fixtures/eventi/statistiche in corso).

    MAI un retrain/scrittura sul dataset pre-match: usa esclusivamente
    `LiveDataService`/`LiveDataRepository` (tabelle `live_*`)."""
    cfg = load_app_config()
    leagues = leagues if leagues is not None else cfg.leagues
    history = JobHistory()
    params = {
        "leagues": leagues,
        "include_events": include_events,
        "include_statistics": include_statistics,
    }
    if job_id:
        history.mark_running(job_id=job_id, params=params)
    else:
        started = history.create_job(job_type="live_sync", status="running", params=params, started_at=JobHistory._now_iso())
        job_id = started["job_id"]

    start = time.perf_counter()
    try:
        service = LiveDataService(cfg=cfg)
        report = service.sync_live_data(
            leagues=leagues,
            include_events=include_events,
            include_statistics=include_statistics,
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
