"""Aggregazione giornaliera dei report di import (`JobHistory`).

Ogni esecuzione di un job di import (`import`, `daily_refresh`,
`today_update`, `future_sync`, `settlement`...) salva il proprio report
(`inserted`/`updated`/`skipped`/`failed`/...) come oggetto JSON via
`JobHistory.mark_success(job_id, summary=report)` - ma la FORMA di
`summary` varia per job_type (verificato sui dati reali in produzione,
2026-10-08):
  - `import`/`future_sync`: `summary["report"] = {inserted, updated, ...}`
  - `settlement`: `summary["import_report"] = {...}` (stesse chiavi, nome
    diverso)
  - `daily_refresh`: `summary["played_matches"] = {...}` (e potenzialmente
    altre sotto-fasi)
  - a volte: `summary["skipped_locked"] = True` (il job non e' nemmeno
    partito, un altro processo aveva gia' il lock - va contato a parte,
    NON come "0 fixture prese": sono due informazioni diverse).

Nessuno di questi report e' mai stato aggregato in uno storico
giorno-per-giorno leggibile (la Data Quality Dashboard mostra solo lo
stato ATTUALE cumulativo del DB, non l'andamento degli import nel tempo).

Estrazione GENERICA (non un mapping per job_type, che si romperebbe al
primo job nuovo o cambio di forma): cammina ricorsivamente dentro
`summary` (solo dict, mai liste - esclude naturalmente `params`/`errors`)
e somma OGNI sotto-dizionario che contiene tutte e quattro le chiavi
`inserted`/`updated`/`skipped`/`failed`. Se un run (es. `daily_refresh`)
contiene piu' sotto-report, vengono sommati entrambi: e' corretto, quella
giornata ha davvero fatto piu' lavoro.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from src.jobs.job_history import JobHistory

_REPORT_KEYS = ("inserted", "updated", "skipped", "failed")
_MAX_WALK_DEPTH = 4


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt_value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt_value if dt_value.tzinfo else dt_value.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _find_report_dicts(payload: Any, depth: int = 0) -> list[dict[str, Any]]:
    """Cammina ricorsivamente `payload` e ritorna ogni sotto-dizionario che
    ha la "forma" di un report di import (tutte e quattro le chiavi in
    `_REPORT_KEYS`). Mai dentro le liste (es. `params.leagues`, `errors`)."""
    if depth > _MAX_WALK_DEPTH or not isinstance(payload, dict):
        return []

    found: list[dict[str, Any]] = []
    if all(key in payload for key in _REPORT_KEYS):
        found.append(payload)
        return found  # un report-dict non contiene a sua volta altri report-dict rilevanti

    for value in payload.values():
        if isinstance(value, dict):
            found.extend(_find_report_dicts(value, depth=depth + 1))
    return found


def _sum_reports(summary: dict[str, Any]) -> dict[str, int]:
    totals = {"inserted": 0, "updated": 0, "skipped": 0, "failed": 0, "fixtures_seen": 0}
    for report in _find_report_dicts(summary):
        for key in ("inserted", "updated", "skipped", "failed"):
            totals[key] += int(report.get(key) or 0)
        totals["fixtures_seen"] += int(report.get("fixtures_seen") or 0)
    return totals


class JobDailySummaryService:
    """Aggrega `JobHistory.tail()` per giorno - dependency injection dello
    storico job (default `JobHistory()`, iniettabile per i test), stesso
    principio di `DataQualityService(match_repo=..., snapshot_repo=...)`."""

    def __init__(self, history: Optional[JobHistory] = None):
        self.history = history or JobHistory()

    def build_report(self, days: int = 30, job_type: Optional[str] = None) -> dict[str, Any]:
        if days <= 0:
            raise ValueError("days deve essere positivo")

        rows = self.history.tail(limit=5000, job_type=job_type)
        since = datetime.now(timezone.utc) - timedelta(days=days)

        by_date: dict[str, dict[str, Any]] = {}
        for row in rows:
            finished_at = _parse_iso_datetime(row.get("finished_at")) or _parse_iso_datetime(row.get("timestamp"))
            if finished_at is None or finished_at < since:
                continue
            date_key = finished_at.date().isoformat()

            bucket = by_date.setdefault(
                date_key,
                {
                    "date": date_key,
                    "runs": 0,
                    "locked_skips": 0,
                    "inserted": 0,
                    "updated": 0,
                    "skipped": 0,
                    "failed": 0,
                    "fixtures_seen": 0,
                    "by_job_type": defaultdict(lambda: {"runs": 0, "inserted": 0, "updated": 0, "skipped": 0, "failed": 0}),
                },
            )

            bucket["runs"] += 1
            summary = row.get("summary") or {}
            job_type_value = str(row.get("job_type") or "unknown")

            if summary.get("skipped_locked"):
                bucket["locked_skips"] += 1
                continue

            totals = _sum_reports(summary)
            for key in ("inserted", "updated", "skipped", "failed", "fixtures_seen"):
                bucket[key] += totals[key]

            per_type = bucket["by_job_type"][job_type_value]
            per_type["runs"] += 1
            for key in ("inserted", "updated", "skipped", "failed"):
                per_type[key] += totals[key]

        days_rows = []
        for bucket in by_date.values():
            bucket["by_job_type"] = dict(bucket["by_job_type"])
            days_rows.append(bucket)
        days_rows.sort(key=lambda row: row["date"], reverse=True)

        return {"days": days_rows}
