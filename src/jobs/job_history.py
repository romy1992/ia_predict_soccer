from __future__ import annotations

import datetime as dt
import json
import os
from uuid import uuid4
from typing import Any, Optional


class JobHistory:
    """Track import/retrain job executions for dashboard status."""

    _UNSET = object()

    def __init__(self, path: str = os.path.join("best_models", "jobs_history.jsonl")):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    @staticmethod
    def _now_iso() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _duration_seconds(started_at: Optional[str], finished_at: Optional[str]) -> Optional[float]:
        if not started_at or not finished_at:
            return None
        try:
            start_dt = dt.datetime.fromisoformat(started_at)
            end_dt = dt.datetime.fromisoformat(finished_at)
            return float((end_dt - start_dt).total_seconds())
        except Exception:
            return None

    def _read_rows(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.path):
            return []

        rows: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def _write_rows(self, rows: list[dict[str, Any]]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _find_index(self, rows: list[dict[str, Any]], job_id: str) -> Optional[int]:
        for index, row in enumerate(rows):
            if str(row.get("job_id") or "") == str(job_id):
                return index
        return None

    def create_job(
        self,
        job_type: str,
        status: str,
        params: Optional[dict[str, Any]] = None,
        summary: Optional[dict[str, Any]] = None,
        error: Optional[dict[str, Any]] = None,
        job_id: Optional[str] = None,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
    ) -> dict[str, Any]:
        now_iso = self._now_iso()
        row = {
            "job_id": job_id or str(uuid4()),
            "job_type": job_type,
            "status": status,
            "timestamp": now_iso,
            "queued_at": now_iso if status == "queued" else None,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": self._duration_seconds(started_at=started_at, finished_at=finished_at),
            "params": params or {},
            "summary": summary or {},
            "error": error,
        }

        rows = self._read_rows()
        rows.append(row)
        self._write_rows(rows)
        return row

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
        summary: Optional[dict[str, Any]] = None,
        error: Any = _UNSET,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
    ) -> dict[str, Any]:
        rows = self._read_rows()
        index = self._find_index(rows=rows, job_id=job_id)
        if index is None:
            raise ValueError(f"Job id non trovato: {job_id}")

        row = rows[index]
        if status:
            row["status"] = status
        if params is not None:
            row["params"] = params
        if summary is not None:
            row["summary"] = summary
        if error is not self._UNSET:
            row["error"] = error
        if started_at is not None:
            row["started_at"] = started_at
        if finished_at is not None:
            row["finished_at"] = finished_at

        row["timestamp"] = self._now_iso()
        row["duration_seconds"] = self._duration_seconds(
            started_at=row.get("started_at"),
            finished_at=row.get("finished_at"),
        )

        rows[index] = row
        self._write_rows(rows)
        return row

    def queue_job(self, job_type: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return self.create_job(job_type=job_type, status="queued", params=params)

    def mark_running(self, job_id: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        now_iso = self._now_iso()
        rows = self._read_rows()
        index = self._find_index(rows=rows, job_id=job_id)
        if index is None:
            return self.create_job(job_type="unknown", status="running", params=params, job_id=job_id, started_at=now_iso)

        row = rows[index]
        row["status"] = "running"
        row["timestamp"] = now_iso
        row["started_at"] = row.get("started_at") or now_iso
        if params is not None:
            row["params"] = params
        rows[index] = row
        self._write_rows(rows)
        return row

    def mark_success(self, job_id: str, summary: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        now_iso = self._now_iso()
        return self.update_job(
            job_id=job_id,
            status="success",
            summary=summary or {},
            finished_at=now_iso,
            error=None,
        )

    def mark_failed(self, job_id: str, error: dict[str, Any]) -> dict[str, Any]:
        now_iso = self._now_iso()
        return self.update_job(
            job_id=job_id,
            status="failed",
            error=error,
            finished_at=now_iso,
        )

    def append(
        self,
        job_type: str,
        status: str,
        duration_seconds: float,
        details: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        now_iso = self._now_iso()
        row = self.create_job(
            job_type=job_type,
            status=status,
            params={},
            summary=details or {},
            error=details if status == "failed" else None,
            started_at=now_iso,
            finished_at=now_iso,
        )
        if duration_seconds is not None:
            rows = self._read_rows()
            index = self._find_index(rows=rows, job_id=row["job_id"])
            if index is not None:
                rows[index]["duration_seconds"] = float(duration_seconds)
                self._write_rows(rows)
                row = rows[index]
        return row

    def tail(self, limit: int = 100, job_type: Optional[str] = None, status: Optional[str] = None) -> list[dict[str, Any]]:
        rows = self._read_rows()
        filtered: list[dict[str, Any]] = []
        for row in rows:
            if job_type and row.get("job_type") != job_type:
                continue
            if status and row.get("status") != status:
                continue
            filtered.append(row)

        return filtered[-limit:]





