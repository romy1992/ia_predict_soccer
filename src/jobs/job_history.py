from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
from uuid import uuid4
from typing import Any, Optional

try:
    import fcntl
except ImportError:  # Windows (dev/test locale): nessun altro processo scrive lo stesso file
    fcntl = None


class JobHistory:
    """Track import/retrain job executions for dashboard status.

    `jobs_history.jsonl` e' condiviso (stesso volume Docker) tra il
    container `api` (bottoni manuali, es. "Ricalcola previsioni") e il
    container `scheduler` (heartbeat APScheduler, stesso job type in
    background): due PROCESSI diversi possono quindi fare
    read-modify-write sullo stesso file in concorrenza. Senza lock,
    l'ultimo a scrivere vince e puo' cancellare la riga appena creata
    dall'altro processo (visto in produzione come `ValueError: Job id non
    trovato` non appena un job pubblica progressi frequenti via
    `update_job`, es. la barra di avanzamento di "Ricalcola previsioni").
    `_locked()` (flock su un file di lock dedicato, valido tra processi
    diversi sullo stesso host/volume) serializza ogni read-modify-write;
    `_write_rows` scrive su file temporaneo + `os.replace` cosi' un
    lettore concorrente (es. `tail()`, che non prende il lock) vede sempre
    o il contenuto vecchio o quello nuovo, mai un file a meta'."""

    _UNSET = object()

    def __init__(self, path: str = os.path.join("best_models", "jobs_history.jsonl")):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock_path = f"{self.path}.lock"

    @contextlib.contextmanager
    def _locked(self):
        with open(self._lock_path, "a") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

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
        tmp_path = f"{self.path}.tmp-{os.getpid()}-{uuid4().hex}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, self.path)

    def _find_index(self, rows: list[dict[str, Any]], job_id: str) -> Optional[int]:
        for index, row in enumerate(rows):
            if str(row.get("job_id") or "") == str(job_id):
                return index
        return None

    def _build_row(
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
        return {
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
        row = self._build_row(
            job_type=job_type,
            status=status,
            params=params,
            summary=summary,
            error=error,
            job_id=job_id,
            started_at=started_at,
            finished_at=finished_at,
        )
        with self._locked():
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
        with self._locked():
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
        with self._locked():
            rows = self._read_rows()
            index = self._find_index(rows=rows, job_id=job_id)
            if index is None:
                row = self._build_row(
                    job_type="unknown", status="running", params=params, job_id=job_id, started_at=now_iso
                )
                rows.append(row)
                self._write_rows(rows)
                return row

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
        row = self._build_row(
            job_type=job_type,
            status=status,
            params={},
            summary=details or {},
            error=details if status == "failed" else None,
            started_at=now_iso,
            finished_at=now_iso,
        )
        if duration_seconds is not None:
            row["duration_seconds"] = float(duration_seconds)
        with self._locked():
            rows = self._read_rows()
            rows.append(row)
            self._write_rows(rows)
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




