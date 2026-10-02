from __future__ import annotations

import datetime as dt
from typing import Any, Optional
from uuid import uuid4

from src.storage import bucket_store


class JobHistory:
    """Track import/retrain job executions for dashboard status.

    Un oggetto per job (`<prefix>/<job_id>.json`) sul Bucket S3-compatible,
    non piu' un file `jobs_history.jsonl` condiviso via Volume. Il vecchio
    design (append su un file, protetto da `fcntl.flock` tra `api` e
    `scheduler`) esisteva per un bug reale gia' visto in produzione
    (`ValueError: Job id non trovato` da read-modify-write interleaved tra
    i due container). S3 non ha flock ne' compare-and-swap su questo
    bucket: un oggetto per job elimina la race alla radice invece di
    riprodurre il lock - nessun altro processo scrive MAI lo stesso
    `job_id` (ogni chiamante crea/possiede il proprio), quindi ogni
    mutazione e' un get-modifica-put sulla propria chiave, senza bisogno
    di sincronizzazione cross-processo."""

    _UNSET = object()

    def __init__(self, path: str = "best_models/jobs"):
        # Nome parametro invariato ("path") per compatibilita' con i call
        # site esistenti - semanticamente ora e' un PREFISSO di chiavi sul
        # bucket, non piu' un file locale.
        self.prefix = path.rstrip("/").replace("\\", "/")

    def _job_key(self, job_id: str) -> str:
        return f"{self.prefix}/{job_id}.json"

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

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        """Una riga per `job_id`, o `None`. Serve al polling del bottone
        "Ricalcola previsioni": dopo un refresh pagina il frontend riprende
        dallo stesso job, non dallo stato React perso."""
        if not job_id:
            return None
        return bucket_store.get_json(self._job_key(job_id))

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
        bucket_store.put_json(self._job_key(row["job_id"]), row)
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
        key = self._job_key(job_id)
        row = bucket_store.get_json(key)
        if row is None:
            raise ValueError(f"Job id non trovato: {job_id}")

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

        bucket_store.put_json(key, row)
        return row

    def queue_job(self, job_type: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return self.create_job(job_type=job_type, status="queued", params=params)

    def mark_running(self, job_id: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        now_iso = self._now_iso()
        key = self._job_key(job_id)
        row = bucket_store.get_json(key)
        if row is None:
            row = self._build_row(
                job_type="unknown", status="running", params=params, job_id=job_id, started_at=now_iso
            )
            bucket_store.put_json(key, row)
            return row

        row["status"] = "running"
        row["timestamp"] = now_iso
        row["started_at"] = row.get("started_at") or now_iso
        if params is not None:
            row["params"] = params
        bucket_store.put_json(key, row)
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
        bucket_store.put_json(self._job_key(row["job_id"]), row)
        return row

    def tail(self, limit: int = 100, job_type: Optional[str] = None, status: Optional[str] = None) -> list[dict[str, Any]]:
        rows = bucket_store.list_json(self.prefix + "/")
        filtered: list[dict[str, Any]] = []
        for row in rows:
            if job_type and row.get("job_type") != job_type:
                continue
            if status and row.get("status") != status:
                continue
            filtered.append(row)

        # Ordine esplicito per `timestamp`: a differenza del vecchio file ad
        # append, l'ordine di listing S3 non e' l'ordine di creazione.
        filtered.sort(key=lambda row: row.get("timestamp", ""))
        return filtered[-limit:]
