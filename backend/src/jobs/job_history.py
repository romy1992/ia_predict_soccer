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

    # Indice dei job in corso (queued/running): un solo piccolo file, cosi' la pagina Impostazioni sa
    # cosa sta girando SENZA elencare tutte le righe di cronologia ad ogni richiesta.
    _ACTIVE_KEY = "best_models/jobs_active.json"
    _ACTIVE_MAX_AGE_HOURS = 6

    def _track(self, job_id: str, job_type: Optional[str], active: bool) -> None:
        """Best-effort: un errore sull'indice non deve mai far fallire il job."""
        try:
            index = bucket_store.get_json(self._ACTIVE_KEY) or {}
            if active:
                index[job_id] = job_type or ""
            elif job_id in index:
                del index[job_id]
            else:
                return
            bucket_store.put_json(self._ACTIVE_KEY, index)
        except Exception:
            pass

    def active_runs(self) -> dict[str, dict[str, Any]]:
        """Ultima riga in corso per `job_type` (righe ferme da piu' di qualche ora = job morti, ignorate)."""
        try:
            index = bucket_store.get_json(self._ACTIVE_KEY) or {}
        except Exception:
            return {}
        cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=self._ACTIVE_MAX_AGE_HOURS)).isoformat()
        out: dict[str, dict[str, Any]] = {}
        for job_id in list(index):
            try:
                row = bucket_store.get_json(self._job_key(job_id))
            except Exception:
                continue
            if not row or row.get("status") not in ("queued", "running"):
                continue
            if str(row.get("timestamp") or "") < cutoff:
                continue
            job_type = str(row.get("job_type") or "")
            if job_type not in out or str(row.get("timestamp") or "") > str(out[job_type].get("timestamp") or ""):
                out[job_type] = row
        return out

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
        if status in ("queued", "running"):
            self._track(row["job_id"], job_type, True)
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
        if row.get("status") in ("queued", "running"):
            self._track(job_id, row.get("job_type"), True)
        else:
            self._track(job_id, row.get("job_type"), False)
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
            self._track(job_id, "unknown", True)
            return row

        row["status"] = "running"
        row["timestamp"] = now_iso
        row["started_at"] = row.get("started_at") or now_iso
        if params is not None:
            row["params"] = params
        bucket_store.put_json(key, row)
        self._track(job_id, row.get("job_type"), True)
        return row

    def set_progress(self, job_id: str, pct: int, phase: str) -> None:
        """Aggiorna l'avanzamento (0-100) senza toccare altro. Best-effort:
        un errore di scrittura sul bucket non deve mai far fallire il job."""
        try:
            row = bucket_store.get_json(self._job_key(job_id))
            if row is None:
                return
            summary = dict(row.get("summary") or {})
            summary["progress_pct"] = max(0, min(100, int(pct)))
            summary["phase"] = phase
            row["summary"] = summary
            row["timestamp"] = self._now_iso()
            bucket_store.put_json(self._job_key(job_id), row)
        except Exception:
            pass

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
        """Ultimi job. Non scarica l'intero prefisso: la dashboard chiama
        questo endpoint a ogni apertura e un GET per ogni job storico
        (uno ogni mezz'ora, per sempre) faceva morire la richiesta sul
        telefono prima del caricamento delle partite ("Failed to fetch")."""
        metas = bucket_store.list_object_meta(self.prefix + "/")

        def _modified_key(meta: dict[str, Any]) -> str:
            value = meta.get("last_modified")
            if value is None:
                return ""
            if hasattr(value, "isoformat"):
                return value.isoformat()
            return str(value)

        metas.sort(key=_modified_key)
        newest_first = list(reversed(metas))
        # Senza filtri bastano gli ultimi `limit` oggetti. Con un filtro
        # si guarda un po' piu' indietro (i fallimenti recenti stanno tra
        # i job nuovi), senza rileggere anni di cronologia.
        scan_cap = limit if not (job_type or status) else max(limit * 8, limit)
        chosen = newest_first[: max(scan_cap, 0)]
        rows = [row for row in bucket_store.get_json_many([meta["key"] for meta in chosen]) if row]

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
