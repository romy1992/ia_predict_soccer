from __future__ import annotations

import datetime as dt
import json
import os
from typing import Any, Optional


class JobHistory:
    """Track import/retrain job executions for dashboard status."""

    def __init__(self, path: str = os.path.join("best_models", "jobs_history.jsonl")):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def append(
        self,
        job_type: str,
        status: str,
        duration_seconds: float,
        details: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        row = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "job_type": job_type,
            "status": status,
            "duration_seconds": float(duration_seconds),
            "details": details or {},
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def tail(self, limit: int = 100, job_type: Optional[str] = None) -> list[dict[str, Any]]:
        if not os.path.exists(self.path):
            return []

        rows: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if job_type and row.get("job_type") != job_type:
                    continue
                rows.append(row)

        return rows[-limit:]


