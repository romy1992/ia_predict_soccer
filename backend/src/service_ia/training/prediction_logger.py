from __future__ import annotations

import datetime as dt
import json
import os
from typing import Any, Dict, Optional


class PredictionLogger:
    """Append prediction outcomes to a JSONL file for dashboard monitoring."""

    def __init__(self, path: str = os.path.join("best_models", "predictions_log.jsonl")):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def log(
        self,
        fixture_id: int,
        market: str,
        prediction: int,
        probability: float,
        model_run_id: Optional[str],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        row = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "fixture_id": int(fixture_id),
            "market": market,
            "prediction": int(prediction),
            "probability": float(probability),
            "model_run_id": model_run_id,
            "extra": extra or {},
        }

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        return row

    def tail(self, limit: int = 100, market: Optional[str] = None) -> list[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []

        rows: list[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if market and obj.get("market") != market:
                    continue
                rows.append(obj)

        return rows[-limit:]


