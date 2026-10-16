from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, Optional

from src.storage import bucket_store


class PredictionLogger:
    """Append prediction outcomes to a JSON-lines object on the bucket for
    dashboard monitoring. Scritto solo da `api` (nessuna race cross-processo
    con `scheduler`): round-trip get-append-put sull'intero oggetto,
    accettabile per un singolo writer."""

    # La dashboard chiede solo le ultime righe. Leggere l'intero jsonl a
    # ogni apertura (cresce a ogni previsione) saturava la risposta e sul
    # telefono la fetch falliva prima di mostrare le partite.
    _TAIL_BYTES = 256 * 1024

    def __init__(self, key: str = "best_models/predictions_log.jsonl"):
        self.key = key

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

        try:
            existing = bucket_store.get_bytes(self.key)
        except bucket_store.BucketKeyNotFound:
            existing = b""
        new_line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        bucket_store.put_bytes(self.key, existing + new_line)

        return row

    def tail(self, limit: int = 100, market: Optional[str] = None) -> list[Dict[str, Any]]:
        try:
            size = bucket_store.object_size(self.key)
        except bucket_store.BucketKeyNotFound:
            return []

        window = self._TAIL_BYTES if not market else max(self._TAIL_BYTES, 1024 * 1024)
        if size <= window:
            raw = bucket_store.get_bytes(self.key)
        else:
            raw = bucket_store.get_bytes_range(self.key, max(0, size - window), size - 1)
            # Il primo pezzo puo' essere una riga tagliata a meta'.
            cut = raw.find(b"\n")
            if cut >= 0:
                raw = raw[cut + 1 :]

        rows: list[Dict[str, Any]] = []
        for line in raw.decode("utf-8").splitlines():
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
