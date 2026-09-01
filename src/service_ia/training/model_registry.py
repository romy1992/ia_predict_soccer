import datetime as dt
import json
import logging
import os
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO)


class ModelRegistry:
    """Persist metadata for each trained model and expose latest lookup helpers."""

    def __init__(self, registry_dir: str = os.path.join("best_models", "registry")):
        self.registry_dir = os.path.abspath(registry_dir)
        self.index_file = os.path.join(self.registry_dir, "index.jsonl")
        os.makedirs(self.registry_dir, exist_ok=True)

    def register(
        self,
        model_path: str,
        market: str,
        model_name: str,
        metrics: Optional[Dict[str, Any]] = None,
        feature_names: Optional[list[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now_utc = dt.datetime.now(dt.timezone.utc)
        timestamp = now_utc.strftime("%Y%m%dT%H%M%S%fZ")
        market_slug = market.replace("/", "_").replace(" ", "_")
        metadata_path = os.path.join(self.registry_dir, f"{market_slug}_{timestamp}.json")

        payload: Dict[str, Any] = {
            "run_id": f"{market_slug}_{timestamp}",
            "created_at": now_utc.isoformat(),
            "market": market,
            "model_name": model_name,
            "model_path": os.path.abspath(model_path),
            "metrics": metrics or {},
            "feature_names": feature_names or [],
            "params": params or {},
            "extra": extra or {},
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        with open(self.index_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        logging.info("Model metadata registered: %s", metadata_path)
        return payload

    def get_latest(self, market: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.index_file):
            return None

        latest: Optional[Dict[str, Any]] = None
        with open(self.index_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if market and row.get("market") != market:
                    continue

                if latest is None or row.get("created_at", "") > latest.get("created_at", ""):
                    latest = row

        return latest

    def list_markets(self) -> list[str]:
        if not os.path.exists(self.index_file):
            return []

        markets = set()
        with open(self.index_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                market = row.get("market")
                if market:
                    markets.add(market)

        return sorted(markets)

    def tail(self, limit: int = 100, market: Optional[str] = None) -> list[Dict[str, Any]]:
        if not os.path.exists(self.index_file):
            return []

        rows: list[Dict[str, Any]] = []
        with open(self.index_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if market and row.get("market") != market:
                    continue
                rows.append(row)

        return rows[-limit:]




