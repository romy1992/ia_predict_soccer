import datetime as dt
import hashlib
import json
import logging
import os
import subprocess
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO)


class ModelRegistry:
    STAGES = {"candidate", "champion", "production", "retired"}

    """Persist metadata for each trained model and expose latest lookup helpers."""

    def __init__(self, registry_dir: str = os.path.join("best_models", "registry")):
        self.registry_dir = os.path.abspath(registry_dir)
        self.index_file = os.path.join(self.registry_dir, "index.jsonl")
        self.promotion_file = os.path.join(self.registry_dir, "promotion_history.jsonl")
        os.makedirs(self.registry_dir, exist_ok=True)

    @staticmethod
    def _validate_stage(stage: Optional[str]) -> str:
        value = (stage or "candidate").strip().lower()
        if value not in ModelRegistry.STAGES:
            raise ValueError(f"Stage non valido: {stage}")
        return value

    @staticmethod
    def _safe_jsonl_rows(path: str) -> list[Dict[str, Any]]:
        if not os.path.exists(path):
            return []

        rows: list[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    @staticmethod
    def _append_jsonl(path: str, payload: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _default_feature_version(feature_names: list[str]) -> str:
        if not feature_names:
            return "features:unspecified"
        signature = "|".join(sorted(str(name) for name in feature_names))
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
        return f"features:{len(feature_names)}:{digest}"

    @staticmethod
    def _normalize_windows(windows: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        values = windows if isinstance(windows, dict) else {}
        return {
            "train": values.get("train"),
            "validation": values.get("validation"),
            "test": values.get("test"),
        }

    @staticmethod
    def _detect_git_sha() -> Optional[str]:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        try:
            output = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return output or None
        except Exception:
            return None

    @staticmethod
    def _latest_row(rows: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not rows:
            return None
        return max(enumerate(rows), key=lambda item: (item[1].get("created_at", ""), item[0]))[1]

    @staticmethod
    def _row_base_stage(row: Dict[str, Any]) -> str:
        raw = (row.get("stage") or "candidate").strip().lower() if isinstance(row.get("stage"), str) else "candidate"
        return raw if raw in ModelRegistry.STAGES else "candidate"

    def _registration_rows(self) -> list[Dict[str, Any]]:
        return self._safe_jsonl_rows(self.index_file)

    def _promotion_rows(self) -> list[Dict[str, Any]]:
        return self._safe_jsonl_rows(self.promotion_file)

    def _lifecycle_maps(self) -> tuple[Dict[str, str], Dict[str, list[Dict[str, Any]]]]:
        registrations = self._registration_rows()
        stage_by_run: Dict[str, str] = {}
        history_by_run: Dict[str, list[Dict[str, Any]]] = {}

        for row in registrations:
            run_id = row.get("run_id")
            if not run_id:
                continue
            stage_by_run[run_id] = self._row_base_stage(row)
            history_by_run.setdefault(run_id, [])

        promotions = sorted(self._promotion_rows(), key=lambda item: item.get("changed_at", ""))
        for event in promotions:
            run_id = event.get("run_id")
            if not run_id or run_id not in stage_by_run:
                continue
            to_stage = (event.get("to_stage") or "").strip().lower()
            if to_stage not in self.STAGES:
                continue
            stage_by_run[run_id] = to_stage
            history_by_run.setdefault(run_id, []).append(event)

        return stage_by_run, history_by_run

    def _decorate_row(
        self,
        row: Dict[str, Any],
        stage_by_run: Dict[str, str],
        history_by_run: Dict[str, list[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        payload = dict(row)
        run_id = payload.get("run_id")
        base_stage = self._row_base_stage(payload)
        payload["stage"] = base_stage
        payload["current_stage"] = stage_by_run.get(run_id, base_stage)
        payload["promotion_history"] = history_by_run.get(run_id, [])
        return payload

    def _append_promotion_event(
        self,
        run_id: str,
        market: str,
        from_stage: str,
        to_stage: str,
        reason: Optional[str],
        actor: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        now_utc = dt.datetime.now(dt.timezone.utc)
        event = {
            "event_id": f"promotion_{run_id}_{now_utc.strftime('%Y%m%dT%H%M%S%fZ')}",
            "run_id": run_id,
            "market": market,
            "from_stage": from_stage,
            "to_stage": to_stage,
            "changed_at": now_utc.isoformat(),
            "actor": actor,
            "reason": reason,
            "metadata": metadata or {},
        }
        self._append_jsonl(self.promotion_file, event)

    def register(
        self,
        model_path: str,
        market: str,
        model_name: str,
        metrics: Optional[Dict[str, Any]] = None,
        feature_names: Optional[list[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        dataset_version: Optional[str] = None,
        feature_version: Optional[str] = None,
        windows: Optional[Dict[str, Any]] = None,
        git_sha: Optional[str] = None,
        stage: str = "candidate",
    ) -> Dict[str, Any]:
        now_utc = dt.datetime.now(dt.timezone.utc)
        timestamp = now_utc.strftime("%Y%m%dT%H%M%S%fZ")
        market_slug = market.replace("/", "_").replace(" ", "_")
        metadata_path = os.path.join(self.registry_dir, f"{market_slug}_{timestamp}.json")
        normalized_stage = self._validate_stage(stage)
        resolved_extra = extra or {}
        resolved_dataset_version = dataset_version or resolved_extra.get("dataset_version") or "dataset:unspecified"
        resolved_feature_version = (
            feature_version or resolved_extra.get("feature_version") or self._default_feature_version(feature_names or [])
        )
        resolved_windows = self._normalize_windows(windows or resolved_extra.get("windows"))
        resolved_git_sha = git_sha or resolved_extra.get("git_sha") or self._detect_git_sha()

        payload: Dict[str, Any] = {
            "run_id": f"{market_slug}_{timestamp}",
            "created_at": now_utc.isoformat(),
            "market": market,
            "model_name": model_name,
            "model_path": os.path.abspath(model_path),
            "metrics": metrics or {},
            "feature_names": feature_names or [],
            "params": params or {},
            "extra": resolved_extra,
            "dataset_version": resolved_dataset_version,
            "feature_version": resolved_feature_version,
            "windows": resolved_windows,
            "git_sha": resolved_git_sha,
            "stage": normalized_stage,
            "metadata_path": os.path.abspath(metadata_path),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        self._append_jsonl(self.index_file, payload)

        logging.info("Model metadata registered: %s", metadata_path)
        return payload

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        rows = self._registration_rows()
        match = None
        for row in rows:
            if row.get("run_id") == run_id:
                match = row
        if match is None:
            return None
        stage_by_run, history_by_run = self._lifecycle_maps()
        return self._decorate_row(match, stage_by_run=stage_by_run, history_by_run=history_by_run)

    def promote(
        self,
        run_id: str,
        to_stage: str,
        reason: Optional[str] = None,
        actor: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        target_stage = self._validate_stage(to_stage)
        run = self.get_run(run_id)
        if run is None:
            return None

        market = run.get("market") or ""
        current_stage = (run.get("current_stage") or run.get("stage") or "candidate").strip().lower()
        current_stage = current_stage if current_stage in self.STAGES else "candidate"
        if current_stage == target_stage:
            return run

        # Keep one production run per market; older production is demoted to champion.
        if target_stage == "production":
            current_production = self.get_production(market=market)
            if current_production and current_production.get("run_id") != run_id:
                previous_stage = (
                    current_production.get("current_stage") or current_production.get("stage") or "candidate"
                )
                self._append_promotion_event(
                    run_id=current_production["run_id"],
                    market=market,
                    from_stage=previous_stage,
                    to_stage="champion",
                    reason=f"Superseded by {run_id}",
                    actor=actor,
                    metadata={"superseded_by": run_id},
                )

        self._append_promotion_event(
            run_id=run_id,
            market=market,
            from_stage=current_stage,
            to_stage=target_stage,
            reason=reason,
            actor=actor,
            metadata=metadata,
        )
        return self.get_run(run_id)

    def get_latest(self, market: Optional[str] = None) -> Optional[Dict[str, Any]]:
        rows = self._registration_rows()
        if not rows:
            return None

        if market:
            rows = [row for row in rows if row.get("market") == market]
        latest = self._latest_row(rows)
        if latest is None:
            return None

        stage_by_run, history_by_run = self._lifecycle_maps()
        return self._decorate_row(latest, stage_by_run=stage_by_run, history_by_run=history_by_run)

    def get_production(self, market: Optional[str] = None) -> Optional[Dict[str, Any]]:
        rows = self._registration_rows()
        if not rows:
            return None

        if market:
            rows = [row for row in rows if row.get("market") == market]
        if not rows:
            return None

        stage_by_run, history_by_run = self._lifecycle_maps()
        production_rows = []
        for row in rows:
            run_id = row.get("run_id")
            current_stage = stage_by_run.get(run_id, self._row_base_stage(row))
            if current_stage == "production":
                production_rows.append(row)

        latest_production = self._latest_row(production_rows)
        if latest_production is None:
            return None
        return self._decorate_row(latest_production, stage_by_run=stage_by_run, history_by_run=history_by_run)

    def list_markets(self) -> list[str]:
        rows = self._registration_rows()
        markets = set()
        for row in rows:
            market = row.get("market")
            if market:
                markets.add(market)

        return sorted(markets)

    def tail(self, limit: int = 100, market: Optional[str] = None) -> list[Dict[str, Any]]:
        rows = self._registration_rows()
        if market:
            rows = [row for row in rows if row.get("market") == market]

        stage_by_run, history_by_run = self._lifecycle_maps()
        decorated = [
            self._decorate_row(row, stage_by_run=stage_by_run, history_by_run=history_by_run)
            for row in rows
        ]
        return decorated[-limit:]




