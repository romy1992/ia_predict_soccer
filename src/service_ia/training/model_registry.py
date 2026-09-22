import dataclasses
import datetime as dt
import hashlib
import json
import logging
import os
import subprocess
from typing import Any, Dict, Optional

from src.ml.registry.promotion_policy import (
    DEFAULT_PROMOTION_POLICY,
    PromotionPolicy,
    evaluate_promotion,
)
from src.service_ia.training.model_paths import is_archived_model_path

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

    def list_active_markets(self) -> list[str]:
        """Come `list_markets()` ma esclude due categorie di mercati che
        un operatore ha chiesto esplicitamente di non offrire piu' come
        mercato attivo in Dashboard, pur restando entrambe caricabili e
        visibili nello storico (`list_markets()`/diagnostics le vedono
        ancora):

        1. Il modello in produzione e' stato lasciato in `archivio/` dalla
           riorganizzazione di `best_models/` del 2026-09-15 (es.
           under_over_4_5, mai rifatto con la procedura nuova a 33
           feature separate per esito) - il mercato ha ANCORA una
           `production` formale, solo la sua cartella e' quella vecchia.

        2. NESSUN modello in produzione E l'ultimo run registrato e'
           esplicitamente `retired` (2026-09-21, caso corners: 4 modelli
           ritirati perche' addestrati su dati contaminati e con ROI
           negativo verificato - vedi
           `docs/soccer_oracle_v2_detailed/PROMPT_mercato_corners.md`).
           Un mercato senza production e con l'ultimo run ancora
           `candidate` (mai stato promosso, non un ritiro esplicito)
           resta invece incluso: e' un caso diverso, gia' gestito a
           parte in Dashboard col badge "In coda"."""
        active = []
        for market in self.list_markets():
            production = self.get_production(market=market)
            if production is not None and is_archived_model_path(production.get("model_path")):
                continue
            if production is None:
                latest = self.get_latest(market=market)
                if latest is not None and latest.get("current_stage") == "retired":
                    continue
            active.append(market)
        return active

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

    # ------------------------------------------------------------------
    # OPS-02: Candidate -> Champion -> Production promotion
    # ------------------------------------------------------------------
    @staticmethod
    def _evaluation_payload(
        evaluation,
        run_id: str,
        market: str,
        production_run_id: Optional[str],
    ) -> Dict[str, Any]:
        payload = dataclasses.asdict(evaluation)
        payload["run_id"] = run_id
        payload["market"] = market
        payload["production_run_id"] = production_run_id
        return payload

    def evaluate_promotion(
        self,
        run_id: str,
        to_stage: str = "production",
        policy: PromotionPolicy = DEFAULT_PROMOTION_POLICY,
    ) -> Optional[Dict[str, Any]]:
        """Dry-run (acceptance criteria "Gate metriche" + "Confronto
        production/candidate"): valuta se `run_id` potrebbe essere
        promosso a `to_stage` SENZA eseguire alcuna modifica. Utile per
        ispezionare l'esito prima di decidere (endpoint API/CLI). Ritorna
        `None` solo se il run non esiste."""
        run = self.get_run(run_id)
        if run is None:
            return None

        target_stage = self._validate_stage(to_stage)
        market = run.get("market") or ""
        production = self.get_production(market=market) if target_stage == "production" else None
        production_run_id = production.get("run_id") if production else None

        evaluation = evaluate_promotion(
            candidate_metrics=run.get("metrics"),
            production_metrics=production.get("metrics") if production else None,
            to_stage=target_stage,
            policy=policy,
        )
        return self._evaluation_payload(evaluation, run_id=run_id, market=market, production_run_id=production_run_id)

    def promote_with_policy(
        self,
        run_id: str,
        to_stage: str = "production",
        policy: PromotionPolicy = DEFAULT_PROMOTION_POLICY,
        reason: Optional[str] = None,
        actor: str = "manual",
        force: bool = False,
    ) -> Dict[str, Any]:
        """Promozione CONTROLLATA (acceptance criteria "Ultimo training non
        diventa automaticamente production"): esegue `promote()` (invariato)
        SOLO se `evaluate_promotion` da' esito positivo, oppure se
        l'operatore forza esplicitamente il bypass (`force=True`, SEMPRE
        tracciato come override manuale nell'audit — mai un bypass
        silenzioso). Se il gate blocca e `force=False`, la promozione NON
        avviene e il tentativo viene comunque loggato nell'audit trail
        come evento bloccato SENZA alterare lo stage corrente del run
        (acceptance criteria "Audit promotion" — traccia anche i tentativi
        respinti, non solo le promozioni riuscite)."""
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run non trovato: {run_id}")

        target_stage = self._validate_stage(to_stage)
        market = run.get("market") or ""
        current_stage = (run.get("current_stage") or run.get("stage") or "candidate").strip().lower()
        current_stage = current_stage if current_stage in self.STAGES else "candidate"

        production = self.get_production(market=market) if target_stage == "production" else None
        production_run_id = production.get("run_id") if production else None

        evaluation = evaluate_promotion(
            candidate_metrics=run.get("metrics"),
            production_metrics=production.get("metrics") if production else None,
            to_stage=target_stage,
            policy=policy,
        )
        evaluation_payload = self._evaluation_payload(
            evaluation, run_id=run_id, market=market, production_run_id=production_run_id
        )

        if not evaluation.allowed and not force:
            # Evento "audit-only": to_stage=from_stage cosi' `_lifecycle_maps`
            # non altera lo stage corrente del run (mai un tentativo
            # bloccato che viene scambiato per una promozione riuscita).
            self._append_promotion_event(
                run_id=run_id,
                market=market,
                from_stage=current_stage,
                to_stage=current_stage,
                reason=reason,
                actor=actor,
                metadata={
                    "event_type": "promotion_blocked",
                    "attempted_stage": target_stage,
                    "policy_version": policy.version,
                    "evaluation": evaluation_payload,
                },
            )
            return {
                "promoted": False,
                "run_id": run_id,
                "market": market,
                "to_stage": target_stage,
                "evaluation": evaluation_payload,
                "run": self.get_run(run_id),
            }

        updated_run = self.promote(
            run_id=run_id,
            to_stage=target_stage,
            reason=reason,
            actor=actor,
            metadata={
                "event_type": "promotion_forced" if (not evaluation.allowed and force) else "promotion_approved",
                "policy_version": policy.version,
                "evaluation": evaluation_payload,
            },
        )
        return {
            "promoted": True,
            "run_id": run_id,
            "market": market,
            "to_stage": target_stage,
            "evaluation": evaluation_payload,
            "run": updated_run,
        }

    def _previous_production_run_id(self, market: str, exclude_run_id: Optional[str]) -> Optional[str]:
        promotions = sorted(
            (
                event
                for event in self._promotion_rows()
                if event.get("market") == market and (event.get("to_stage") or "").strip().lower() == "production"
            ),
            key=lambda item: item.get("changed_at", ""),
        )
        candidates = [event.get("run_id") for event in promotions if event.get("run_id") != exclude_run_id]
        return candidates[-1] if candidates else None

    def rollback(
        self,
        market: str,
        to_run_id: Optional[str] = None,
        reason: Optional[str] = None,
        actor: str = "manual",
    ) -> Dict[str, Any]:
        """Rollback esplicito: riporta lo stage 'production' di `market` a
        un run PRECEDENTE. Se `to_run_id` non e' specificato, usa l'ultima
        production PRECEDENTE a quella corrente (dall'audit trail
        `promotion_history`, MAI un'euristica indovinata). Bypassa
        DELIBERATAMENTE il gate metriche (e' un'azione di emergenza
        esplicita dell'operatore, sempre tracciata nell'audit con
        `event_type='rollback'`): un run gia' stato production in passato
        e' per definizione gia' passato da una validazione."""
        current_production = self.get_production(market=market)
        current_production_run_id = current_production.get("run_id") if current_production else None

        target_run_id = to_run_id or self._previous_production_run_id(
            market=market, exclude_run_id=current_production_run_id
        )
        if target_run_id is None:
            raise ValueError(f"Nessun run precedente disponibile per rollback sul mercato '{market}'")

        target_run = self.get_run(target_run_id)
        if target_run is None:
            raise ValueError(f"Run di rollback non trovato: {target_run_id}")
        if target_run.get("market") != market:
            raise ValueError(f"Il run {target_run_id} non appartiene al mercato '{market}'")

        updated_run = self.promote(
            run_id=target_run_id,
            to_stage="production",
            reason=reason or "rollback",
            actor=actor,
            metadata={
                "event_type": "rollback",
                "rolled_back_from": current_production_run_id,
            },
        )
        return {
            "market": market,
            "rolled_back_from": current_production_run_id,
            "rolled_back_to": target_run_id,
            "run": updated_run,
        }

    def list_promotion_events(self, market: Optional[str] = None, limit: int = 100) -> list[Dict[str, Any]]:
        """Audit trail COMPLETO (acceptance criteria "Audit promotion"):
        TUTTI gli eventi di `promotion_history.jsonl` in ordine
        cronologico, incluse le promozioni bloccate dal gate
        (`event_type='promotion_blocked'`) e i rollback
        (`event_type='rollback'`) — non solo le promozioni riuscite."""
        events = self._promotion_rows()
        if market:
            events = [event for event in events if event.get("market") == market]
        events = sorted(events, key=lambda item: item.get("changed_at", ""))
        return events[-limit:]




