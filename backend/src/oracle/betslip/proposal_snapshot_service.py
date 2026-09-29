from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Optional

from src.oracle.betslip.betslip_builder import BetslipGenerationResult, GeneratedSlip
from src.oracle.ledger.ledger_service import PredictionLedgerService
from src.oracle.ledger.settlement_rules import outcome_wins
from src.repository.betting_slip_proposal_repository import BettingSlipProposalRepository
from src.service_ia.model.match import BettingSlipProposalSnapshot


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class BetslipProposalSnapshotService:
    """Salva l'output ufficiale del generatore senza ricalcolarne le metriche."""

    def __init__(
        self,
        repo: Optional[BettingSlipProposalRepository] = None,
        ledger_service: Optional[PredictionLedgerService] = None,
    ):
        self.repo = repo or BettingSlipProposalRepository()
        self.ledger_service = ledger_service or PredictionLedgerService()

    @staticmethod
    def _logical_id(reference_date: str, slip: GeneratedSlip) -> str:
        raw = f"{reference_date}:{slip.slip_id}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:48]

    @staticmethod
    def _snapshot_key(reference_date: str, payload: dict[str, Any]) -> str:
        # Include quote, probabilità e versioni: una variazione reale crea
        # una revisione; generated_at non è nel payload della singola slip.
        raw = json.dumps(
            {"reference_date": reference_date, "slip": payload},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def save_generation(
        self,
        *,
        reference_date: str,
        generation: BetslipGenerationResult,
        generated_at: Optional[datetime] = None,
    ) -> dict[str, int]:
        captured_at = _aware(
            generated_at
            or datetime.fromisoformat(generation.generated_at)
            if generation.generated_at
            else datetime.now(timezone.utc)
        )
        report = {"proposals_seen": 0, "proposals_created": 0, "proposals_unchanged": 0}
        profile_groups = (
            generation.decision_groups.values()
            if generation.decision_groups
            else [generation.profiles]
        )
        for profiles in profile_groups:
            for slips in profiles.values():
                self._save_slips(
                    slips=slips,
                    reference_date=reference_date,
                    captured_at=captured_at,
                    report=report,
                )
        return report

    def _save_slips(
        self,
        *,
        slips: list[GeneratedSlip],
        reference_date: str,
        captured_at: datetime,
        report: dict[str, int],
    ) -> None:
        for slip in slips:
            report["proposals_seen"] += 1
            payload = dataclasses.asdict(slip)
            model_versions = sorted(
                {
                    leg.model_run_id
                    for leg in slip.legs
                    if getattr(leg, "model_run_id", None)
                }
            )
            snapshot = BettingSlipProposalSnapshot(
                snapshot_key=self._snapshot_key(reference_date, payload),
                logical_slip_id=self._logical_id(reference_date, slip),
                reference_date=reference_date,
                profile=slip.profile_name,
                situation=slip.situation,
                event_count=slip.n_legs,
                combined_odd=slip.combined_odd,
                adjusted_probability=slip.adjusted_probability,
                combined_model_void_odd=slip.combined_model_void_odd,
                combined_edge_absolute=slip.combined_edge_absolute,
                combined_expected_roi=slip.combined_expected_roi,
                model_version=",".join(model_versions) or None,
                policy_version=slip.decision_policy_version,
                correlation_version=slip.correlation_ruleset_version,
                diversification_version=slip.diversification_policy_version,
                payload=payload,
                shadow_status="PENDING",
                shadow_stake=1.0,
                staking_policy_version="shadow_flat_unit_v1",
                generated_at=captured_at,
            )
            _, created = self.repo.save_revision(snapshot)
            report["proposals_created"] += int(created)
            report["proposals_unchanged"] += int(not created)

    def list_saved(
        self,
        *,
        reference_date: str,
        latest_only: bool = True,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Restituisce gli snapshot già salvati, senza rigenerare il passato."""
        rows = self.repo.list_all(
            reference_date=reference_date,
            latest_only=latest_only,
            limit=limit,
        )
        result = []
        for row in rows:
            payload = dict(row.payload or {})
            payload.update(
                {
                    "snapshot_id": row.id,
                    "reference_date": row.reference_date,
                    "saved_at": row.generated_at.isoformat() if row.generated_at else None,
                    "is_latest": bool(row.is_latest),
                    "shadow_status": getattr(row, "shadow_status", "PENDING"),
                    "shadow_stake": getattr(row, "shadow_stake", 1.0),
                    "shadow_effective_odd": getattr(row, "shadow_effective_odd", None),
                    "shadow_return": getattr(row, "shadow_return", None),
                    "shadow_profit": getattr(row, "shadow_profit", None),
                    "shadow_settlement": getattr(row, "shadow_settlement", None),
                    "shadow_settled_at": (
                        row.shadow_settled_at.isoformat()
                        if getattr(row, "shadow_settled_at", None)
                        else None
                    ),
                    "staking_policy_version": getattr(
                        row,
                        "staking_policy_version",
                        "shadow_flat_unit_v1",
                    ),
                }
            )
            result.append(payload)
        return result

    def settle_pending(
        self,
        *,
        before: Optional[datetime] = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        now = _aware(before or datetime.now(timezone.utc))
        report: dict[str, Any] = {
            "shadow_candidates": 0,
            "shadow_settled": 0,
            "shadow_pending": 0,
            "errors": [],
        }
        for snapshot in self.repo.list_pending_settlement(before=now, limit=limit):
            report["shadow_candidates"] += 1
            try:
                legs = list((snapshot.payload or {}).get("legs") or [])
                details = []
                for leg in legs:
                    kickoff_raw = leg.get("kickoff_at")
                    if kickoff_raw:
                        kickoff = _aware(datetime.fromisoformat(kickoff_raw.replace("Z", "+00:00")))
                        if kickoff > now:
                            details.append({**leg, "status": "PENDING"})
                            continue
                    resolution = self.ledger_service._resolve_outcome_for_fixture(
                        fixture_id=int(leg["fixture_id"]),
                        market=str(leg["market"]),
                        line=leg.get("line"),
                    )
                    if resolution.pending:
                        status = "PENDING"
                        void_reason = None
                    elif resolution.void_status or resolution.is_push:
                        status = "VOID"
                        void_reason = resolution.void_status or "void_push"
                    else:
                        won = outcome_wins(
                            str(leg["market"]),
                            str(leg["outcome"]),
                            resolution.actual_outcome,
                        )
                        status = "VOID" if won is None else "WON" if won else "LOST"
                        void_reason = "void_market_rule" if won is None else None
                    details.append(
                        {
                            **leg,
                            "status": status,
                            "actual_outcome": resolution.actual_outcome,
                            "void_reason": void_reason,
                        }
                    )

                statuses = [item["status"] for item in details]
                if not statuses:
                    snapshot.shadow_status = "VOID"
                elif "PENDING" in statuses:
                    snapshot.shadow_status = "PENDING"
                elif "LOST" in statuses:
                    snapshot.shadow_status = "LOST"
                elif all(status == "VOID" for status in statuses):
                    snapshot.shadow_status = "VOID"
                else:
                    snapshot.shadow_status = "WON"

                snapshot.shadow_settlement = {"legs": details}
                if snapshot.shadow_status == "PENDING":
                    report["shadow_pending"] += 1
                else:
                    active_odds = [
                        float(item.get("odd") or item.get("market_odd"))
                        for item in details
                        if item["status"] != "VOID"
                    ]
                    effective = math.prod(active_odds) if active_odds else 1.0
                    snapshot.shadow_effective_odd = effective
                    snapshot.shadow_return = (
                        snapshot.shadow_stake * effective
                        if snapshot.shadow_status == "WON"
                        else snapshot.shadow_stake
                        if snapshot.shadow_status == "VOID"
                        else 0.0
                    )
                    snapshot.shadow_profit = snapshot.shadow_return - snapshot.shadow_stake
                    snapshot.shadow_settled_at = now
                    report["shadow_settled"] += 1
                self.repo.save_settlement(snapshot)
            except Exception as exc:
                report["errors"].append({"snapshot_id": snapshot.id, "message": str(exc)})
        return report
