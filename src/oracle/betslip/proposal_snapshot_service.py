from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional

from src.oracle.betslip.betslip_builder import BetslipGenerationResult, GeneratedSlip
from src.repository.betting_slip_proposal_repository import BettingSlipProposalRepository
from src.service_ia.model.match import BettingSlipProposalSnapshot


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class BetslipProposalSnapshotService:
    """Salva l'output ufficiale del generatore senza ricalcolarne le metriche."""

    def __init__(self, repo: Optional[BettingSlipProposalRepository] = None):
        self.repo = repo or BettingSlipProposalRepository()

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
        for slips in generation.profiles.values():
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
                    payload=payload,
                    generated_at=captured_at,
                )
                _, created = self.repo.save_revision(snapshot)
                report["proposals_created"] += int(created)
                report["proposals_unchanged"] += int(not created)
        return report

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
                }
            )
            result.append(payload)
        return result
