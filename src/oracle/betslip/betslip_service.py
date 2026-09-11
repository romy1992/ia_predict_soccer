"""Betslip Service (SLIP-03) — orchestrazione DB.

Genera le schedine (`betslip_builder.py`, puro) a partire dal Pick Pool
gia' calcolato per una giornata (SLIP-01, `PickPoolService.build_pool_for_day`,
che a sua volta riusa `DashboardService.get_day_matches` COSI' COM'E'):
nessuna nuova query odds/predizioni, nessun ricalcolo di edge/EV/decisione.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from src.oracle.betslip.betslip_builder import (
    DEFAULT_SLIP_PROFILES,
    BetslipGenerationResult,
    SlipProfile,
    generate_betslips,
)
from src.oracle.betslip.correlation_engine import DEFAULT_CORRELATION_RULESET, CorrelationRuleSet, candidates_from_pool_picks
from src.oracle.betslip.pick_pool import (
    DEFAULT_PICK_POOL_POLICY,
    PickPoolPolicy,
    PickPoolResult,
    build_pick_pool,
)
from src.oracle.betslip.pick_pool_service import PickPoolService
from src.oracle.betslip.proposal_snapshot_service import BetslipProposalSnapshotService


class BetslipService:
    def __init__(
        self,
        pick_pool_service: Optional[PickPoolService] = None,
        proposal_snapshot_service: Optional[BetslipProposalSnapshotService] = None,
    ):
        self.pick_pool_service = pick_pool_service or PickPoolService()
        self.proposal_snapshot_service = proposal_snapshot_service or BetslipProposalSnapshotService()

    def generate_for_day(
        self,
        target_date: date,
        pool_policy: PickPoolPolicy = DEFAULT_PICK_POOL_POLICY,
        markets: Optional[list[str]] = None,
        profiles: tuple[SlipProfile, ...] = DEFAULT_SLIP_PROFILES,
        ruleset: CorrelationRuleSet = DEFAULT_CORRELATION_RULESET,
        max_pool_size: int = 14,
        max_slips_per_profile: int = 5,
    ) -> tuple[PickPoolResult, BetslipGenerationResult]:
        """Ritorna sia il `PickPoolResult` (SLIP-01, per tracciabilita' di
        pool_id/policy_version) sia il `BetslipGenerationResult` (SLIP-03).
        Le pick del pool vengono convertite con `candidates_from_pool_picks`
        (SLIP-02) senza alcun ricalcolo."""
        pool_result = self.pick_pool_service.build_pool_for_day(
            target_date=target_date, policy=pool_policy, markets=markets
        )
        candidates = candidates_from_pool_picks(pool_result.picks)
        generation = generate_betslips(
            candidates,
            profiles=profiles,
            ruleset=ruleset,
            max_pool_size=max_pool_size,
            max_slips_per_profile=max_slips_per_profile,
        )
        return pool_result, generation

    def generate_exploration_for_day(
        self,
        target_date: date,
        pool_policy: PickPoolPolicy = DEFAULT_PICK_POOL_POLICY,
        markets: Optional[list[str]] = None,
        profiles: tuple[SlipProfile, ...] = DEFAULT_SLIP_PROFILES,
        ruleset: CorrelationRuleSet = DEFAULT_CORRELATION_RULESET,
        max_pool_size: int = 14,
        max_slips_per_profile: int = 5,
    ) -> tuple[PickPoolResult, BetslipGenerationResult]:
        """Genera tre coorti separate senza promuovere quelle esplorative."""
        candidates = self.pick_pool_service.candidates_for_day(
            target_date=target_date,
            markets=markets,
        )
        pool_result = build_pick_pool(candidates, policy=pool_policy)
        play = generate_betslips(
            candidates,
            profiles=profiles,
            ruleset=ruleset,
            max_pool_size=max_pool_size,
            max_slips_per_profile=max_slips_per_profile,
            allowed_decisions=frozenset({"PLAY"}),
        )
        borderline = generate_betslips(
            candidates,
            profiles=profiles,
            ruleset=ruleset,
            max_pool_size=max_pool_size,
            max_slips_per_profile=max_slips_per_profile * 3,
            allowed_decisions=frozenset({"PLAY", "BORDERLINE"}),
        )
        no_bet = generate_betslips(
            candidates,
            profiles=profiles,
            ruleset=ruleset,
            max_pool_size=max_pool_size,
            max_slips_per_profile=max_slips_per_profile * 3,
            allowed_decisions=frozenset({"PLAY", "BORDERLINE", "NO BET"}),
        )

        def only(result: BetslipGenerationResult, label: str) -> dict:
            return {
                profile: [slip for slip in slips if slip.situation == label][
                    :max_slips_per_profile
                ]
                for profile, slips in result.profiles.items()
            }

        def merge_groups(*groups: dict) -> dict:
            merged = {}
            for profile in (item.name for item in profiles):
                seen = set()
                rows = []
                for group in groups:
                    for slip in group.get(profile, []):
                        if slip.slip_id not in seen:
                            seen.add(slip.slip_id)
                            rows.append(slip)
                merged[profile] = rows[:max_slips_per_profile]
            return merged

        play_only = only(play, "PLAY")
        borderline_only = merge_groups(
            only(play, "BORDERLINE"),
            only(borderline, "BORDERLINE"),
        )
        no_bet_only = merge_groups(
            only(play, "NO BET"),
            only(borderline, "NO BET"),
            only(no_bet, "NO BET"),
        )
        play.profiles = play_only
        play.decision_groups = {
            "PLAY": play_only,
            "BORDERLINE": borderline_only,
            "NO BET": no_bet_only,
        }
        play.warnings.extend(f"borderline:{item}" for item in borderline.warnings)
        play.warnings.extend(f"no_bet:{item}" for item in no_bet.warnings)
        return pool_result, play

    def generate_and_snapshot_for_day(
        self,
        target_date: date,
        pool_policy: PickPoolPolicy = DEFAULT_PICK_POOL_POLICY,
        markets: Optional[list[str]] = None,
        profiles: tuple[SlipProfile, ...] = DEFAULT_SLIP_PROFILES,
        ruleset: CorrelationRuleSet = DEFAULT_CORRELATION_RULESET,
        max_pool_size: int = 14,
        max_slips_per_profile: int = 5,
        now: Optional[datetime] = None,
    ) -> tuple[PickPoolResult, BetslipGenerationResult, dict[str, int]]:
        """Generazione esplicita con snapshot idempotente delle proposte.

        È usata dal job server-side e dall'azione manuale POST; la GET di
        consultazione resta priva di scritture.
        """
        captured_at = now or datetime.now(timezone.utc)
        if captured_at.tzinfo is None:
            captured_at = captured_at.replace(tzinfo=timezone.utc)
        if target_date < captured_at.date():
            raise ValueError(
                "Le date passate sono disponibili solo in consultazione: "
                "non è consentito generare schedine retroattive"
            )

        pool_result, generation = self.generate_exploration_for_day(
            target_date=target_date,
            pool_policy=pool_policy,
            markets=markets,
            profiles=profiles,
            ruleset=ruleset,
            max_pool_size=max_pool_size,
            max_slips_per_profile=max_slips_per_profile,
        )
        skipped_started = 0

        def is_pre_kickoff(slip) -> bool:
            kickoffs = []
            for leg in slip.legs:
                if not leg.kickoff_at:
                    return False
                try:
                    parsed = datetime.fromisoformat(leg.kickoff_at.replace("Z", "+00:00"))
                except ValueError:
                    return False
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                kickoffs.append(parsed)
            return bool(kickoffs) and all(kickoff > captured_at for kickoff in kickoffs)

        source_groups = generation.decision_groups or {"PLAY": generation.profiles}
        filtered_groups = {}
        for decision, decision_profiles in source_groups.items():
            filtered_groups[decision] = {}
            for profile, slips in decision_profiles.items():
                eligible = []
                for slip in slips:
                    if is_pre_kickoff(slip):
                        slip.shadow_status = "PENDING"
                        eligible.append(slip)
                    else:
                        skipped_started += 1
                filtered_groups[decision][profile] = eligible
        generation.decision_groups = filtered_groups
        generation.profiles = filtered_groups.get("PLAY", {})
        if skipped_started:
            generation.warnings.append(f"not_saved_started_or_missing_kickoff:{skipped_started}")
        report = self.proposal_snapshot_service.save_generation(
            reference_date=target_date.isoformat(),
            generation=generation,
            generated_at=captured_at,
        )
        report["proposals_skipped_started"] = skipped_started
        return pool_result, generation, report

