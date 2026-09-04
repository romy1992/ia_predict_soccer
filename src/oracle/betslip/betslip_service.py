"""Betslip Service (SLIP-03) — orchestrazione DB.

Genera le schedine (`betslip_builder.py`, puro) a partire dal Pick Pool
gia' calcolato per una giornata (SLIP-01, `PickPoolService.build_pool_for_day`,
che a sua volta riusa `DashboardService.get_day_matches` COSI' COM'E'):
nessuna nuova query odds/predizioni, nessun ricalcolo di edge/EV/decisione.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from src.oracle.betslip.betslip_builder import (
    DEFAULT_SLIP_PROFILES,
    BetslipGenerationResult,
    SlipProfile,
    generate_betslips,
)
from src.oracle.betslip.correlation_engine import DEFAULT_CORRELATION_RULESET, CorrelationRuleSet, candidates_from_pool_picks
from src.oracle.betslip.pick_pool import DEFAULT_PICK_POOL_POLICY, PickPoolPolicy, PickPoolResult
from src.oracle.betslip.pick_pool_service import PickPoolService


class BetslipService:
    def __init__(self, pick_pool_service: Optional[PickPoolService] = None):
        self.pick_pool_service = pick_pool_service or PickPoolService()

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

