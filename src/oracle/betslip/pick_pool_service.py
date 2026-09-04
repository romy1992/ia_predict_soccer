"""Pick Pool Service (SLIP-01) — orchestrazione DB.

Costruisce i `CandidatePick` (`pick_pool.py`) da una giornata di partite
gia' servita da `DashboardService.get_day_matches` (MATCH-01: le
`decision_cards` - probabilita'/quota/fair probability/edge/EV/badge - sono
gia' calcolate quando le quote DB sono disponibili, NESSUN nuovo calcolo di
edge/EV/decisione qui), poi delega a `build_pick_pool`.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from src.api.dashboard_service import DashboardService
from src.oracle.betslip.pick_pool import (
    CandidatePick,
    DEFAULT_PICK_POOL_POLICY,
    PickPoolPolicy,
    PickPoolResult,
    build_pick_pool,
)


def candidate_from_decision_card(
    fixture_id: int,
    kickoff_at: Optional[str],
    card: dict[str, Any],
) -> CandidatePick:
    """Adapter: da UNA riga di `decision_cards` (gia' calcolata da
    `DashboardService._build_decision_cards`, BET-01/02/04) a
    `CandidatePick` - nessun ricalcolo di edge/EV/decisione, solo un
    mapping di campi gia' pronti."""
    return CandidatePick(
        fixture_id=int(fixture_id),
        market=card.get("market"),
        outcome=card.get("pick"),
        decision=card.get("value_label"),
        p_model=card.get("predicted_probability"),
        p_market_fair=card.get("bookmaker_fair_probability"),
        odd=card.get("odd"),
        fair_odd=card.get("fair_odd"),
        prob_edge=card.get("edge"),
        ev=card.get("ev"),
        samples=int(card.get("bookmakers_count") or 0),
        model_run_id=card.get("run_id"),
        model_name=card.get("model_name"),
        policy_version=card.get("policy_version"),
        kickoff_at=kickoff_at,
    )


class PickPoolService:
    def __init__(self, dashboard_service: Optional[DashboardService] = None):
        self.dashboard_service = dashboard_service or DashboardService()

    def candidates_for_day(
        self,
        target_date: date,
        markets: Optional[list[str]] = None,
    ) -> list[CandidatePick]:
        """Tutti i pick candidati (PLAY/BORDERLINE/NO BET - il filtro della
        policy avviene SOLO in `build_pick_pool`, non qui) per le fixture di
        `target_date` con `decision_cards` gia' calcolate. Nessuna nuova
        query odds/predizioni: riusa `get_day_matches` COSI' COM'E' (stessa
        query unica DB gia' ottimizzata, MATCH-01)."""
        day = self.dashboard_service.get_day_matches(
            target_date=target_date, limit=0, with_predictions=True, markets=markets
        )
        candidates: list[CandidatePick] = []
        for row in day.rows:
            fixture_id = row.get("fixture_id")
            if fixture_id is None:
                continue
            for card in row.get("decision_cards") or []:
                candidates.append(
                    candidate_from_decision_card(fixture_id=fixture_id, kickoff_at=row.get("datetime"), card=card)
                )
        return candidates

    def build_pool_for_day(
        self,
        target_date: date,
        policy: PickPoolPolicy = DEFAULT_PICK_POOL_POLICY,
        markets: Optional[list[str]] = None,
    ) -> PickPoolResult:
        candidates = self.candidates_for_day(target_date=target_date, markets=markets)
        return build_pick_pool(candidates=candidates, policy=policy)

