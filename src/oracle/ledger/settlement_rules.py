"""Regole canoniche di settlement per le PLAY ufficiali.

Il normale periodo ``full_time`` usa sempre il risultato dei 90 minuti più
recupero. AET/PEN descrivono lo stato della fixture, ma non cambiano
silenziosamente il periodo della scommessa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


RESULT_STATUSES = frozenset({"FT", "AET", "PEN"})
VOID_STATUS_BY_MATCH_STATUS = {
    "CANC": "void_cancelled",
    "WO": "void_cancelled",
    "ABD": "void_abandoned",
}
PENDING_MATCH_STATUSES = frozenset({"NS", "TBD", "1H", "HT", "2H", "ET", "BT", "P", "INT", "PST"})

_DC_MEMBERS = {
    "1x": frozenset({"Home", "Draw"}),
    "homedraw": frozenset({"Home", "Draw"}),
    "12": frozenset({"Home", "Away"}),
    "homeaway": frozenset({"Home", "Away"}),
    "x2": frozenset({"Draw", "Away"}),
    "drawaway": frozenset({"Draw", "Away"}),
    # Il modello DC legacy e' binario: la classe 0 rappresenta
    # esclusivamente la vittoria ospite, non l'intera X2.
    "away": frozenset({"Away"}),
}


@dataclass(frozen=True)
class SettlementResolution:
    actual_outcome: Optional[str] = None
    void_status: Optional[str] = None
    pending: bool = False
    is_push: bool = False


def _normalise(value: Any) -> str:
    return "".join(char for char in str(value or "").lower() if char.isalnum())


def _score_result(home_score: Any, away_score: Any) -> Optional[str]:
    if home_score is None or away_score is None:
        return None
    home, away = int(home_score), int(away_score)
    if home > away:
        return "Home"
    if home < away:
        return "Away"
    return "Draw"


def _line_value(market: str, line: Optional[str]) -> Optional[float]:
    raw = line
    if raw is None and market.startswith("under_over_"):
        raw = market.removeprefix("under_over_").replace("_", ".")
    if raw is None:
        raw = "9.5" if market == "corners" else "4.5" if market == "cards" else None
    if raw is None:
        return None
    try:
        return float(str(raw).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _total_market_outcome(total: int, line: float) -> tuple[str, bool]:
    if float(total) == float(line):
        return "Push", True
    direction = "Over" if total > line else "Under"
    return f"{direction} {line:g}", False


def resolve_settlement(
    *,
    market: str,
    match_status: str,
    home_score: Any,
    away_score: Any,
    stat_home: Optional[dict[str, Any]] = None,
    stat_away: Optional[dict[str, Any]] = None,
    period: str = "full_time",
    line: Optional[str] = None,
) -> SettlementResolution:
    """Risoluzione esplicita senza inventare risultati mancanti."""
    status = str(match_status or "").upper()
    if status in VOID_STATUS_BY_MATCH_STATUS:
        return SettlementResolution(void_status=VOID_STATUS_BY_MATCH_STATUS[status])
    if status not in RESULT_STATUSES:
        return SettlementResolution(pending=True)
    if period != "full_time":
        return SettlementResolution(void_status="void_market_rule")

    base_result = _score_result(home_score, away_score)
    if market in {"1x2", "h2h", "dc"}:
        if base_result is None:
            return SettlementResolution(void_status="void_no_result")
        if market == "h2h":
            # Mercato binario legacy: semanticamente Home Win / Not Home Win.
            return SettlementResolution(actual_outcome="Home" if base_result == "Home" else "Not Home")
        # Per Double Chance conserviamo l'esito base: ogni outcome viene
        # verificato indipendentemente tramite ``outcome_wins``.
        return SettlementResolution(actual_outcome=base_result)

    if base_result is None:
        return SettlementResolution(void_status="void_no_result")
    home, away = int(home_score), int(away_score)
    if market == "goal_no_goal":
        return SettlementResolution(actual_outcome="Yes" if home > 0 and away > 0 else "No")

    if market.startswith("under_over_"):
        market_line = _line_value(market, line)
        if market_line is None:
            return SettlementResolution(void_status="void_market_rule")
        outcome, push = _total_market_outcome(home + away, market_line)
        return SettlementResolution(actual_outcome=outcome, is_push=push)

    if market in {"corners", "cards"}:
        if not stat_home or not stat_away:
            return SettlementResolution(void_status="void_no_result")
        if market == "corners":
            values = (stat_home.get("corners"), stat_away.get("corners"))
        else:
            values = (
                None if stat_home.get("yellow_cards") is None or stat_home.get("red_cards") is None
                else int(stat_home["yellow_cards"]) + int(stat_home["red_cards"]),
                None if stat_away.get("yellow_cards") is None or stat_away.get("red_cards") is None
                else int(stat_away["yellow_cards"]) + int(stat_away["red_cards"]),
            )
        if any(value is None for value in values):
            return SettlementResolution(void_status="void_no_result")
        market_line = _line_value(market, line)
        if market_line is None:
            return SettlementResolution(void_status="void_market_rule")
        outcome, push = _total_market_outcome(sum(int(value) for value in values), market_line)
        return SettlementResolution(actual_outcome=outcome, is_push=push)

    return SettlementResolution(void_status="void_market_rule")


def outcome_wins(market: str, selected_outcome: str, actual_outcome: Optional[str]) -> Optional[bool]:
    if actual_outcome is None:
        return None
    if market == "dc":
        members = _DC_MEMBERS.get(_normalise(selected_outcome))
        return None if members is None else actual_outcome in members
    return _normalise(selected_outcome) == _normalise(actual_outcome)
