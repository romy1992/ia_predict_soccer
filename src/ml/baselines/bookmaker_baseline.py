from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

EXCLUSIVE_MARKETS = {
    "h2h",
    "goal_no_goal",
    "under_over_1_5",
    "under_over_2_5",
    "under_over_3_5",
    "under_over_4_5",
}


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").strip().lower())


def _line_from_market(market: str) -> Optional[str]:
    if not market.startswith("under_over_"):
        return None
    return market.replace("under_over_", "").replace("_", ".")


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except Exception:
        return None
    return result if result > 0 else None


def implied_probability(odd: Any) -> Optional[float]:
    odd_float = _safe_float(odd)
    if odd_float is None:
        return None
    return 1.0 / odd_float


def _normalize_outcome_for_market(market: str, outcome: str) -> str:
    value = (outcome or "").strip()
    norm = _normalize_text(value)

    if market == "h2h":
        mapping = {
            "home": "Home",
            "draw": "Draw",
            "away": "Away",
        }
        return mapping.get(norm, value)

    if market == "goal_no_goal":
        mapping = {
            "yes": "Yes",
            "no": "No",
            "goal": "Yes",
            "nogoal": "No",
        }
        return mapping.get(norm, value)

    if market.startswith("under_over_"):
        threshold = _line_from_market(market)
        if norm.startswith("over"):
            return f"Over {threshold}" if threshold else value
        if norm.startswith("under"):
            return f"Under {threshold}" if threshold else value

    return value


def compute_market_baseline(market: str, odds_rows: list[dict[str, Any]]) -> dict[str, Any]:
    parsed_rows: list[dict[str, Any]] = []
    for row in odds_rows:
        odd = _safe_float(row.get("avg_odd"))
        implied_raw = implied_probability(odd)
        if odd is None or implied_raw is None:
            continue

        outcome = _normalize_outcome_for_market(market=market, outcome=str(row.get("outcome") or ""))
        parsed_rows.append(
            {
                "outcome": outcome,
                "avg_odd": odd,
                "implied_raw": implied_raw,
                "bookmakers": int(row.get("bookmakers") or 0),
            }
        )

    sum_implied_raw = float(sum(item["implied_raw"] for item in parsed_rows))
    is_exclusive = market in EXCLUSIVE_MARKETS

    sum_fair_probability = 0.0
    for item in parsed_rows:
        if is_exclusive and sum_implied_raw > 0:
            fair_probability = item["implied_raw"] / sum_implied_raw
        else:
            fair_probability = item["implied_raw"]
        item["fair_probability"] = float(fair_probability)
        sum_fair_probability += float(fair_probability)

    parsed_rows.sort(key=lambda item: item["outcome"])

    return {
        "market": market,
        "is_exclusive": is_exclusive,
        "sum_implied_raw": sum_implied_raw,
        "overround": (sum_implied_raw - 1.0) if is_exclusive and parsed_rows else None,
        "sum_fair_probability": sum_fair_probability,
        "outcomes": parsed_rows,
    }


def build_fixture_baseline(odds_summary: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    markets: dict[str, Any] = {}
    for market, rows in (odds_summary or {}).items():
        if not isinstance(rows, list) or not rows:
            continue
        markets[market] = compute_market_baseline(market=market, odds_rows=rows)

    return {
        "markets": markets,
        "generated": bool(markets),
    }


def get_market_outcome_baseline(
    fixture_baseline: dict[str, Any],
    market: str,
    outcome: str,
) -> Optional[dict[str, Any]]:
    market_payload = ((fixture_baseline or {}).get("markets") or {}).get(market)
    if not market_payload:
        return None

    normalized = _normalize_text(_normalize_outcome_for_market(market=market, outcome=outcome))
    for item in market_payload.get("outcomes") or []:
        if _normalize_text(str(item.get("outcome") or "")) == normalized:
            return item
    return None


def persist_fixture_baseline(
    fixture_id: int,
    fixture_baseline: dict[str, Any],
    output_dir: str = os.path.join("best_models", "baselines"),
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.abspath(os.path.join(output_dir, f"fixture_{int(fixture_id)}.json"))
    with open(file_path, "w", encoding="utf-8") as file_handle:
        json.dump(fixture_baseline, file_handle, ensure_ascii=False, indent=2)
    return file_path
