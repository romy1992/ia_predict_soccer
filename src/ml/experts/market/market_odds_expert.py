"""Market/Odds Expert (EXP-04).

Espone probabilita' fair, dispersione bookmaker e movement delle quote,
SEMPRE rispettando un cutoff temporale esplicito (`as_of`): nessuno snapshot
con `captured_at > as_of` viene mai usato, per costruzione (difesa in
profondita': il filtro viene riapplicato qui anche se il chiamante ha gia'
filtrato a monte via `OddsSnapshotRepository.list_for_fixture_until`).

Nota su "closing" (acceptance criteria EXP-04):
la vera closing odd esiste solo se il cutoff richiesto e' a/dopo il kickoff
della partita. Se si genera una predizione pre-match con largo anticipo,
`closing` DEVE essere None: altrimenti si etichetterebbe come "closing" un
valore che in realta' e' solo l'ultimo dato disponibile in quel momento
(leakage semantico). Questo modulo e' la variante point-in-time-safe da
usare nella pipeline ML; il metodo legacy
`OddsSnapshotRepository.opening_latest_closing` resta invariato per l'uso
di sola visualizzazione nel Match Detail (nessun leakage li', perche' non
alimenta un modello: e' l'utente che consulta la dashboard oggi).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from src.ml.baselines.bookmaker_baseline import compute_market_baseline, implied_probability
from src.repository.odds_snapshot_repository import OddsSnapshotRepository

EXPERT_NAME = "market_odds"
EXPERT_CONFIG_VERSION = 1


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _filter_until(snapshots: list[dict[str, Any]], as_of: datetime) -> list[dict[str, Any]]:
    """Difesa in profondita': scarta sempre gli snapshot con captured_at > as_of."""
    filtered: list[dict[str, Any]] = []
    for row in snapshots:
        captured_at = _parse_datetime(row.get("captured_at"))
        if captured_at is None or captured_at > as_of:
            continue
        filtered.append(row)
    return filtered


def compute_fair_probabilities(snapshots: list[dict[str, Any]], market: str, as_of: datetime) -> dict[str, Any]:
    """Fair probabilities per outcome, da media quote dei bookmaker disponibili a `as_of`."""
    valid = _filter_until(snapshots, as_of)

    by_outcome: dict[str, list[float]] = {}
    bookmakers_by_outcome: dict[str, set[str]] = {}
    for row in valid:
        odd = row.get("odd")
        if odd is None or float(odd) <= 0:
            continue
        outcome = str(row.get("outcome") or "")
        by_outcome.setdefault(outcome, []).append(float(odd))
        bookmakers_by_outcome.setdefault(outcome, set()).add(str(row.get("bookmaker") or "unknown"))

    odds_rows = [
        {
            "outcome": outcome,
            "avg_odd": float(np.mean(values)),
            "bookmakers": len(bookmakers_by_outcome.get(outcome, set())),
        }
        for outcome, values in by_outcome.items()
    ]
    return compute_market_baseline(market=market, odds_rows=odds_rows)


def compute_bookmaker_dispersion(snapshots: list[dict[str, Any]], as_of: datetime) -> dict[str, dict[str, float]]:
    """Dispersione delle probabilita' implicite tra bookmaker diversi, per outcome."""
    valid = _filter_until(snapshots, as_of)

    by_outcome: dict[str, list[float]] = {}
    for row in valid:
        implied = implied_probability(row.get("odd"))
        if implied is None:
            continue
        outcome = str(row.get("outcome") or "")
        by_outcome.setdefault(outcome, []).append(implied)

    result: dict[str, dict[str, float]] = {}
    for outcome, values in by_outcome.items():
        arr = np.asarray(values, dtype=float)
        result[outcome] = {
            "bookmaker_count": int(arr.size),
            "mean_implied_probability": float(np.mean(arr)),
            "std_implied_probability": float(np.std(arr)),
        }
    return result


def compute_odds_movement(snapshots: list[dict[str, Any]], as_of: datetime) -> dict[str, dict[str, Any]]:
    """Movimento quote (opening -> latest AS OF il cutoff) per outcome, mediato tra bookmaker."""
    valid = _filter_until(snapshots, as_of)

    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in valid:
        outcome = str(row.get("outcome") or "")
        bookmaker = str(row.get("bookmaker") or "unknown")
        by_key.setdefault((bookmaker, outcome), []).append(row)

    opening_by_outcome: dict[str, list[float]] = {}
    latest_by_outcome: dict[str, list[float]] = {}
    for (_, outcome), rows in by_key.items():
        ordered = sorted(rows, key=lambda r: _parse_datetime(r.get("captured_at")) or as_of)
        opening_implied = implied_probability(ordered[0].get("odd"))
        latest_implied = implied_probability(ordered[-1].get("odd"))
        if opening_implied is None or latest_implied is None:
            continue
        opening_by_outcome.setdefault(outcome, []).append(opening_implied)
        latest_by_outcome.setdefault(outcome, []).append(latest_implied)

    result: dict[str, dict[str, Any]] = {}
    for outcome in set(opening_by_outcome) | set(latest_by_outcome):
        opening_vals = opening_by_outcome.get(outcome, [])
        latest_vals = latest_by_outcome.get(outcome, [])
        if not opening_vals or not latest_vals:
            continue
        opening_mean = float(np.mean(opening_vals))
        latest_mean = float(np.mean(latest_vals))
        result[outcome] = {
            "opening_implied_probability": opening_mean,
            "latest_implied_probability": latest_mean,
            "drift": float(latest_mean - opening_mean),
        }
    return result


def build_opening_latest_closing(
    snapshots: list[dict[str, Any]],
    as_of: datetime,
    kickoff_at: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Opening/latest/closing SOLO se temporalmente lecito (acceptance criteria EXP-04).

    `closing` viene valorizzato SOLO quando `kickoff_at` e' noto ed
    `as_of >= kickoff_at`. In tutti gli altri casi (predizione generata con
    anticipo, kickoff sconosciuto) `closing` resta `None`: dichiararlo
    comunque significherebbe spacciare per "quota di chiusura" un dato che
    in realta' e' solo l'ultimo disponibile in quel momento.
    """
    valid = _filter_until(snapshots, as_of)
    closing_allowed = kickoff_at is not None and as_of >= kickoff_at

    grouped: dict[tuple[str, str, str, Any, str], list[dict[str, Any]]] = {}
    for row in valid:
        key = (
            str(row.get("bookmaker") or "unknown"),
            str(row.get("market") or ""),
            str(row.get("period") or "full_time"),
            row.get("line"),
            str(row.get("outcome") or ""),
        )
        grouped.setdefault(key, []).append(row)

    payload: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        ordered = sorted(rows, key=lambda r: _parse_datetime(r.get("captured_at")) or as_of)
        opening = ordered[0]
        latest = ordered[-1]
        payload.append(
            {
                "bookmaker": key[0],
                "market": key[1],
                "period": key[2],
                "line": key[3],
                "outcome": key[4],
                "opening": {"odd": opening.get("odd"), "captured_at": opening.get("captured_at")},
                "latest": {"odd": latest.get("odd"), "captured_at": latest.get("captured_at")},
                "closing": (
                    {"odd": latest.get("odd"), "captured_at": latest.get("captured_at")}
                    if closing_allowed
                    else None
                ),
            }
        )

    payload.sort(key=lambda row: (row["market"], row["line"] or "", row["bookmaker"], row["outcome"]))
    return payload


class MarketOddsExpert:
    """Wrapper DB-aware: recupera gli snapshot filtrati e delega alle funzioni pure sopra."""

    VERSION: str = ""  # valorizzato sotto la classe

    def __init__(self, snapshot_repo: Optional[OddsSnapshotRepository] = None):
        self.snapshot_repo = snapshot_repo or OddsSnapshotRepository()

    def _fetch_snapshots(
        self,
        fixture_id: int,
        as_of: datetime,
        market: Optional[str] = None,
        period: Optional[str] = None,
        line: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        rows = self.snapshot_repo.list_for_fixture_until(
            fixture_id=fixture_id, prediction_at=as_of, market=market, period=period, line=line
        )
        return [row.to_dict() for row in rows]

    def build_market_signal(
        self,
        fixture_id: int,
        market: str,
        as_of: datetime,
        kickoff_at: Optional[datetime] = None,
        period: str = "full_time",
        line: Optional[str] = None,
    ) -> dict[str, Any]:
        snapshots = self._fetch_snapshots(fixture_id=fixture_id, as_of=as_of, market=market, period=period, line=line)
        return {
            "signal_version": self.VERSION,
            "fixture_id": int(fixture_id),
            "market": market,
            "as_of": as_of.isoformat(),
            "fair_probabilities": compute_fair_probabilities(snapshots, market=market, as_of=as_of),
            "dispersion": compute_bookmaker_dispersion(snapshots, as_of=as_of),
            "movement": compute_odds_movement(snapshots, as_of=as_of),
            "opening_latest_closing": build_opening_latest_closing(snapshots, as_of=as_of, kickoff_at=kickoff_at),
        }

    @staticmethod
    def _build_version() -> str:
        config = {"expert": EXPERT_NAME, "config_version": EXPERT_CONFIG_VERSION}
        return hashlib.sha1(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:16]


MarketOddsExpert.VERSION = MarketOddsExpert._build_version()

