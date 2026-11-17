"""CLV del Prediction Ledger ufficiale.

`overall`/`breakdowns`/`rows` restano costruiti dalle sole PLAY (come
sempre). `by_decision` (additivo) ripete il riepilogo per PLAY, BORDERLINE,
NO BET e ALL, con il dettaglio per mercato.

Formula: `clv_odd_pct = quota presa / quota di chiusura - 1` (entrambe medie
fra bookmaker, vedi `src/oracle/backtest/clv.py`); `clv_prob` e' la
differenza fra fair probability (senza margine) alla giocata e alla chiusura.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from src.oracle.backtest.clv import (
    REASON_INSUFFICIENT_BET_DATA,
    ClvResult,
    _REASON_TEXT,
    build_clv_result,
    compute_clv_report,
    pricing_market_for,
)
from src.oracle.ledger.official_performance_service import (
    _DECISION_BUCKETS,
    OFFICIAL_COHORT,
    OFFICIAL_SOURCE,
)
from src.repository.odds_snapshot_repository import OddsSnapshotRepository
from src.repository.prediction_ledger_repository import PredictionLedgerRepository


def _summary(results: list[ClvResult]) -> dict[str, Any]:
    """Riepilogo di un gruppo di righe. `count`/`avg_*`/`positive_clv_rate`/
    `median_clv_odd_pct` sono calcolati sulle sole righe con CLV
    disponibile; la copertura e i motivi di indisponibilita' su tutte."""
    report = compute_clv_report(results)
    payload = dataclasses.asdict(report.overall)
    lead_minutes = [
        result.closing_minutes_before_kickoff
        for result in results
        if result.available and result.closing_minutes_before_kickoff is not None
    ]
    reasons = Counter(result.reason_code or "unknown" for result in results if not result.available)
    payload.update(
        {
            "total_records": report.total_considered,
            "available_records": report.available_count,
            "unavailable_records": report.unavailable_count,
            "coverage": (
                report.available_count / report.total_considered
                if report.total_considered
                else None
            ),
            # Additivi: perche' manca il CLV e quanto prima del kickoff e'
            # stata osservata la chiusura usata.
            "unavailable_reasons": dict(sorted(reasons.items())),
            "median_closing_minutes_before_kickoff": float(np.median(lead_minutes)) if lead_minutes else None,
        }
    )
    return payload


def _serialize(result: ClvResult, id_prediction: Optional[str] = None, decision: Optional[str] = None) -> dict[str, Any]:
    payload = dataclasses.asdict(result)
    payload["id_prediction"] = id_prediction
    payload["decision"] = decision
    payload["unavailable_reason"] = payload.pop("reason")
    payload["unavailable_reason_code"] = payload.pop("reason_code")
    return payload


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class _FixtureSnapshotCache:
    """Snapshot caricati una volta per fixture durante un report.

    Espone la stessa `list_for_fixture` del repository (e' cio' che usa
    `build_clv_result`), servita dalla memoria. Con un repository che offre
    `list_for_fixture_markets` basta UNA query per fixture per tutti i
    mercati richiesti (fino al kickoff); altrimenti una per mercato."""

    def __init__(self, repo: Any, needs: dict[int, tuple[set[str], Optional[datetime]]]):
        self.repo = repo
        self.needs = needs
        self._rows: dict[tuple[int, str], list[Any]] = {}
        self._loaded: set[int] = set()

    def _load(self, fixture_id: int, market: str) -> list[Any]:
        markets, until = self.needs.get(fixture_id, (set(), None))
        if fixture_id not in self._loaded and hasattr(self.repo, "list_for_fixture_markets") and markets:
            self._loaded.add(fixture_id)
            for row in self.repo.list_for_fixture_markets(
                fixture_id=fixture_id, markets=sorted(markets), until=until
            ):
                self._rows.setdefault((fixture_id, row.market), []).append(row)
            for name in markets:
                self._rows.setdefault((fixture_id, name), [])
        key = (fixture_id, market)
        if key not in self._rows:
            self._rows[key] = list(self.repo.list_for_fixture(fixture_id=fixture_id, market=market))
        return self._rows[key]

    def list_for_fixture(
        self,
        fixture_id: int,
        market: Optional[str] = None,
        period: Optional[str] = None,
        line: Optional[str] = None,
    ) -> list[Any]:
        if not market:
            return list(self.repo.list_for_fixture(fixture_id=fixture_id, market=market, period=period, line=line))
        rows = self._load(int(fixture_id), market)
        if period:
            rows = [row for row in rows if row.period == period]
        if line is not None:
            rows = [row for row in rows if row.line == line]
        return rows


class OfficialClvService:
    def __init__(
        self,
        ledger_repo: Optional[PredictionLedgerRepository] = None,
        snapshot_repo: Optional[OddsSnapshotRepository] = None,
    ):
        self.ledger_repo = ledger_repo or PredictionLedgerRepository()
        self.snapshot_repo = snapshot_repo or OddsSnapshotRepository()

    def _result_for_row(self, row: Any, snapshot_repo: Any = None) -> ClvResult:
        if row.odd is None:
            # Senza quota presa non esiste un CLV sulla quota: nessuna query.
            return ClvResult(
                market=row.market, outcome=row.outcome, fixture_id=row.fixture_id, model_name=row.model_name,
                odd_at_bet=None, p_fair_at_bet=row.p_market_fair, closing_odd=None, p_fair_closing=None,
                clv_odd_pct=None, clv_prob=None, available=False,
                reason=_REASON_TEXT[REASON_INSUFFICIENT_BET_DATA], reason_code=REASON_INSUFFICIENT_BET_DATA,
                pricing_market=pricing_market_for(row.market),
            )
        return build_clv_result(
            market=row.market,
            outcome=row.outcome,
            odd_at_bet=row.odd,
            p_fair_at_bet=row.p_market_fair,
            fixture_id=row.fixture_id,
            kickoff_at=row.kickoff_at,
            period=row.period,
            line=row.line,
            model_name=row.model_name,
            snapshot_repo=snapshot_repo or self.snapshot_repo,
            # Quote della giocata: la chiusura deve essere piu' recente,
            # altrimenti coinciderebbe con la quota presa (falso CLV = 0).
            bet_captured_at=row.captured_at,
            bet_odds_captured_at=row.odds_captured_at,
        )

    def _snapshot_cache(self, rows: list[Any]) -> _FixtureSnapshotCache:
        needs: dict[int, tuple[set[str], Optional[datetime]]] = {}
        for row in rows:
            markets, until = needs.get(int(row.fixture_id), (set(), None))
            pricing_market = pricing_market_for(row.market)
            markets.add(pricing_market)
            if pricing_market == "dc":
                markets.add("h2h")
            kickoff = _aware(row.kickoff_at)
            if kickoff is not None and (until is None or kickoff > until):
                until = kickoff
            needs[int(row.fixture_id)] = (markets, until)
        return _FixtureSnapshotCache(self.snapshot_repo, needs)

    def report(
        self,
        *,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        market: Optional[str] = None,
        league: Optional[int] = None,
        model_name: Optional[str] = None,
        policy_version: Optional[str] = None,
        include_by_decision: bool = True,
    ) -> dict[str, Any]:
        """`include_by_decision=False` calcola solo le PLAY (come prima):
        lo usa `/predictions/official/performance`, che legge soltanto
        l'overall PLAY, per non pagare il CLV di tutte le decisioni."""
        all_rows = self.ledger_repo.list_all(
            market=market,
            since=since,
            until=until,
            cohort=OFFICIAL_COHORT,
            league=league,
            model_name=model_name,
            policy_version=policy_version,
            limit=1_000_000,
        )
        rows = [row for row in all_rows if row.decision == "PLAY" and row.odd is not None]
        considered = all_rows if include_by_decision else rows
        cache = self._snapshot_cache(considered)
        result_by_row = {id(row): self._result_for_row(row, snapshot_repo=cache) for row in considered}
        paired = [(row, result_by_row[id(row)]) for row in rows]
        results = [result for _, result in paired]

        breakdowns: dict[str, dict[str, Any]] = {}
        dimensions = {
            "market": lambda row: row.market,
            "model": lambda row: row.model_name or "unknown",
            "league": lambda row: str(row.league or "unknown"),
            "period": lambda row: row.period or "unknown",
        }
        for dimension, key_fn in dimensions.items():
            groups: dict[str, list[ClvResult]] = {}
            for row, result in paired:
                groups.setdefault(str(key_fn(row)), []).append(result)
            breakdowns[dimension] = {key: _summary(value) for key, value in sorted(groups.items())}

        by_decision: dict[str, Any] = {}
        if include_by_decision:
            for label, decision_values in _DECISION_BUCKETS.items():
                bucket = [
                    (row, result_by_row[id(row)])
                    for row in all_rows
                    if decision_values is None or row.decision in decision_values
                ]
                by_market: dict[str, list[ClvResult]] = {}
                for row, result in bucket:
                    by_market.setdefault(str(row.market), []).append(result)
                by_decision[label] = {
                    "sample_size": len(bucket),
                    "overall": _summary([result for _, result in bucket]),
                    "breakdowns": {"market": {key: _summary(value) for key, value in sorted(by_market.items())}},
                }

        return {
            "source": OFFICIAL_SOURCE,
            "cohort": OFFICIAL_COHORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "filters": {
                "since": since.isoformat() if since else None,
                "until": until.isoformat() if until else None,
                "market": market,
                "league": league,
                "model": model_name,
                "policy_version": policy_version,
            },
            "sample_size": len(rows),
            "settled_count": sum(1 for row in rows if row.is_settled),
            "void_count": sum(1 for row in rows if str(row.settlement_status or "").startswith("void_")),
            "pending_count": sum(1 for row in rows if not row.is_settled),
            "overall": _summary(results),
            "breakdowns": breakdowns,
            "by_decision": by_decision,
            "rows": [_serialize(result, row.id_prediction, row.decision) for row, result in paired],
        }

    def detail(self, id_prediction: str) -> Optional[dict[str, Any]]:
        row = self.ledger_repo.get_by_id(id_prediction)
        if row is None or row.cohort != OFFICIAL_COHORT:
            return None
        return {
            "source": OFFICIAL_SOURCE,
            "cohort": OFFICIAL_COHORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **_serialize(self._result_for_row(row), row.id_prediction, row.decision),
        }
