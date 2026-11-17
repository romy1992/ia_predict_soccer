from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.oracle.ledger.official_clv_service import OfficialClvService

KICKOFF = datetime(2026, 11, 10, 20, 0, tzinfo=timezone.utc)
BET_AT = KICKOFF - timedelta(minutes=58)
BET_ODDS_AT = KICKOFF - timedelta(minutes=75)


class _Snapshot:
    def __init__(self, fixture_id, bookmaker, market, outcome, odd, captured_at, line=None):
        self.fixture_id = fixture_id
        self.bookmaker = bookmaker
        self.market = market
        self.period = "full_time"
        self.line = line
        self.outcome = outcome
        self.odd = odd
        self.captured_at = captured_at

    def to_dict(self):
        return {
            "fixture_id": self.fixture_id,
            "bookmaker": self.bookmaker,
            "market": self.market,
            "period": self.period,
            "line": self.line,
            "outcome": self.outcome,
            "odd": self.odd,
            "captured_at": self.captured_at.isoformat(),
        }


class _SnapshotRepo:
    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.calls = 0

    def list_for_fixture(self, fixture_id, market=None, period=None, line=None):
        self.calls += 1
        rows = [s for s in self.snapshots if s.fixture_id == fixture_id]
        if market:
            rows = [s for s in rows if s.market == market]
        if period:
            rows = [s for s in rows if s.period == period]
        if line is not None:
            rows = [s for s in rows if s.line == line]
        return sorted(rows, key=lambda s: s.captured_at)


class _LedgerRepo:
    def __init__(self, rows):
        self.rows = rows

    def list_all(self, **_kwargs):
        return list(self.rows)

    def get_by_id(self, id_prediction):
        return next((row for row in self.rows if row.id_prediction == id_prediction), None)


def _row(id_prediction, fixture_id, market, outcome, odd, decision="PLAY", line=None, p_fair=0.5, **extra):
    payload = dict(
        id_prediction=id_prediction,
        fixture_id=fixture_id,
        market=market,
        outcome=outcome,
        odd=odd,
        p_market_fair=p_fair,
        decision=decision,
        kickoff_at=KICKOFF,
        captured_at=BET_AT,
        odds_captured_at=BET_ODDS_AT,
        period="full_time",
        line=line,
        model_name="champion",
        league=135,
        cohort="official_paper",
        is_settled=False,
        settlement_status=None,
    )
    payload.update(extra)
    return SimpleNamespace(**payload)


def _h2h(fixture_id, home, draw, away, captured_at, bookmaker="bookA"):
    return [
        _Snapshot(fixture_id, bookmaker, "h2h", "Home", home, captured_at),
        _Snapshot(fixture_id, bookmaker, "h2h", "Draw", draw, captured_at),
        _Snapshot(fixture_id, bookmaker, "h2h", "Away", away, captured_at),
    ]


def test_play_without_snapshot_after_the_bet_has_no_clv_and_a_reason_code():
    snapshots = _h2h(1, 2.00, 3.40, 4.00, BET_ODDS_AT)
    service = OfficialClvService(
        ledger_repo=_LedgerRepo([_row("p1", 1, "1x2", "Home", 2.00)]),
        snapshot_repo=_SnapshotRepo(snapshots),
    )

    report = service.report()

    assert report["overall"]["available_records"] == 0
    assert report["overall"]["avg_clv_odd_pct"] is None
    row = report["rows"][0]
    assert row["available"] is False
    assert row["unavailable_reason_code"] == "closing_not_after_bet"
    assert row["closing_minutes_before_kickoff"] == 75.0


def test_detail_uses_closing_after_the_bet():
    snapshots = _h2h(1, 2.00, 3.40, 4.00, BET_ODDS_AT) + _h2h(1, 1.90, 3.50, 4.20, KICKOFF - timedelta(minutes=6))
    service = OfficialClvService(
        ledger_repo=_LedgerRepo([_row("p1", 1, "1x2", "Home", 2.00)]),
        snapshot_repo=_SnapshotRepo(snapshots),
    )

    detail = service.detail("p1")

    assert detail["available"] is True
    assert detail["unavailable_reason_code"] is None
    assert abs(detail["clv_odd_pct"] - (2.00 / 1.90 - 1.0)) < 1e-12
    assert detail["closing_minutes_before_kickoff"] == 6.0
