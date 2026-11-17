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


def _ou(fixture_id, over, under, captured_at, bookmaker="bookA"):
    return [
        _Snapshot(fixture_id, bookmaker, "under_over_2_5", "Over 2.5", over, captured_at, line="2.5"),
        _Snapshot(fixture_id, bookmaker, "under_over_2_5", "Under 2.5", under, captured_at, line="2.5"),
    ]


def _summary_fixture():
    closing_at = KICKOFF - timedelta(minutes=10)
    snapshots = []
    # Fixture 1: PLAY 1x2 Home 2.00 -> chiude 1.90 (CLV > 0), BORDERLINE O/U Over 2.10 -> 2.20 (CLV < 0).
    snapshots += _h2h(1, 2.00, 3.40, 4.00, BET_ODDS_AT) + _h2h(1, 1.90, 3.50, 4.20, closing_at)
    snapshots += _ou(1, 2.10, 1.75, BET_ODDS_AT) + _ou(1, 2.20, 1.70, closing_at)
    # Fixture 2: PLAY 1x2 Away 3.00 -> chiude 2.50 (CLV > 0); NO BET O/U senza snapshot dopo la giocata.
    snapshots += _h2h(2, 2.40, 3.20, 3.00, BET_ODDS_AT) + _h2h(2, 2.70, 3.30, 2.50, closing_at)
    snapshots += _ou(2, 1.95, 1.85, BET_ODDS_AT)
    # Fixture 3: PLAY O/U Over 1.80 -> chiude 1.80 (CLV = 0, non positivo); PLAY 1x2 senza snapshot.
    snapshots += _ou(3, 1.80, 2.00, BET_ODDS_AT) + _ou(3, 1.80, 2.00, closing_at)
    rows = [
        _row("p1", 1, "1x2", "Home", 2.00),
        _row("b1", 1, "under_over_2_5", "Over 2.5", 2.10, decision="BORDERLINE", line="2.5"),
        _row("p2", 2, "1x2", "Away", 3.00),
        _row("n2", 2, "under_over_2_5", "Under 2.5", 1.85, decision="NO BET", line="2.5"),
        _row("p3", 3, "under_over_2_5", "Over 2.5", 1.80, line="2.5"),
        _row("p4", 3, "1x2", "Draw", 3.30),
    ]
    return rows, snapshots


def test_by_decision_summary_per_market_with_median_positive_rate_and_coverage():
    rows, snapshots = _summary_fixture()
    service = OfficialClvService(ledger_repo=_LedgerRepo(rows), snapshot_repo=_SnapshotRepo(snapshots))

    report = service.report()

    clv_p1 = 2.00 / 1.90 - 1.0
    clv_p2 = 3.00 / 2.50 - 1.0
    play = report["by_decision"]["PLAY"]
    assert play["sample_size"] == 4
    overall = play["overall"]
    assert overall["total_records"] == 4
    assert overall["available_records"] == 3
    assert overall["count"] == 3
    assert abs(overall["avg_clv_odd_pct"] - (clv_p1 + clv_p2 + 0.0) / 3) < 1e-12
    assert abs(overall["median_clv_odd_pct"] - clv_p1) < 1e-12
    assert abs(overall["positive_clv_rate"] - 2 / 3) < 1e-12
    assert overall["coverage"] == 0.75
    assert overall["unavailable_reasons"] == {"no_snapshots": 1}
    assert overall["median_closing_minutes_before_kickoff"] == 10.0
    assert set(play["breakdowns"]["market"]) == {"1x2", "under_over_2_5"}
    assert play["breakdowns"]["market"]["1x2"]["available_records"] == 2
    assert play["breakdowns"]["market"]["1x2"]["unavailable_reasons"] == {"no_snapshots": 1}
    assert play["breakdowns"]["market"]["under_over_2_5"]["positive_clv_rate"] == 0.0

    # L'overall storico resta SOLO-PLAY e coincide con by_decision.PLAY.
    assert report["overall"] == overall
    assert report["sample_size"] == 4
    assert {row["id_prediction"] for row in report["rows"]} == {"p1", "p2", "p3", "p4"}
    assert {row["decision"] for row in report["rows"]} == {"PLAY"}

    borderline = report["by_decision"]["BORDERLINE"]["overall"]
    assert borderline["available_records"] == 1
    assert abs(borderline["avg_clv_odd_pct"] - (2.10 / 2.20 - 1.0)) < 1e-12
    assert borderline["positive_clv_rate"] == 0.0

    no_bet = report["by_decision"]["NO BET"]["overall"]
    assert no_bet["available_records"] == 0
    assert no_bet["coverage"] == 0.0
    assert no_bet["unavailable_reasons"] == {"closing_not_after_bet": 1}
    assert no_bet["avg_clv_odd_pct"] is None

    everything = report["by_decision"]["ALL"]
    assert everything["sample_size"] == 6
    assert everything["overall"]["available_records"] == 4
    assert everything["overall"]["unavailable_reasons"] == {"closing_not_after_bet": 1, "no_snapshots": 1}


def test_play_only_mode_skips_other_decisions():
    rows, snapshots = _summary_fixture()
    repo = _SnapshotRepo(snapshots)
    service = OfficialClvService(ledger_repo=_LedgerRepo(rows), snapshot_repo=repo)

    report = service.report(include_by_decision=False)

    assert report["by_decision"] == {}
    assert report["overall"]["total_records"] == 4
    # Solo i mercati delle PLAY: fixture 1 h2h, 2 h2h, 3 O/U + h2h.
    assert repo.calls == 4


def test_snapshots_are_loaded_once_per_fixture_when_the_repository_supports_it():
    rows, snapshots = _summary_fixture()

    class _BatchRepo(_SnapshotRepo):
        def __init__(self, snapshots):
            super().__init__(snapshots)
            self.batch_calls = []

        def list_for_fixture_markets(self, fixture_id, markets, until=None):
            self.batch_calls.append((fixture_id, tuple(markets), until))
            return [
                s for s in self.snapshots
                if s.fixture_id == fixture_id and s.market in markets and (until is None or s.captured_at <= until)
            ]

    repo = _BatchRepo(snapshots)
    report = OfficialClvService(ledger_repo=_LedgerRepo(rows), snapshot_repo=repo).report()

    assert repo.calls == 0
    assert sorted(call[0] for call in repo.batch_calls) == [1, 2, 3]
    assert all(call[2] == KICKOFF for call in repo.batch_calls)
    assert report["by_decision"]["ALL"]["overall"]["available_records"] == 4


def test_double_chance_rows_also_load_h2h_for_the_fair_probability():
    closing_at = KICKOFF - timedelta(minutes=5)
    snapshots = _h2h(7, 2.00, 3.50, 4.00, BET_ODDS_AT) + _h2h(7, 2.00, 3.50, 4.00, closing_at) + [
        _Snapshot(7, "bookA", "dc", "Home/Draw", 1.30, BET_ODDS_AT),
        _Snapshot(7, "bookA", "dc", "Home/Draw", 1.25, closing_at),
    ]
    service = OfficialClvService(
        ledger_repo=_LedgerRepo([_row("d1", 7, "dc", "Home/Draw", 1.30, p_fair=0.70)]),
        snapshot_repo=_SnapshotRepo(snapshots),
    )

    row = service.report()["rows"][0]

    raw = [1 / 2.00, 1 / 3.50, 1 / 4.00]
    assert row["available"] is True
    assert abs(row["p_fair_closing"] - (raw[0] + raw[1]) / sum(raw)) < 1e-12
    assert abs(row["clv_odd_pct"] - (1.30 / 1.25 - 1.0)) < 1e-12


def test_row_without_bet_odd_is_counted_as_uncovered_without_queries():
    repo = _SnapshotRepo([])
    service = OfficialClvService(
        ledger_repo=_LedgerRepo([_row("n1", 9, "under_over_2_5", "Over 2.5", None, decision="NO BET", line="2.5")]),
        snapshot_repo=repo,
    )

    report = service.report()

    assert report["by_decision"]["NO BET"]["overall"]["unavailable_reasons"] == {"insufficient_bet_data": 1}
    assert repo.calls == 0


def test_official_clv_endpoint_exposes_by_decision():
    from unittest import mock

    from fastapi.testclient import TestClient

    from src.api import main as api_main

    rows, snapshots = _summary_fixture()
    service = OfficialClvService(ledger_repo=_LedgerRepo(rows), snapshot_repo=_SnapshotRepo(snapshots))
    with (
        mock.patch.object(api_main, "OfficialClvService", lambda: service),
        mock.patch.object(api_main.auth, "decode_valid_token", lambda _token: "tester"),
    ):
        response = TestClient(api_main.app).get("/predictions/official/clv")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload["by_decision"]) == {"PLAY", "BORDERLINE", "NO BET", "ALL"}
    assert payload["by_decision"]["PLAY"]["overall"]["available_records"] == 3
    assert "median_clv_odd_pct" in payload["overall"]
    assert payload["rows"][0]["unavailable_reason_code"] in {None, "no_snapshots"}


def test_repository_loads_several_markets_up_to_kickoff_in_one_query():
    import uuid
    from unittest import mock

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from src.repository.odds_snapshot_repository import OddsSnapshotRepository
    from src.service_ia.model.match import Base, OddsSnapshot

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def _orm(fixture_id, market, outcome, odd, captured_at):
        return OddsSnapshot(
            id_snapshot=str(uuid.uuid4()), id_match=None, fixture_id=fixture_id, bookmaker="bookA",
            market=market, period="full_time", line=None, outcome=outcome, odd=odd,
            captured_at=captured_at, source="test",
        )

    with session_factory() as session:
        session.add_all(
            [
                _orm(5, "h2h", "Home", 2.0, KICKOFF - timedelta(minutes=5)),
                _orm(5, "h2h", "Home", 1.5, KICKOFF + timedelta(minutes=5)),
                _orm(5, "dc", "Home/Draw", 1.3, KICKOFF - timedelta(minutes=5)),
                _orm(5, "corners", "Over 9.5", 1.9, KICKOFF - timedelta(minutes=5)),
                _orm(6, "h2h", "Home", 2.2, KICKOFF - timedelta(minutes=5)),
            ]
        )
        session.commit()

    with mock.patch("src.repository.odds_snapshot_repository.SessionLocal", new=session_factory):
        repo = OddsSnapshotRepository()
        rows = repo.list_for_fixture_markets(fixture_id=5, markets=["h2h", "dc"], until=KICKOFF)
        assert repo.list_for_fixture_markets(fixture_id=5, markets=[]) == []

    assert sorted((row.market, row.odd) for row in rows) == [("dc", 1.3), ("h2h", 2.0)]
