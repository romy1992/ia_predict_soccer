from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.dashboard_service import DashboardService
from src.oracle.ledger.ledger_service import PredictionLedgerService
from src.oracle.ledger.official_capture_service import OfficialPredictionCaptureService
from src.oracle.ledger.official_clv_service import OfficialClvService
from src.oracle.ledger.official_performance_service import OfficialPerformanceService
from src.repository.match_repository import MatchRepository
from src.repository.odds_snapshot_repository import OddsSnapshotRepository
from src.repository.prediction_ledger_repository import PredictionLedgerRepository
from src.service_ia.model.match import Base, Match, OddsSnapshot


class _Registry:
    def list_markets(self):
        return ["under_over_2_5"]

    def get_production(self, market):
        return None


class _Snapshots:
    def resolve_predictions(self, **_kwargs):
        return {
            "under_over_2_5": {
                "prediction": 1,
                "probability": 0.75,
                "model_name": "test_champion",
                "run_id": "run-e2e",
            }
        }


def _odd(match_id, fixture_id, bookmaker, outcome, odd, captured_at):
    return OddsSnapshot(
        id_snapshot=str(uuid.uuid4()),
        id_match=match_id,
        fixture_id=fixture_id,
        bookmaker=bookmaker,
        market="under_over_2_5",
        period="full_time",
        line="2.5",
        outcome=outcome,
        odd=odd,
        captured_at=captured_at,
        source="e2e_test",
    )


def test_official_capture_settlement_roi_and_clv_end_to_end():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    kickoff = now + timedelta(minutes=30)
    fixture_id = 990001
    match_id = str(uuid.uuid4())

    session = session_factory()
    match = Match(
        id_match_fk=match_id,
        id_fixture=fixture_id,
        name_home="Home",
        name_away="Away",
        current_league=39,
        status="NS",
        date_match=kickoff.isoformat(),
    )
    session.add(match)
    for bookmaker in ("book-a", "book-b"):
        session.add(_odd(match_id, fixture_id, bookmaker, "Over 2.5", 2.0, now - timedelta(minutes=5)))
        session.add(_odd(match_id, fixture_id, bookmaker, "Under 2.5", 1.9, now - timedelta(minutes=5)))
    session.commit()
    session.refresh(match)

    match_repo = MatchRepository()
    match_repo.session.close()
    match_repo.session = session
    ledger_repo = PredictionLedgerRepository()
    ledger = PredictionLedgerService(repo=ledger_repo, match_repo=match_repo)

    with (
        mock.patch("src.repository.prediction_ledger_repository.SessionLocal", new=session_factory),
        mock.patch("src.repository.odds_snapshot_repository.SessionLocal", new=session_factory),
    ):
        capture = OfficialPredictionCaptureService(
            ledger_service=ledger,
            snapshot_service=_Snapshots(),
            dashboard_service=DashboardService(),
            registry=_Registry(),
        )
        first = capture.capture(now=now, cutoff_minutes=60, matches=[match])
        second = capture.capture(now=now, cutoff_minutes=60, matches=[match])

        assert first["plays_created"] == 1
        assert second["plays_created"] == 0
        assert second["duplicates"] == 1
        rows = ledger_repo.list_all(cohort="official_paper")
        assert len(rows) == 1

        match.status = "FT"
        match.score_home = 2
        match.score_away = 1
        session.commit()
        settlement = ledger.settle_pending(before=kickoff + timedelta(hours=3))
        assert settlement["ledger_settled_win"] == 1
        assert ledger.settle_pending(before=kickoff + timedelta(hours=3))["ledger_candidates"] == 0

        for bookmaker, over, under in (
            ("book-a", 1.80, 2.10),
            ("book-b", 1.82, 2.08),
        ):
            session.add(_odd(match_id, fixture_id, bookmaker, "Over 2.5", over, kickoff - timedelta(minutes=1)))
            session.add(_odd(match_id, fixture_id, bookmaker, "Under 2.5", under, kickoff - timedelta(minutes=1)))
        session.commit()

        performance = OfficialPerformanceService(repo=ledger_repo).report()
        clv = OfficialClvService(
            ledger_repo=ledger_repo,
            snapshot_repo=OddsSnapshotRepository(),
        ).report()

    assert performance["overall"]["wins"] == 1
    assert performance["overall"]["total_profit"] == 1.0
    assert performance["overall"]["roi"] == 1.0
    assert clv["overall"]["available_records"] == 1
    assert clv["overall"]["coverage"] == 1.0
    assert clv["overall"]["avg_clv_odd_pct"] > 0
