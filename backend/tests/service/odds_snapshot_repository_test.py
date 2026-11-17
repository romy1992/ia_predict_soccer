from __future__ import annotations

from datetime import datetime, timezone

from src.repository.odds_snapshot_repository import OddsSnapshotRepository
from src.service_ia.model.match import OddsSnapshot

CAPTURED = datetime(2026, 11, 17, 18, 0, tzinfo=timezone.utc)


def _snapshot(id_snapshot: str, odd: float) -> OddsSnapshot:
    return OddsSnapshot(
        id_snapshot=id_snapshot, id_match="m1", fixture_id=1, bookmaker="b", market="h2h",
        period="full_time", line=None, outcome="Home", odd=odd, captured_at=CAPTURED, source="api_sports",
    )


def test_upsert_rows_ordinate_per_chiave_e_senza_duplicati():
    """Due scritture parallele (import e cattura delle quote di chiusura)
    devono bloccare le righe nello stesso ordine: righe ordinate per PK, una
    sola riga per PK con l'ultimo valore ricevuto."""
    rows = OddsSnapshotRepository._upsert_rows(
        [_snapshot("c", 1.5), _snapshot("a", 2.0), _snapshot("b", 3.0), _snapshot("a", 2.1)]
    )

    assert [row["id_snapshot"] for row in rows] == ["a", "b", "c"]
    assert rows[0]["odd"] == 2.1
    assert rows[0]["captured_at"] == CAPTURED


def test_latest_captured_at_per_fixture():
    from datetime import timedelta
    from unittest import mock

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from src.service_ia.model.match import Base

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    with session_factory() as session:
        older = _snapshot("a", 2.0)
        newer = _snapshot("b", 2.1)
        newer.captured_at = CAPTURED + timedelta(minutes=5)
        other = _snapshot("c", 1.8)
        other.fixture_id = 2
        other.captured_at = CAPTURED + timedelta(hours=1)
        older.id_match = newer.id_match = other.id_match = None
        session.add_all([older, newer, other])
        session.commit()

    with mock.patch("src.repository.odds_snapshot_repository.SessionLocal", new=session_factory):
        repo = OddsSnapshotRepository()
        latest = repo.latest_captured_at(1)
        missing = repo.latest_captured_at(99)

    assert latest.replace(tzinfo=timezone.utc) == CAPTURED + timedelta(minutes=5)
    assert missing is None
