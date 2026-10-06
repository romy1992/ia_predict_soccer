import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.repository.betting_slip_proposal_repository import BettingSlipProposalRepository
from src.service_ia.model.match import Base, BettingSlipProposalSnapshot


def _naive(value):
    """SQLite non conserva il timezone sui DateTime(timezone=True) - un
    round-trip ritorna un datetime naive con lo stesso istante UTC. Mai un
    problema reale (Postgres, usato in produzione, lo conserva per
    davvero); qui serve solo per confronti robusti nel test."""
    return value.replace(tzinfo=None) if value and value.tzinfo else value


def _make_session_factory():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


def _snapshot(
    *,
    snapshot_key,
    logical_slip_id,
    reference_date="2026-10-06",
    profile="SAFE",
    last_confirmed_at=None,
    generated_at=None,
):
    return BettingSlipProposalSnapshot(
        snapshot_key=snapshot_key,
        logical_slip_id=logical_slip_id,
        reference_date=reference_date,
        profile=profile,
        situation="PLAY",
        event_count=2,
        combined_odd=2.0,
        policy_version="policy-v1",
        correlation_version="correlation-v1",
        payload={"slip_id": logical_slip_id},
        generated_at=generated_at or datetime.now(timezone.utc),
        last_confirmed_at=last_confirmed_at,
    )


class TestBettingSlipProposalRepository(unittest.TestCase):
    def setUp(self):
        self.session_factory = _make_session_factory()
        self._patch = mock.patch(
            "src.repository.betting_slip_proposal_repository.SessionLocal", new=self.session_factory
        )
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.repo = BettingSlipProposalRepository()

    def test_save_revision_sets_last_confirmed_at_on_insert(self):
        t1 = datetime(2026, 10, 6, 10, 0, tzinfo=timezone.utc)
        saved, created = self.repo.save_revision(
            _snapshot(snapshot_key="k1", logical_slip_id="lineage-1", last_confirmed_at=t1)
        )
        self.assertTrue(created)
        self.assertEqual(_naive(saved.last_confirmed_at), _naive(t1))

    def test_save_revision_touches_last_confirmed_at_on_unchanged_payload_without_new_row(self):
        t1 = datetime(2026, 10, 6, 10, 0, tzinfo=timezone.utc)
        t2 = t1 + timedelta(minutes=30)
        self.repo.save_revision(_snapshot(snapshot_key="k1", logical_slip_id="lineage-1", last_confirmed_at=t1))

        saved, created = self.repo.save_revision(
            _snapshot(snapshot_key="k1", logical_slip_id="lineage-1", last_confirmed_at=t2)
        )

        self.assertFalse(created, "stesso snapshot_key: nessuna nuova riga")
        self.assertEqual(_naive(saved.last_confirmed_at), _naive(t2))
        all_rows = self.repo.list_all(reference_date="2026-10-06")
        self.assertEqual(len(all_rows), 1, "la ri-conferma non deve creare una seconda riga")

    def test_save_revision_never_touches_is_latest_or_generated_at_on_unchanged_payload(self):
        t1 = datetime(2026, 10, 6, 10, 0, tzinfo=timezone.utc)
        t2 = t1 + timedelta(minutes=30)
        original_generated_at = datetime(2026, 10, 6, 9, 0, tzinfo=timezone.utc)
        self.repo.save_revision(
            _snapshot(
                snapshot_key="k1",
                logical_slip_id="lineage-1",
                last_confirmed_at=t1,
                generated_at=original_generated_at,
            )
        )

        saved, _ = self.repo.save_revision(
            _snapshot(
                snapshot_key="k1",
                logical_slip_id="lineage-1",
                last_confirmed_at=t2,
                generated_at=t2,  # il chiamante lo passerebbe comunque, ma save_revision deve ignorarlo
            )
        )

        self.assertTrue(saved.is_latest)
        self.assertEqual(_naive(saved.generated_at), _naive(original_generated_at))

    def test_list_current_returns_only_rows_confirmed_by_latest_run(self):
        t1 = datetime(2026, 10, 6, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 10, 6, 10, 30, tzinfo=timezone.utc)
        # lineage-1: confermata solo al giro 1 (poi mai piu' rigenerata)
        self.repo.save_revision(_snapshot(snapshot_key="k1", logical_slip_id="lineage-1", last_confirmed_at=t1))
        # lineage-2: confermata al giro 1 E riconfermata al giro 2 (payload invariato)
        self.repo.save_revision(_snapshot(snapshot_key="k2", logical_slip_id="lineage-2", last_confirmed_at=t1))
        self.repo.save_revision(_snapshot(snapshot_key="k2", logical_slip_id="lineage-2", last_confirmed_at=t2))

        current = self.repo.list_current(reference_date="2026-10-06")

        self.assertEqual({row.logical_slip_id for row in current}, {"lineage-2"})

    def test_list_current_empty_when_no_rows_for_date(self):
        self.assertEqual(self.repo.list_current(reference_date="2026-10-06"), [])

    def test_list_current_ignores_rows_not_latest(self):
        t1 = datetime(2026, 10, 6, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 10, 6, 10, 30, tzinfo=timezone.utc)
        # Stesso logical_slip_id, payload DIVERSO (snapshot_key diverso):
        # la prima riga viene superseduta (is_latest=False), mai nella
        # vista corrente anche se il suo last_confirmed_at fosse piu' alto.
        self.repo.save_revision(_snapshot(snapshot_key="k1", logical_slip_id="lineage-1", last_confirmed_at=t2))
        self.repo.save_revision(_snapshot(snapshot_key="k2", logical_slip_id="lineage-1", last_confirmed_at=t1))

        current = self.repo.list_current(reference_date="2026-10-06")

        self.assertEqual(len(current), 1)
        self.assertEqual(current[0].snapshot_key, "k2")
        self.assertTrue(current[0].is_latest)


if __name__ == "__main__":
    unittest.main()
