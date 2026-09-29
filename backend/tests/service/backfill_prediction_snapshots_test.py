import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from scripts import backfill_prediction_snapshots as backfill_module
from src.service_ia.model.match import Base, Match


def _make_session_factory():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


class _FakeRepo:
    def __init__(self, existing_snapshots=None):
        self.existing_snapshots = existing_snapshots or {}

    def get_latest_bulk(self, fixture_ids):
        return {key: True for key in self.existing_snapshots if key[0] in fixture_ids}


class _FakeSnapshotService:
    def __init__(self, raise_for_fixture=None, existing_snapshots=None):
        self.calls = []
        self.raise_for_fixture = raise_for_fixture
        self.repo = _FakeRepo(existing_snapshots)

    def resolve_predictions(self, fixture_id, markets, db_match=None, status=None, allow_compute=True):
        self.calls.append(fixture_id)
        if self.raise_for_fixture == fixture_id:
            raise RuntimeError("boom")
        return {m: {"prediction": 1, "probability": 0.5} for m in markets}


class TestBackfillPredictionSnapshots(unittest.TestCase):
    """`run_backfill` (2026-09-10, punto 3/4 di
    PROMPT_fast_historical_predictions.md): stesso pattern di test gia' in
    uso per `run_prediction_snapshot_refresh` (SQLite in-memory + fake
    snapshot service) - qui in piu' si verifica la keyset pagination
    (`_fetch_final_fixture_ids_batch`, mai un OFFSET) attraverso PIU' batch
    consecutivi."""

    def setUp(self):
        self.session_factory = _make_session_factory()
        self._session_patch = mock.patch.object(backfill_module, "SessionLocal", self.session_factory)
        self._session_patch.start()
        self.addCleanup(self._session_patch.stop)

        self._registry_patch = mock.patch.object(
            backfill_module, "ModelRegistry", lambda: mock.Mock(list_markets=lambda: ["h2h"])
        )
        self._registry_patch.start()
        self.addCleanup(self._registry_patch.stop)

    def _seed_match(self, fixture_id: int, status: str, days_ago: int = 1):
        target_date = (datetime.now(timezone.utc) - timedelta(days=days_ago)).date()
        with self.session_factory() as session:
            session.add(
                Match(
                    id_match_fk=str(uuid.uuid4()),
                    id_fixture=fixture_id,
                    status=status,
                    date_match=f"{target_date.isoformat()}T18:00:00+00:00",
                    season=2026,
                    current_league=39,
                )
            )
            session.commit()

    def test_collects_final_fixtures_without_snapshot(self):
        self._seed_match(1, status="FT")
        self._seed_match(2, status="NS")  # non conclusa, ignorata

        fake_service = _FakeSnapshotService()
        with mock.patch.object(backfill_module, "PredictionSnapshotService", lambda: fake_service):
            summary = backfill_module.run_backfill()

        self.assertEqual(fake_service.calls, [1])
        self.assertEqual(summary["fixtures_needing_backfill"], 1)
        self.assertEqual(summary["fixtures_processed"], 1)
        self.assertEqual(summary["predictions_resolved"], 1)
        self.assertEqual(summary["errors"], [])

    def test_skips_fixtures_already_fully_covered(self):
        self._seed_match(1, status="FT")

        fake_service = _FakeSnapshotService(existing_snapshots={(1, "h2h"): True})
        with mock.patch.object(backfill_module, "PredictionSnapshotService", lambda: fake_service):
            summary = backfill_module.run_backfill()

        self.assertEqual(fake_service.calls, [])
        self.assertEqual(summary["fixtures_needing_backfill"], 0)
        self.assertEqual(summary["fixtures_processed"], 0)

    def test_isolates_per_fixture_errors_without_stopping(self):
        self._seed_match(1, status="FT")
        self._seed_match(2, status="FT")

        fake_service = _FakeSnapshotService(raise_for_fixture=1)
        with mock.patch.object(backfill_module, "PredictionSnapshotService", lambda: fake_service):
            summary = backfill_module.run_backfill()

        self.assertEqual(set(fake_service.calls), {1, 2})
        self.assertEqual(len(summary["errors"]), 1)
        self.assertEqual(summary["errors"][0]["fixture_id"], 1)
        self.assertEqual(summary["predictions_resolved"], 1)

    def test_paginates_across_multiple_batches(self):
        """Keyset pagination: con `batch_size=1` e 3 fixture concluse, TUTTE
        e 3 devono essere raccolte (mai fermarsi al primo batch, mai un
        loop infinito) - verifica che `after_fixture_id` avanzi
        correttamente ad ogni giro."""
        self._seed_match(1, status="FT")
        self._seed_match(2, status="FT")
        self._seed_match(3, status="FT")

        fake_service = _FakeSnapshotService()
        with mock.patch.object(backfill_module, "PredictionSnapshotService", lambda: fake_service):
            summary = backfill_module.run_backfill(batch_size=1)

        self.assertEqual(sorted(fake_service.calls), [1, 2, 3])
        self.assertEqual(summary["fixtures_scanned"], 3)
        self.assertEqual(summary["fixtures_processed"], 3)

    def test_limit_stops_early_without_processing_the_rest(self):
        self._seed_match(1, status="FT")
        self._seed_match(2, status="FT")
        self._seed_match(3, status="FT")

        fake_service = _FakeSnapshotService()
        with mock.patch.object(backfill_module, "PredictionSnapshotService", lambda: fake_service):
            summary = backfill_module.run_backfill(batch_size=1, limit=2)

        self.assertEqual(summary["fixtures_processed"], 2)
        self.assertEqual(len(fake_service.calls), 2)

    def test_no_final_fixtures_returns_zero_counts(self):
        self._seed_match(1, status="NS")

        fake_service = _FakeSnapshotService()
        with mock.patch.object(backfill_module, "PredictionSnapshotService", lambda: fake_service):
            summary = backfill_module.run_backfill()

        self.assertEqual(summary["fixtures_scanned"], 0)
        self.assertEqual(summary["fixtures_processed"], 0)
        self.assertEqual(summary["errors"], [])


if __name__ == "__main__":
    unittest.main()
