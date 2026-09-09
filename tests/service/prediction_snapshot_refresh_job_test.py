import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.jobs import scheduler as scheduler_module
from src.jobs.job_history import JobHistory as RealJobHistory
from src.service_ia.model.match import Base, Match


def _make_session_factory():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


class _FakeSnapshotService:
    def __init__(self, raise_for_fixture=None):
        self.calls = []
        self.raise_for_fixture = raise_for_fixture

    def resolve_predictions(self, fixture_id, markets, db_match=None, status=None):
        self.calls.append(fixture_id)
        if self.raise_for_fixture == fixture_id:
            raise RuntimeError("boom")
        return {m: {"prediction": 1, "probability": 0.5} for m in markets}


class TestRunPredictionSnapshotRefresh(unittest.TestCase):
    """`run_prediction_snapshot_refresh` (2026-09-09): a differenza degli
    altri job body in `scheduler.py` (thin wrapper attorno a collaboratori
    gia' testati altrove), qui la finestra di query + l'isolamento errori
    per-fixture sono logica NUOVA specifica di questo task - vale la pena
    di un test dedicato, a differenza degli altri `run_manual_*` (mai
    testati direttamente in questo file, solo tramite identita' della
    funzione target in `TestJobTargetsAreCorrectAndIndependent`)."""

    def setUp(self):
        self.session_factory = _make_session_factory()
        self._session_patch = mock.patch.object(scheduler_module, "SessionLocal", self.session_factory)
        self._session_patch.start()
        self.addCleanup(self._session_patch.stop)

        self._registry_patch = mock.patch.object(
            scheduler_module, "ModelRegistry", lambda: mock.Mock(list_markets=lambda: ["h2h"])
        )
        self._registry_patch.start()
        self.addCleanup(self._registry_patch.stop)

        self._tmp_history_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_history_dir.cleanup)
        history_path = f"{self._tmp_history_dir.name}/jobs_history.jsonl"

        # Sottoclasse (non una lambda): il corpo del job chiama anche
        # `JobHistory._now_iso()` come staticmethod SULLA CLASSE - una
        # lambda non espone quell'attributo, una vera sottoclasse si'
        # (ereditato da `RealJobHistory`).
        class _FixedPathJobHistory(RealJobHistory):
            def __init__(self):
                super().__init__(path=history_path)

        self._history_patch = mock.patch.object(scheduler_module, "JobHistory", _FixedPathJobHistory)
        self._history_patch.start()
        self.addCleanup(self._history_patch.stop)

    def _seed_match(self, fixture_id: int, status: str, days_from_today: int):
        target_date = (datetime.now(timezone.utc) + timedelta(days=days_from_today)).date()
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

    def test_only_considers_ns_fixtures_within_window(self):
        self._seed_match(1, status="NS", days_from_today=1)
        self._seed_match(2, status="FT", days_from_today=1)  # finale, escluso
        self._seed_match(3, status="NS", days_from_today=20)  # fuori finestra, escluso

        fake_service = _FakeSnapshotService()
        with mock.patch.object(scheduler_module, "PredictionSnapshotService", lambda: fake_service):
            result = scheduler_module.run_prediction_snapshot_refresh(days_ahead=7)

        self.assertEqual(fake_service.calls, [1])
        self.assertEqual(result["fixtures_considered"], 1)
        self.assertEqual(result["predictions_resolved"], 1)

    def test_isolates_per_fixture_errors_without_failing_the_whole_job(self):
        self._seed_match(1, status="NS", days_from_today=1)
        self._seed_match(2, status="NS", days_from_today=2)

        fake_service = _FakeSnapshotService(raise_for_fixture=1)
        with mock.patch.object(scheduler_module, "PredictionSnapshotService", lambda: fake_service):
            result = scheduler_module.run_prediction_snapshot_refresh(days_ahead=7)

        self.assertEqual(result["fixtures_considered"], 2)
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(result["errors"][0]["fixture_id"], 1)
        self.assertEqual(result["predictions_resolved"], 1)

    def test_no_matching_fixtures_returns_zero_counts(self):
        fake_service = _FakeSnapshotService()
        with mock.patch.object(scheduler_module, "PredictionSnapshotService", lambda: fake_service):
            result = scheduler_module.run_prediction_snapshot_refresh(days_ahead=7)

        self.assertEqual(result["fixtures_considered"], 0)
        self.assertEqual(result["predictions_resolved"], 0)
        self.assertEqual(result["errors"], [])

    def test_defaults_days_ahead_from_cfg_when_not_given(self):
        fake_service = _FakeSnapshotService()
        with mock.patch.object(scheduler_module, "PredictionSnapshotService", lambda: fake_service):
            result = scheduler_module.run_prediction_snapshot_refresh()
        self.assertIn("days_ahead", result)
        self.assertIsInstance(result["days_ahead"], int)


if __name__ == "__main__":
    unittest.main()
