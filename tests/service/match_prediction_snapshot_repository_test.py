import unittest
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.repository.match_prediction_snapshot_repository import MatchPredictionSnapshotRepository
from src.service_ia.model.match import Base, MatchPredictionSnapshot


def _make_session_factory():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


class TestMatchPredictionSnapshotRepository(unittest.TestCase):
    def setUp(self):
        self.session_factory = _make_session_factory()
        self._patch = mock.patch(
            "src.repository.match_prediction_snapshot_repository.SessionLocal", new=self.session_factory
        )
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.repo = MatchPredictionSnapshotRepository()

    def _snapshot(self, fixture_id=1, market="h2h", prediction=1, probability=0.6, fingerprint="fp1", run_id="run1"):
        return MatchPredictionSnapshot(
            fixture_id=fixture_id,
            market=market,
            prediction=prediction,
            probability=probability,
            model_name="logistic",
            model_run_id=run_id,
            feature_fingerprint=fingerprint,
        )

    def test_get_latest_returns_none_when_nothing_saved(self):
        self.assertIsNone(self.repo.get_latest(fixture_id=1, market="h2h"))

    def test_save_then_get_latest_roundtrip(self):
        self.repo.save(self._snapshot())
        row = self.repo.get_latest(fixture_id=1, market="h2h")
        self.assertIsNotNone(row)
        self.assertEqual(row.prediction, 1)
        self.assertAlmostEqual(row.probability, 0.6)
        self.assertEqual(row.feature_fingerprint, "fp1")

    def test_get_latest_returns_most_recent_row_not_first(self):
        self.repo.save(self._snapshot(fingerprint="fp_old", probability=0.5))
        self.repo.save(self._snapshot(fingerprint="fp_new", probability=0.9))

        row = self.repo.get_latest(fixture_id=1, market="h2h")
        self.assertEqual(row.feature_fingerprint, "fp_new")
        self.assertAlmostEqual(row.probability, 0.9)

    def test_get_latest_is_scoped_to_market(self):
        self.repo.save(self._snapshot(market="h2h", fingerprint="fp_h2h"))
        self.repo.save(self._snapshot(market="under_over_2_5", fingerprint="fp_ou"))

        h2h = self.repo.get_latest(fixture_id=1, market="h2h")
        ou = self.repo.get_latest(fixture_id=1, market="under_over_2_5")
        self.assertEqual(h2h.feature_fingerprint, "fp_h2h")
        self.assertEqual(ou.feature_fingerprint, "fp_ou")

    def test_list_for_fixture_preserves_full_history(self):
        self.repo.save(self._snapshot(fingerprint="fp_1"))
        self.repo.save(self._snapshot(fingerprint="fp_2"))
        self.repo.save(self._snapshot(fingerprint="fp_3"))

        rows = self.repo.list_for_fixture(fixture_id=1, market="h2h")
        self.assertEqual([r.feature_fingerprint for r in rows], ["fp_1", "fp_2", "fp_3"])

    def test_get_latest_bulk_returns_latest_per_fixture_and_market(self):
        self.repo.save(self._snapshot(fixture_id=1, market="h2h", fingerprint="fp_old"))
        self.repo.save(self._snapshot(fixture_id=1, market="h2h", fingerprint="fp_new"))
        self.repo.save(self._snapshot(fixture_id=2, market="goal_no_goal", fingerprint="fp_gg"))

        latest = self.repo.get_latest_bulk(fixture_ids=[1, 2])
        self.assertEqual(latest[(1, "h2h")].feature_fingerprint, "fp_new")
        self.assertEqual(latest[(2, "goal_no_goal")].feature_fingerprint, "fp_gg")
        self.assertEqual(len(latest), 2)

    def test_get_latest_bulk_empty_input_returns_empty_dict(self):
        self.assertEqual(self.repo.get_latest_bulk(fixture_ids=[]), {})


if __name__ == "__main__":
    unittest.main()
