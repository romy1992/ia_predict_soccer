"""Test per OPS-03 (Monitoring performance modello e data drift) —
orchestrazione DB-aware (`src/ml/monitoring/monitoring_service.py`).

Stesso pattern di `tests/service/prediction_ledger_test.py`: SQLite
in-memory (StaticPool) con `SessionLocal` patchato nel modulo del
repository - nessun mock sulla query, esercita DAVVERO SQLAlchemy.
`ModelRegistry`/`PredictionLogger`/`FilterMarketService` sono invece
sostituiti con semplici fake (duck typing) iniettati via costruttore,
perche' la feature coverage richiederebbe fixture/modelli reali su disco
non pertinenti a questo test (gia' coperti da `model_registry_test.py`/
`filter_market_service_test.py`).
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.jobs.job_history import JobHistory
from src.ml.monitoring.monitoring_policy import MonitoringThresholds
from src.ml.monitoring.monitoring_service import MonitoringService
from src.repository.prediction_ledger_repository import PredictionLedgerRepository
from src.service_ia.model.match import Base, PredictionLedger


def _make_in_memory_engine():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return engine


class _FakeRegistry:
    """Fake `ModelRegistry`: ritorna un run fisso con `feature_names` note
    per il mercato atteso, `None` altrimenti (stesso comportamento del
    `ModelRegistry` reale quando nessun run e' registrato)."""

    def __init__(self, market: str, feature_names: list[str]):
        self._market = market
        self._feature_names = feature_names

    def get_production(self, market):
        if market != self._market:
            return None
        return {"run_id": "run_prod_1", "current_stage": "production", "feature_names": self._feature_names}

    def get_latest(self, market):
        return self.get_production(market)


class _FakeRegistryEmpty:
    def get_production(self, market):
        return None

    def get_latest(self, market):
        return None


class _FakePredictionLogger:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def tail(self, limit=100, market=None):
        filtered = [row for row in self._rows if market is None or row.get("market") == market]
        return filtered[-limit:]


class _FakeFilterService:
    """Ritorna un frame FISSO per fixture note (con solo alcune delle
    feature attese come colonne, per simulare feature strutturalmente
    sparite), `None` per fixture sconosciute (match non trovato)."""

    def __init__(self, frames_by_fixture: dict[int, pd.DataFrame]):
        self._frames = frames_by_fixture

    def build_prediction_frame(self, market, fixture_id):
        frame = self._frames.get(fixture_id)
        if frame is None:
            return None
        return frame


class TestPredictionVolumeReport(unittest.TestCase):
    def setUp(self):
        self.engine = _make_in_memory_engine()
        self.session_factory = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        self._patch = mock.patch("src.repository.prediction_ledger_repository.SessionLocal", new=self.session_factory)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.repo = PredictionLedgerRepository()
        self.service = MonitoringService(
            ledger_repo=self.repo,
            job_history=JobHistory(path=self._tmp_job_history_path()),
            registry=_FakeRegistryEmpty(),
            prediction_logger=_FakePredictionLogger([]),
        )

    def _tmp_job_history_path(self):
        import tempfile
        return tempfile._get_default_tempdir() + f"/jobs_history_test_{id(self)}.jsonl"

    def _add_row(self, *, market="h2h", created_at, is_settled=False, p_model=None, won=None, odd=None, decision="PLAY"):
        row = PredictionLedger(
            fixture_id=1,
            market=market,
            outcome="Home",
            decision=decision,
            stake=1.0,
            p_model=p_model,
            odd=odd,
            created_at=created_at,
            is_settled=is_settled,
            won=won,
        )
        self.repo.save(row)

    def test_volume_counts_only_within_window_with_daily_buckets(self):
        now = datetime.now(timezone.utc)
        self._add_row(created_at=now - timedelta(days=1))
        self._add_row(created_at=now - timedelta(days=1))
        self._add_row(created_at=now - timedelta(days=40))  # fuori dalla finestra di 30gg

        report = self.service.prediction_volume_report(market="h2h", days=30)

        self.assertEqual(report["total"], 2)
        self.assertEqual(len(report["series"]), 30)
        # L'ultimo bucket della serie e' oggi, il penultimo e' ieri (con le 2 righe).
        yesterday_key = (now - timedelta(days=1)).date().isoformat()
        bucket = next(b for b in report["series"] if b["date"] == yesterday_key)
        self.assertEqual(bucket["count"], 2)

    def test_volume_zero_when_no_rows(self):
        report = self.service.prediction_volume_report(market="h2h", days=7)
        self.assertEqual(report["total"], 0)
        self.assertTrue(all(b["count"] == 0 for b in report["series"]))


class TestCalibrationDriftReport(unittest.TestCase):
    def setUp(self):
        self.engine = _make_in_memory_engine()
        self.session_factory = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        self._patch = mock.patch("src.repository.prediction_ledger_repository.SessionLocal", new=self.session_factory)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.repo = PredictionLedgerRepository()
        self.service = MonitoringService(
            ledger_repo=self.repo,
            job_history=JobHistory(path=self._tmp_path()),
            registry=_FakeRegistryEmpty(),
            prediction_logger=_FakePredictionLogger([]),
        )

    def _tmp_path(self):
        import tempfile
        return tempfile._get_default_tempdir() + f"/jobs_history_test_{id(self)}.jsonl"

    def _add_settled_row(self, *, created_at, p_model, won, market="h2h"):
        row = PredictionLedger(
            fixture_id=1,
            market=market,
            outcome="Home",
            decision="PLAY",
            stake=1.0,
            p_model=p_model,
            odd=2.0,
            created_at=created_at,
            is_settled=True,
            won=won,
            pnl=1.0 if won else -1.0,
            settlement_status="settled",
        )
        self.repo.save(row)

    def test_none_when_no_settled_rows(self):
        self.assertIsNone(self.service.calibration_drift_report(market="h2h"))

    def test_unavailable_when_not_enough_samples(self):
        now = datetime.now(timezone.utc)
        for _ in range(3):
            self._add_settled_row(created_at=now - timedelta(days=1), p_model=0.6, won=True)

        result = self.service.calibration_drift_report(
            market="h2h", thresholds=MonitoringThresholds(min_samples_for_calibration=20)
        )
        self.assertFalse(result["available"])
        self.assertIsNotNone(result["reason"])
        self.assertIsNone(result["drift"])

    def test_available_with_enough_samples_and_computes_drift(self):
        now = datetime.now(timezone.utc)
        thresholds = MonitoringThresholds(min_samples_for_calibration=10)

        # Baseline (>30gg fa): calibrazione buona (p_model coerente col won).
        for _ in range(15):
            self._add_settled_row(created_at=now - timedelta(days=60), p_model=0.9, won=True)
        for _ in range(15):
            self._add_settled_row(created_at=now - timedelta(days=60), p_model=0.1, won=False)

        # Recente (<30gg): calibrazione scadente (probabilita' alta ma spesso persa).
        for _ in range(15):
            self._add_settled_row(created_at=now - timedelta(days=1), p_model=0.9, won=False)
        for _ in range(15):
            self._add_settled_row(created_at=now - timedelta(days=1), p_model=0.1, won=True)

        result = self.service.calibration_drift_report(market="h2h", recent_days=30, thresholds=thresholds)

        self.assertTrue(result["available"])
        self.assertEqual(result["recent_metrics"]["sample_size"], 30)
        self.assertEqual(result["baseline_metrics"]["sample_size"], 30)
        # La finestra recente e' deliberatamente scalibrata: ECE peggiore -> drift positivo.
        self.assertGreater(result["drift"]["ece_drift"], 0.0)


class TestRoiRollingReport(unittest.TestCase):
    def setUp(self):
        self.engine = _make_in_memory_engine()
        self.session_factory = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        self._patch = mock.patch("src.repository.prediction_ledger_repository.SessionLocal", new=self.session_factory)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.repo = PredictionLedgerRepository()
        self.service = MonitoringService(
            ledger_repo=self.repo,
            job_history=JobHistory(path=self._tmp_path()),
            registry=_FakeRegistryEmpty(),
            prediction_logger=_FakePredictionLogger([]),
        )

    def _tmp_path(self):
        import tempfile
        return tempfile._get_default_tempdir() + f"/jobs_history_test_{id(self)}.jsonl"

    def test_diagnostic_only_flag_and_window_structure(self):
        now = datetime.now(timezone.utc)
        row = PredictionLedger(
            fixture_id=1, market="h2h", outcome="Home", decision="PLAY", stake=1.0,
            p_model=0.7, odd=2.0, created_at=now - timedelta(days=2),
            is_settled=True, won=True, pnl=1.0, settlement_status="settled",
        )
        self.repo.save(row)

        report = self.service.roi_rolling_report(market="h2h", windows_days=(7, 30))

        self.assertTrue(report["diagnostic_only"])
        self.assertIn("7", report["windows"])
        self.assertIn("30", report["windows"])
        self.assertEqual(report["windows"]["7"]["overall"]["bets"], 1)
        self.assertEqual(report["windows"]["7"]["overall"]["wins"], 1)


class TestFeatureCoverageReport(unittest.TestCase):
    def test_none_when_no_model_registered(self):
        service = MonitoringService(
            ledger_repo=mock.Mock(),
            job_history=mock.Mock(),
            registry=_FakeRegistryEmpty(),
            prediction_logger=_FakePredictionLogger([]),
            filter_service=_FakeFilterService({}),
        )
        self.assertIsNone(service.feature_coverage_report(market="h2h"))

    def test_computes_column_presence_from_sampled_fixtures(self):
        registry = _FakeRegistry(market="h2h", feature_names=["feat_a", "feat_b", "feat_missing"])
        logger = _FakePredictionLogger(
            [
                {"market": "h2h", "fixture_id": 1},
                {"market": "h2h", "fixture_id": 2},
                {"market": "h2h", "fixture_id": 3},  # nessun frame disponibile (match non trovato)
            ]
        )
        frames = {
            1: pd.DataFrame([{"feat_a": 1.0, "feat_b": 2.0}]),
            2: pd.DataFrame([{"feat_a": 1.0}]),  # feat_b assente in questo frame
        }
        filter_service = _FakeFilterService(frames)

        service = MonitoringService(
            ledger_repo=mock.Mock(),
            job_history=mock.Mock(),
            registry=registry,
            prediction_logger=logger,
            filter_service=filter_service,
        )

        coverage = service.feature_coverage_report(market="h2h", sample_limit=10)

        self.assertEqual(coverage["market"], "h2h")
        self.assertEqual(coverage["requested_fixtures"], 3)
        self.assertEqual(coverage["sampled_frames"], 2)  # fixture 3 non ha prodotto un frame
        per_feature = {row["feature"]: row for row in coverage["per_feature"]}
        self.assertEqual(per_feature["feat_a"]["column_presence_ratio"], 1.0)
        self.assertEqual(per_feature["feat_b"]["column_presence_ratio"], 0.5)
        self.assertEqual(per_feature["feat_missing"]["column_presence_ratio"], 0.0)
        self.assertEqual(coverage["fully_missing_features"], ["feat_missing"])


class TestAlertsAndFullReport(unittest.TestCase):
    def setUp(self):
        self.engine = _make_in_memory_engine()
        self.session_factory = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        self._patch = mock.patch("src.repository.prediction_ledger_repository.SessionLocal", new=self.session_factory)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.repo = PredictionLedgerRepository()

    def test_alerts_report_structure_without_market(self):
        service = MonitoringService(
            ledger_repo=self.repo,
            job_history=JobHistory(path=self._tmp_path()),
            registry=_FakeRegistryEmpty(),
            prediction_logger=_FakePredictionLogger([]),
        )
        payload = service.alerts_report(market=None)

        self.assertIn("generated_at", payload)
        self.assertIn("thresholds_version", payload)
        self.assertIsInstance(payload["alerts"], list)
        # Nessuna prediction salvata -> volume 0 -> alert di basso volume (soglia default=1).
        codes = [a["code"] for a in payload["alerts"]]
        self.assertIn("low_prediction_volume", codes)

    def test_full_report_aggregates_all_sections(self):
        service = MonitoringService(
            ledger_repo=self.repo,
            job_history=JobHistory(path=self._tmp_path()),
            registry=_FakeRegistryEmpty(),
            prediction_logger=_FakePredictionLogger([]),
        )
        payload = service.full_report(market=None)

        for key in ("generated_at", "market", "thresholds_version", "prediction_volume", "roi_rolling", "alerts"):
            self.assertIn(key, payload)
        # Senza market, calibration/feature coverage non sono calcolabili.
        self.assertIsNone(payload["calibration_drift"])
        self.assertIsNone(payload["feature_coverage"])

    def _tmp_path(self):
        import tempfile
        return tempfile._get_default_tempdir() + f"/jobs_history_test_{id(self)}.jsonl"


class TestRecentFailedJobsCount(unittest.TestCase):
    def test_counts_only_failed_jobs_within_window(self):
        import tempfile
        path = tempfile._get_default_tempdir() + f"/jobs_history_failcount_{id(self)}.jsonl"
        history = JobHistory(path=path)

        now = datetime.now(timezone.utc)
        recent_failed = history.create_job(job_type="retrain", status="failed", job_id="j1")
        history.update_job("j1", status="failed")
        # Forziamo timestamp esplicito nel passato remoto per un secondo job.
        old_failed = history.create_job(job_type="retrain", status="failed", job_id="j2")
        rows = history._read_rows()
        for row in rows:
            if row["job_id"] == "j2":
                row["timestamp"] = (now - timedelta(days=30)).isoformat()
        history._write_rows(rows)

        service = MonitoringService(
            ledger_repo=mock.Mock(),
            job_history=history,
            registry=_FakeRegistryEmpty(),
            prediction_logger=_FakePredictionLogger([]),
        )
        count = service._recent_failed_jobs_count(since_days=7)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()

