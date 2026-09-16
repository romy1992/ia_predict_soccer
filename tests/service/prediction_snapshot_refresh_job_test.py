import os
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


class _FakeRepo:
    """Stub di `MatchPredictionSnapshotRepository.get_latest_bulk` - simula
    quali (fixture_id, market) hanno gia' una riga salvata, cosi' i test
    possono verificare l'anti-join fatto da `run_prediction_snapshot_refresh`
    PRIMA di richiamare `resolve_predictions` per le fixture concluse."""

    def __init__(self, existing_snapshots=None):
        self.existing_snapshots = existing_snapshots or {}

    def get_latest_bulk(self, fixture_ids):
        return {key: True for key in self.existing_snapshots if key[0] in fixture_ids}


class _FakeSnapshotService:
    def __init__(self, raise_for_fixture=None, existing_snapshots=None):
        self.calls = []
        self.raise_for_fixture = raise_for_fixture
        self.repo = _FakeRepo(existing_snapshots)

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
        os.environ.pop("PREDICTION_REFRESH_STEP_SLEEP", None)
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
        # data FUTURA con status finale (caso sintetico): esclusa sia dalla
        # finestra NS (status sbagliato) sia da quella "appena concluse"
        # (guarda solo all'indietro, mai in avanti).
        self._seed_match(2, status="FT", days_from_today=1)
        self._seed_match(3, status="NS", days_from_today=20)  # fuori finestra, escluso

        fake_service = _FakeSnapshotService()
        with mock.patch.object(scheduler_module, "PredictionSnapshotService", lambda: fake_service):
            result = scheduler_module.run_prediction_snapshot_refresh(days_ahead=7)

        self.assertEqual(fake_service.calls, [1])
        self.assertEqual(result["fixtures_considered"], 1)
        self.assertEqual(result["predictions_resolved"], 1)

    # --- Punto 2/4 (2026-09-10): partite APPENA concluse ---

    def test_recently_finished_fixture_without_snapshot_is_collected(self):
        self._seed_match(10, status="FT", days_from_today=-1)  # ieri, nessuna riga salvata

        fake_service = _FakeSnapshotService()
        with mock.patch.object(scheduler_module, "PredictionSnapshotService", lambda: fake_service):
            result = scheduler_module.run_prediction_snapshot_refresh(days_ahead=7)

        self.assertEqual(fake_service.calls, [10])
        self.assertEqual(result["fixtures_considered"], 1)
        self.assertEqual(result["fixtures_recently_finished"], 1)
        self.assertEqual(result["predictions_resolved"], 1)

    def test_recently_finished_fixture_already_covered_is_skipped(self):
        """Anti-join: una fixture conclusa con GIA' una riga per l'UNICO
        mercato registrato (`h2h`, vedi il mock di `ModelRegistry` in
        `setUp`) non deve mai richiamare `resolve_predictions` - la riga
        congelata basta, nessuna query/inferenza aggiuntiva."""
        self._seed_match(11, status="FT", days_from_today=-1)

        fake_service = _FakeSnapshotService(existing_snapshots={(11, "h2h"): True})
        with mock.patch.object(scheduler_module, "PredictionSnapshotService", lambda: fake_service):
            result = scheduler_module.run_prediction_snapshot_refresh(days_ahead=7)

        self.assertEqual(fake_service.calls, [])
        self.assertEqual(result["fixtures_considered"], 0)
        self.assertEqual(result["fixtures_recently_finished"], 0)

    def test_old_finished_fixture_outside_window_is_not_touched(self):
        """Storico oltre la piccola finestra "recenti": resta compito dello
        script di backfill una tantum (punto 3 del piano), MAI di questo
        job ricorrente."""
        self._seed_match(12, status="FT", days_from_today=-30)

        fake_service = _FakeSnapshotService()
        with mock.patch.object(scheduler_module, "PredictionSnapshotService", lambda: fake_service):
            result = scheduler_module.run_prediction_snapshot_refresh(days_ahead=7)

        self.assertEqual(fake_service.calls, [])
        self.assertEqual(result["fixtures_considered"], 0)

    def test_recently_finished_window_is_configurable(self):
        self._seed_match(13, status="FT", days_from_today=-5)

        fake_service = _FakeSnapshotService()
        with mock.patch.object(scheduler_module, "PredictionSnapshotService", lambda: fake_service):
            result_default_window = scheduler_module.run_prediction_snapshot_refresh(days_ahead=7)
            result_wide_window = scheduler_module.run_prediction_snapshot_refresh(
                days_ahead=7, recently_finished_days=10
            )

        self.assertEqual(result_default_window["fixtures_recently_finished"], 0)
        self.assertEqual(result_wide_window["fixtures_recently_finished"], 1)

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

    # --- `target_date` (2026-09-13): bottone "Ricalcola previsioni del
    # giorno". Sostituisce ENTRAMBE le finestre di default con un solo
    # giorno, qualunque sia lo status delle fixture. ---

    def _today_iso(self) -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _mock_betslip_service(self):
        """`BetslipService` apre una connessione DB reale: qui interessa solo
        il giro delle predizioni, non la generazione delle proposte (coperta
        dai suoi test). Mockarlo tiene questi test deterministici e senza
        dipendenze di rete."""
        service = mock.Mock()
        service.generate_and_snapshot_for_day.return_value = (
            None,
            None,
            {"proposals_seen": 0, "proposals_created": 0, "proposals_unchanged": 0},
        )
        return service

    def test_target_date_covers_every_status_of_that_day(self):
        """Il caso d'uso reale (mercato appena promosso) riguarda tutte le
        partite del giorno guardato in Dashboard, non solo quelle NS: a
        differenza del giro di default qui non si filtra per status."""
        self._seed_match(20, status="NS", days_from_today=0)
        self._seed_match(21, status="FT", days_from_today=0)
        self._seed_match(22, status="1H", days_from_today=0)

        fake_service = _FakeSnapshotService()
        betslip_service = self._mock_betslip_service()
        with mock.patch.object(scheduler_module, "PredictionSnapshotService", lambda: fake_service):
            with mock.patch.object(scheduler_module, "BetslipService", lambda: betslip_service):
                result = scheduler_module.run_prediction_snapshot_refresh(target_date=self._today_iso())

        self.assertEqual(sorted(fake_service.calls), [20, 21, 22])
        self.assertEqual(result["fixtures_considered"], 3)
        self.assertEqual(result["target_date"], self._today_iso())
        self.assertEqual(result["percent"], 100.0)
        self.assertEqual(result["fixtures_done"], 3)
        self.assertEqual(result["fixtures_total"], 3)

    def test_target_date_excludes_other_days_even_inside_default_windows(self):
        """Una fixture NS di domani rientrerebbe nella finestra di default
        `days_ahead`: con `target_date` su oggi non deve essere toccata."""
        self._seed_match(30, status="NS", days_from_today=0)
        self._seed_match(31, status="NS", days_from_today=1)
        self._seed_match(32, status="FT", days_from_today=-1)

        fake_service = _FakeSnapshotService()
        betslip_service = self._mock_betslip_service()
        with mock.patch.object(scheduler_module, "PredictionSnapshotService", lambda: fake_service):
            with mock.patch.object(scheduler_module, "BetslipService", lambda: betslip_service):
                result = scheduler_module.run_prediction_snapshot_refresh(target_date=self._today_iso())

        self.assertEqual(fake_service.calls, [30])
        self.assertEqual(result["fixtures_recently_finished"], 0)

    def test_progress_is_written_during_the_run(self):
        self._seed_match(50, status="NS", days_from_today=0)
        self._seed_match(51, status="NS", days_from_today=0)
        percents = []

        class _Spy(_FakeSnapshotService):
            def resolve_predictions(inner_self, fixture_id, markets, db_match=None, status=None):
                rows = scheduler_module.JobHistory().tail(limit=5, job_type="prediction_snapshot_refresh")
                if rows:
                    percents.append((rows[-1].get("summary") or {}).get("percent"))
                return super().resolve_predictions(fixture_id, markets, db_match=db_match, status=status)

        fake_service = _Spy()
        betslip_service = self._mock_betslip_service()
        with mock.patch.object(scheduler_module, "PredictionSnapshotService", lambda: fake_service):
            with mock.patch.object(scheduler_module, "BetslipService", lambda: betslip_service):
                result = scheduler_module.run_prediction_snapshot_refresh(target_date=self._today_iso())

        self.assertEqual(result["percent"], 100.0)
        self.assertTrue(percents)
        self.assertLess(min(p for p in percents if p is not None), 100.0)

    def test_past_target_date_does_not_attempt_betslip_proposals(self):
        """Le proposte esistono solo da oggi in avanti (il generatore
        rifiuta le giornate passate): ricalcolare un giorno storico non
        deve neppure provarci, altrimenti il report si riempirebbe di
        errori attesi."""
        yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        self._seed_match(40, status="FT", days_from_today=-1)

        fake_service = _FakeSnapshotService()
        betslip_service = mock.Mock()
        with mock.patch.object(scheduler_module, "PredictionSnapshotService", lambda: fake_service):
            with mock.patch.object(scheduler_module, "BetslipService", lambda: betslip_service):
                result = scheduler_module.run_prediction_snapshot_refresh(target_date=yesterday)

        self.assertEqual(fake_service.calls, [40])
        betslip_service.generate_and_snapshot_for_day.assert_not_called()
        self.assertEqual(result["betslip_proposals"]["dates_considered"], 0)

    def test_invalid_target_date_is_rejected(self):
        fake_service = _FakeSnapshotService()
        with mock.patch.object(scheduler_module, "PredictionSnapshotService", lambda: fake_service):
            with self.assertRaises(ValueError):
                scheduler_module.run_prediction_snapshot_refresh(target_date="13-09-2026")


if __name__ == "__main__":
    unittest.main()
