import functools
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from apscheduler.triggers.interval import IntervalTrigger

from src.jobs import scheduler as scheduler_module
from src.service_ia.config.app_config import AppConfig


def _target_func(job):
    """I job sono registrati come `functools.partial(_run_if_due, job_id,
    func, cfg=cfg, ...)` (vedi `_run_if_due` in `src/jobs/scheduler.py`,
    controllo enabled/disabled + schedule da Impostazioni): questo helper
    estrae la funzione VERA da confrontare nei test, a prescindere dal
    wrapping."""
    func = job.func
    if isinstance(func, functools.partial):
        return func.args[-1]
    return func


def _cfg(**overrides) -> AppConfig:
    base = dict(
        leagues=[39],
        seasons=[2026],
        data_sync_interval_minutes=30,
        settlement_interval_minutes=60,
        future_sync_hour=4,
        future_sync_minute=30,
        daily_refresh_hour=5,
        daily_refresh_minute=0,
        daily_refresh_days_ahead=7,
        training_hour=23,
        training_minute=0,
        live_sync_interval_seconds=90,
        live_cache_ttl_seconds=20,
        api_sports_daily_limit=7500,
        database_url="sqlite://",
        database_schema="public",
        data_quality_interval_minutes=60,
    )
    base.update(overrides)
    return AppConfig(**base)


def _fixed_datetime_class(fixed_now: datetime):
    """Sottoclasse di `datetime` che risponde SEMPRE `fixed_now` a `.now()`
    (con conversione di fuso se richiesta), lasciando intatti gli altri
    classmethod ereditati (`fromisoformat`, `replace`, ...) - usata per
    rendere `_is_job_due` deterministico nei test senza dipendere
    dall'orario reale in cui girano."""

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is not None:
                return fixed_now.astimezone(tz)
            return fixed_now

    return _FixedDateTime


class TestBuildSchedulerJobsSeparation(unittest.TestCase):
    """OPS-01 (acceptance criteria): scheduler leggibile/configurabile, con
    job SEPARATI per data sync (frequenti) e training (indipendente) - mai
    un job unico che incatena import+retrain."""

    def test_registers_exactly_seven_independent_jobs(self):
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        job_ids = {job.id for job in sched.get_jobs()}
        self.assertEqual(
            job_ids,
            {
                "data_sync_today",
                "data_settlement",
                "data_quality_report",
                "data_future_sync",
                "data_daily_refresh",
                "ml_training",
                "data_sync_live",
            },
        )

    def test_build_scheduler_uses_default_config_when_none_given(self):
        sched = scheduler_module.build_scheduler()
        self.assertEqual(len(sched.get_jobs()), 7)


class TestMaxInstancesAndCoalesce(unittest.TestCase):
    """Acceptance criteria "max_instances/coalesce corretti": OGNI job,
    senza eccezioni, deve avere max_instances=1 e coalesce=True."""

    def test_every_job_has_max_instances_one_and_coalesce_true(self):
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        jobs = sched.get_jobs()
        self.assertEqual(len(jobs), 7)
        for job in jobs:
            self.assertEqual(job.max_instances, 1, f"{job.id} deve avere max_instances=1")
            self.assertTrue(job.coalesce, f"{job.id} deve avere coalesce=True")


class TestAllJobsShareHeartbeatTrigger(unittest.TestCase):
    """Redesign 2026-09-09 (orario editabile da Impostazioni senza restart):
    nessun job ha piu' un trigger APScheduler calcolato da `cfg.*` — TUTTI
    condividono un unico heartbeat, e l'orario/intervallo EFFETTIVO e'
    deciso ad ogni tick da `_is_job_due` (self-gating) contro lo schedule
    corrente (`resolve_job_schedule`, che riflette un eventuale override
    salvato da Impostazioni)."""

    def test_every_job_uses_the_shared_heartbeat_interval(self):
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        for job in sched.get_jobs():
            self.assertIsInstance(job.trigger, IntervalTrigger)
            self.assertEqual(job.trigger.interval.total_seconds(), scheduler_module._HEARTBEAT_SECONDS)

    def test_changing_cfg_schedule_values_does_not_change_the_trigger(self):
        """`cfg.*` resta solo il DEFAULT usato da `resolve_job_schedule`
        quando non c'e' un override salvato — MAI piu' il trigger
        APScheduler in se' (che resta il battito fisso, uguale per tutti)."""
        sched_a = scheduler_module.build_scheduler(cfg=_cfg(training_hour=1, data_sync_interval_minutes=10))
        sched_b = scheduler_module.build_scheduler(cfg=_cfg(training_hour=20, data_sync_interval_minutes=50))
        for job_id in ("ml_training", "data_sync_today"):
            self.assertEqual(
                sched_a.get_job(job_id).trigger.interval,
                sched_b.get_job(job_id).trigger.interval,
            )


class TestJobTargetsAreCorrectAndIndependent(unittest.TestCase):
    """Verifica esplicita che il training NON sia agganciato ai data job
    (acceptance criteria "No retrain automatico ad ogni import")."""

    def test_training_job_targets_only_run_manual_retrain(self):
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        training_job = sched.get_job("ml_training")
        self.assertIs(_target_func(training_job), scheduler_module.run_manual_retrain)

    def test_data_sync_job_targets_today_update_never_retrain(self):
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        data_job = sched.get_job("data_sync_today")
        self.assertIs(_target_func(data_job), scheduler_module.run_manual_today_update)
        self.assertIsNot(_target_func(data_job), scheduler_module.run_manual_retrain)

    def test_settlement_job_targets_run_manual_settlement_never_retrain(self):
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        settlement_job = sched.get_job("data_settlement")
        self.assertIs(_target_func(settlement_job), scheduler_module.run_manual_settlement)
        self.assertIsNot(_target_func(settlement_job), scheduler_module.run_manual_retrain)

    def test_data_quality_report_job_targets_run_data_quality_report_never_retrain(self):
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        quality_job = sched.get_job("data_quality_report")
        self.assertIs(_target_func(quality_job), scheduler_module.run_data_quality_report)
        self.assertIsNot(_target_func(quality_job), scheduler_module.run_manual_retrain)

    def test_future_sync_job_targets_run_manual_future_sync(self):
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        future_job = sched.get_job("data_future_sync")
        self.assertIs(_target_func(future_job), scheduler_module.run_manual_future_sync)

    def test_daily_refresh_job_targets_run_daily_refresh_never_retrain(self):
        """Il job "Aggiorna tutto" (ieri + prossimi giorni) deve puntare a
        `run_daily_refresh` - MAI a `run_manual_retrain` (nessun retrain
        agganciato a questo job, stesso principio degli altri data job)."""
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        daily_refresh_job = sched.get_job("data_daily_refresh")
        self.assertIs(_target_func(daily_refresh_job), scheduler_module.run_daily_refresh)
        self.assertIsNot(_target_func(daily_refresh_job), scheduler_module.run_manual_retrain)

    def test_live_sync_job_targets_run_manual_live_sync_never_retrain(self):
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        live_job = sched.get_job("data_sync_live")
        self.assertIs(_target_func(live_job), scheduler_module.run_manual_live_sync)
        self.assertIsNot(_target_func(live_job), scheduler_module.run_manual_retrain)

    def test_no_single_job_bundles_import_and_retrain(self):
        # Nessuno dei job registrati deve puntare a `run_daily_pipeline`
        # (l'anti-pattern rimosso da OPS-01: import+retrain nello stesso job).
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        targets = {_target_func(job) for job in sched.get_jobs()}
        self.assertNotIn(scheduler_module.run_daily_pipeline, targets)


class TestLastCompletedRun(unittest.TestCase):
    """`_last_completed_run` e' l'unica fonte di verita' per "quando e'
    girato l'ultima volta" un job - isolata da file reali tramite un
    `JobHistory` finto, cosi' i test non toccano mai
    `best_models/jobs_history.jsonl`."""

    def test_filters_to_success_and_failed_and_returns_the_last_one(self):
        class FakeHistory:
            def tail(self, limit, job_type):
                return [
                    {"job_type": job_type, "status": "running"},
                    {"job_type": job_type, "status": "success", "finished_at": "2026-09-09T10:00:00+00:00"},
                    {"job_type": job_type, "status": "failed", "finished_at": "2026-09-09T11:00:00+00:00"},
                ]

        with mock.patch.object(scheduler_module, "JobHistory", FakeHistory):
            result = scheduler_module._last_completed_run("today_update")
        self.assertEqual(result["status"], "failed")

    def test_returns_none_when_no_completed_rows(self):
        class FakeHistory:
            def tail(self, limit, job_type):
                return [{"job_type": job_type, "status": "running"}]

        with mock.patch.object(scheduler_module, "JobHistory", FakeHistory):
            result = scheduler_module._last_completed_run("today_update")
        self.assertIsNone(result)


class TestParseHistoryTimestamp(unittest.TestCase):
    def test_uses_finished_at_when_present(self):
        row = {"finished_at": "2026-09-09T10:00:00+00:00", "timestamp": "2026-09-09T09:00:00+00:00"}
        parsed = scheduler_module._parse_history_timestamp(row)
        self.assertEqual(parsed.hour, 10)

    def test_falls_back_to_timestamp_when_finished_at_missing(self):
        row = {"timestamp": "2026-09-09T09:00:00+00:00"}
        parsed = scheduler_module._parse_history_timestamp(row)
        self.assertEqual(parsed.hour, 9)

    def test_naive_timestamp_assumed_utc(self):
        row = {"finished_at": "2026-09-09T10:00:00"}
        parsed = scheduler_module._parse_history_timestamp(row)
        self.assertEqual(parsed.tzinfo, timezone.utc)


class TestIsJobDue(unittest.TestCase):
    """Logica di self-gating (`_is_job_due`), il cuore del redesign
    "orario editabile senza restart": `resolve_job_schedule` e
    `_last_completed_run` sono mockati per isolare il test dal filesystem,
    `datetime.now()` e' fissato per rendere i confronti deterministici."""

    def setUp(self):
        self.cfg = _cfg()

    def _is_due(self, job_id, schedule, last_run, fixed_now):
        with mock.patch.object(scheduler_module, "resolve_job_schedule", return_value=schedule), \
             mock.patch.object(scheduler_module, "_last_completed_run", return_value=last_run), \
             mock.patch.object(scheduler_module, "datetime", _fixed_datetime_class(fixed_now)):
            return scheduler_module._is_job_due(job_id, self.cfg)

    def test_daily_not_due_before_target_hour_today(self):
        fixed_now = datetime(2026, 9, 9, 10, 0, tzinfo=scheduler_module._SCHEDULER_TIMEZONE)
        due = self._is_due("ml_training", {"hour": 23, "minute": 0}, last_run=None, fixed_now=fixed_now)
        self.assertFalse(due)

    def test_daily_due_after_target_hour_with_no_previous_run(self):
        fixed_now = datetime(2026, 9, 9, 23, 30, tzinfo=scheduler_module._SCHEDULER_TIMEZONE)
        due = self._is_due("ml_training", {"hour": 23, "minute": 0}, last_run=None, fixed_now=fixed_now)
        self.assertTrue(due)

    def test_daily_not_due_again_same_local_day(self):
        fixed_now = datetime(2026, 9, 9, 23, 30, tzinfo=scheduler_module._SCHEDULER_TIMEZONE)
        last_run_local = datetime(2026, 9, 9, 8, 0, tzinfo=scheduler_module._SCHEDULER_TIMEZONE)
        last_run = {"status": "success", "finished_at": last_run_local.astimezone(timezone.utc).isoformat()}
        due = self._is_due("ml_training", {"hour": 23, "minute": 0}, last_run=last_run, fixed_now=fixed_now)
        self.assertFalse(due)

    def test_daily_due_again_next_local_day(self):
        fixed_now = datetime(2026, 9, 10, 23, 30, tzinfo=scheduler_module._SCHEDULER_TIMEZONE)
        last_run_local = datetime(2026, 9, 9, 8, 0, tzinfo=scheduler_module._SCHEDULER_TIMEZONE)
        last_run = {"status": "success", "finished_at": last_run_local.astimezone(timezone.utc).isoformat()}
        due = self._is_due("ml_training", {"hour": 23, "minute": 0}, last_run=last_run, fixed_now=fixed_now)
        self.assertTrue(due)

    def test_interval_minutes_not_due_before_elapsed(self):
        fixed_now = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
        last_run = {"status": "success", "finished_at": (fixed_now - timedelta(minutes=5)).isoformat()}
        due = self._is_due("data_sync_today", {"interval_minutes": 30}, last_run=last_run, fixed_now=fixed_now)
        self.assertFalse(due)

    def test_interval_minutes_due_after_elapsed(self):
        fixed_now = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
        last_run = {"status": "success", "finished_at": (fixed_now - timedelta(minutes=31)).isoformat()}
        due = self._is_due("data_sync_today", {"interval_minutes": 30}, last_run=last_run, fixed_now=fixed_now)
        self.assertTrue(due)

    def test_interval_seconds_due_with_no_previous_run(self):
        fixed_now = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
        due = self._is_due("data_sync_live", {"interval_seconds": 90}, last_run=None, fixed_now=fixed_now)
        self.assertTrue(due)

    def test_interval_seconds_not_due_before_elapsed(self):
        fixed_now = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
        last_run = {"status": "failed", "finished_at": (fixed_now - timedelta(seconds=30)).isoformat()}
        due = self._is_due("data_sync_live", {"interval_seconds": 90}, last_run=last_run, fixed_now=fixed_now)
        self.assertFalse(due)


class TestRunIfDue(unittest.TestCase):
    """`_run_if_due` combina enabled (Impostazioni) + due (`_is_job_due`) -
    entrambi devono essere veri perche' la funzione reale venga chiamata."""

    def test_skips_when_disabled_even_if_due(self):
        called = []
        with mock.patch.object(scheduler_module, "is_job_enabled", return_value=False), \
             mock.patch.object(scheduler_module, "_is_job_due", return_value=True):
            result = scheduler_module._run_if_due(
                "data_sync_today", lambda: called.append(True) or {"ran": True}, cfg=_cfg()
            )
        self.assertIsNone(result)
        self.assertEqual(called, [])

    def test_skips_when_enabled_but_not_due(self):
        called = []
        with mock.patch.object(scheduler_module, "is_job_enabled", return_value=True), \
             mock.patch.object(scheduler_module, "_is_job_due", return_value=False):
            result = scheduler_module._run_if_due(
                "data_sync_today", lambda: called.append(True) or {"ran": True}, cfg=_cfg()
            )
        self.assertIsNone(result)
        self.assertEqual(called, [])

    def test_runs_when_enabled_and_due(self):
        with mock.patch.object(scheduler_module, "is_job_enabled", return_value=True), \
             mock.patch.object(scheduler_module, "_is_job_due", return_value=True):
            result = scheduler_module._run_if_due("data_sync_today", lambda: {"ran": True}, cfg=_cfg())
        self.assertEqual(result, {"ran": True})

    def test_passes_through_extra_kwargs_to_target_func(self):
        received = {}

        def fake(days_ahead):
            received["days_ahead"] = days_ahead
            return "done"

        with mock.patch.object(scheduler_module, "is_job_enabled", return_value=True), \
             mock.patch.object(scheduler_module, "_is_job_due", return_value=True):
            result = scheduler_module._run_if_due("data_daily_refresh", fake, cfg=_cfg(), days_ahead=7)
        self.assertEqual(received["days_ahead"], 7)
        self.assertEqual(result, "done")


if __name__ == "__main__":
    unittest.main()
