import unittest

from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.jobs import scheduler as scheduler_module
from src.service_ia.config.app_config import AppConfig


def _cfg(**overrides) -> AppConfig:
    base = dict(
        leagues=[39],
        seasons=[2026],
        data_sync_interval_minutes=30,
        settlement_interval_minutes=60,
        future_sync_hour=4,
        future_sync_minute=30,
        training_hour=23,
        training_minute=0,
        live_sync_interval_seconds=90,
        live_cache_ttl_seconds=20,
        database_url="sqlite://",
        database_schema="public",
    )
    base.update(overrides)
    return AppConfig(**base)


def _cron_field(trigger: CronTrigger, name: str) -> str:
    for field in trigger.fields:
        if field.name == name:
            return str(field)
    raise AssertionError(f"campo cron '{name}' non trovato")


class TestBuildSchedulerJobsSeparation(unittest.TestCase):
    """OPS-01 (acceptance criteria): scheduler leggibile/configurabile, con
    job SEPARATI per data sync (frequenti) e training (indipendente) - mai
    un job unico che incatena import+retrain."""

    def test_registers_exactly_four_independent_jobs(self):
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        job_ids = {job.id for job in sched.get_jobs()}
        self.assertEqual(
            job_ids,
            {"data_sync_today", "data_settlement", "data_future_sync", "ml_training", "data_sync_live"},
        )

    def test_build_scheduler_uses_default_config_when_none_given(self):
        sched = scheduler_module.build_scheduler()
        self.assertEqual(len(sched.get_jobs()), 5)


class TestMaxInstancesAndCoalesce(unittest.TestCase):
    """Acceptance criteria "max_instances/coalesce corretti": OGNI job,
    senza eccezioni, deve avere max_instances=1 e coalesce=True."""

    def test_every_job_has_max_instances_one_and_coalesce_true(self):
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        jobs = sched.get_jobs()
        self.assertEqual(len(jobs), 5)
        for job in jobs:
            self.assertEqual(job.max_instances, 1, f"{job.id} deve avere max_instances=1")
            self.assertTrue(job.coalesce, f"{job.id} deve avere coalesce=True")


class TestTriggerTypesPerJob(unittest.TestCase):
    def test_data_sync_and_settlement_use_interval_trigger(self):
        sched = scheduler_module.build_scheduler(
            cfg=_cfg(data_sync_interval_minutes=15, settlement_interval_minutes=45)
        )
        jobs_by_id = {job.id: job for job in sched.get_jobs()}
        self.assertIsInstance(jobs_by_id["data_sync_today"].trigger, IntervalTrigger)
        self.assertIsInstance(jobs_by_id["data_settlement"].trigger, IntervalTrigger)
        self.assertEqual(jobs_by_id["data_sync_today"].trigger.interval.total_seconds(), 15 * 60)
        self.assertEqual(jobs_by_id["data_settlement"].trigger.interval.total_seconds(), 45 * 60)

    def test_future_sync_and_training_use_cron_trigger(self):
        sched = scheduler_module.build_scheduler(cfg=_cfg(future_sync_hour=6, training_hour=2))
        jobs_by_id = {job.id: job for job in sched.get_jobs()}
        self.assertIsInstance(jobs_by_id["data_future_sync"].trigger, CronTrigger)
        self.assertIsInstance(jobs_by_id["ml_training"].trigger, CronTrigger)
        self.assertEqual(_cron_field(jobs_by_id["data_future_sync"].trigger, "hour"), "6")
        self.assertEqual(_cron_field(jobs_by_id["ml_training"].trigger, "hour"), "2")

    def test_live_sync_uses_interval_trigger_in_seconds(self):
        """LIVE-01: polling MOLTO piu' frequente dei data job, espresso in
        secondi (non minuti) - trigger indipendente dagli altri."""
        sched = scheduler_module.build_scheduler(cfg=_cfg(live_sync_interval_seconds=45))
        live_job = sched.get_job("data_sync_live")
        self.assertIsInstance(live_job.trigger, IntervalTrigger)
        self.assertEqual(live_job.trigger.interval.total_seconds(), 45)


class TestJobTargetsAreCorrectAndIndependent(unittest.TestCase):
    """Verifica esplicita che il training NON sia agganciato ai data job
    (acceptance criteria "No retrain automatico ad ogni import")."""

    def test_training_job_targets_only_run_manual_retrain(self):
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        training_job = sched.get_job("ml_training")
        self.assertIs(training_job.func, scheduler_module.run_manual_retrain)

    def test_data_sync_job_targets_today_update_never_retrain(self):
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        data_job = sched.get_job("data_sync_today")
        self.assertIs(data_job.func, scheduler_module.run_manual_today_update)
        self.assertIsNot(data_job.func, scheduler_module.run_manual_retrain)

    def test_settlement_job_targets_run_manual_settlement_never_retrain(self):
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        settlement_job = sched.get_job("data_settlement")
        self.assertIs(settlement_job.func, scheduler_module.run_manual_settlement)
        self.assertIsNot(settlement_job.func, scheduler_module.run_manual_retrain)

    def test_future_sync_job_targets_run_manual_future_sync(self):
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        future_job = sched.get_job("data_future_sync")
        self.assertIs(future_job.func, scheduler_module.run_manual_future_sync)

    def test_live_sync_job_targets_run_manual_live_sync_never_retrain(self):
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        live_job = sched.get_job("data_sync_live")
        self.assertIs(live_job.func, scheduler_module.run_manual_live_sync)
        self.assertIsNot(live_job.func, scheduler_module.run_manual_retrain)

    def test_no_single_job_bundles_import_and_retrain(self):
        # Nessuno dei job registrati deve puntare a `run_daily_pipeline`
        # (l'anti-pattern rimosso da OPS-01: import+retrain nello stesso job).
        sched = scheduler_module.build_scheduler(cfg=_cfg())
        targets = {job.func for job in sched.get_jobs()}
        self.assertNotIn(scheduler_module.run_daily_pipeline, targets)


class TestTrainingScheduleIndependentFromDataJobs(unittest.TestCase):
    def test_changing_training_hour_does_not_affect_data_job_intervals(self):
        sched_a = scheduler_module.build_scheduler(cfg=_cfg(training_hour=1))
        sched_b = scheduler_module.build_scheduler(cfg=_cfg(training_hour=20))

        self.assertEqual(
            sched_a.get_job("data_sync_today").trigger.interval,
            sched_b.get_job("data_sync_today").trigger.interval,
        )
        self.assertEqual(
            sched_a.get_job("data_settlement").trigger.interval,
            sched_b.get_job("data_settlement").trigger.interval,
        )
        self.assertNotEqual(
            _cron_field(sched_a.get_job("ml_training").trigger, "hour"),
            _cron_field(sched_b.get_job("ml_training").trigger, "hour"),
        )

    def test_changing_data_sync_interval_does_not_affect_training_hour(self):
        sched_a = scheduler_module.build_scheduler(cfg=_cfg(data_sync_interval_minutes=10))
        sched_b = scheduler_module.build_scheduler(cfg=_cfg(data_sync_interval_minutes=50))

        self.assertEqual(
            _cron_field(sched_a.get_job("ml_training").trigger, "hour"),
            _cron_field(sched_b.get_job("ml_training").trigger, "hour"),
        )
        self.assertNotEqual(
            sched_a.get_job("data_sync_today").trigger.interval,
            sched_b.get_job("data_sync_today").trigger.interval,
        )


if __name__ == "__main__":
    unittest.main()

