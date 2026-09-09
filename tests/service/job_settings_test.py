import os
import tempfile
import unittest
from unittest import mock

from src.jobs import job_settings
from src.service_ia.config.app_config import AppConfig


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


class _IsolatedSchedulePath(unittest.TestCase):
    """Isola `job_schedule.json` in una directory temporanea per ogni test,
    cosi' nessun test tocca mai il file reale in `best_models/` del
    progetto (stesso principio di `JobHistory(path=...)` nei test
    esistenti, qui applicato via monkeypatch del path module-level)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        schedule_path = os.path.join(self._tmp.name, "job_schedule.json")
        patcher = mock.patch.object(job_settings, "_schedule_path", return_value=schedule_path)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestCfgDefaultSchedule(_IsolatedSchedulePath):
    def test_daily_job_default_from_cfg(self):
        cfg = _cfg(training_hour=3, training_minute=45)
        default = job_settings._cfg_default_schedule("ml_training", cfg)
        self.assertEqual(default, {"hour": 3, "minute": 45})

    def test_interval_minutes_job_default_from_cfg(self):
        cfg = _cfg(data_sync_interval_minutes=12)
        default = job_settings._cfg_default_schedule("data_sync_today", cfg)
        self.assertEqual(default, {"interval_minutes": 12})

    def test_interval_seconds_job_default_from_cfg(self):
        cfg = _cfg(live_sync_interval_seconds=45)
        default = job_settings._cfg_default_schedule("data_sync_live", cfg)
        self.assertEqual(default, {"interval_seconds": 45})


class TestGetJobScheduleOverrides(_IsolatedSchedulePath):
    def test_returns_empty_dict_when_no_file(self):
        self.assertEqual(job_settings.get_job_schedule_overrides(), {})

    def test_returns_empty_dict_on_corrupt_file(self):
        path = job_settings._schedule_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        self.assertEqual(job_settings.get_job_schedule_overrides(), {})


class TestValidateSchedule(_IsolatedSchedulePath):
    def test_rejects_unknown_job_id(self):
        with self.assertRaises(ValueError):
            job_settings._validate_schedule("does_not_exist", {"hour": 1, "minute": 1})

    def test_rejects_wrong_key_set(self):
        with self.assertRaises(ValueError):
            job_settings._validate_schedule("ml_training", {"interval_minutes": 30})

    def test_rejects_missing_key(self):
        with self.assertRaises(ValueError):
            job_settings._validate_schedule("ml_training", {"hour": 5})

    def test_rejects_non_integer_value(self):
        with self.assertRaises(ValueError):
            job_settings._validate_schedule("ml_training", {"hour": "abc", "minute": 0})

    def test_rejects_out_of_bounds_hour(self):
        with self.assertRaises(ValueError):
            job_settings._validate_schedule("ml_training", {"hour": 24, "minute": 0})

    def test_rejects_out_of_bounds_interval_seconds_too_small(self):
        with self.assertRaises(ValueError):
            job_settings._validate_schedule("data_sync_live", {"interval_seconds": 10})

    def test_accepts_valid_daily_schedule(self):
        validated = job_settings._validate_schedule("ml_training", {"hour": 5, "minute": 30})
        self.assertEqual(validated, {"hour": 5, "minute": 30})

    def test_coerces_numeric_strings(self):
        validated = job_settings._validate_schedule("ml_training", {"hour": "5", "minute": "30"})
        self.assertEqual(validated, {"hour": 5, "minute": 30})


class TestUpdateAndResetJobSchedule(_IsolatedSchedulePath):
    def test_update_then_read_back_via_overrides(self):
        job_settings.update_job_schedule("ml_training", {"hour": 2, "minute": 15})
        overrides = job_settings.get_job_schedule_overrides()
        self.assertEqual(overrides["ml_training"], {"hour": 2, "minute": 15})

    def test_update_persists_atomically_no_tmp_file_left(self):
        job_settings.update_job_schedule("data_sync_today", {"interval_minutes": 20})
        path = job_settings._schedule_path()
        self.assertTrue(os.path.exists(path))
        self.assertFalse(os.path.exists(f"{path}.tmp"))

    def test_update_does_not_clobber_other_jobs(self):
        job_settings.update_job_schedule("ml_training", {"hour": 2, "minute": 15})
        job_settings.update_job_schedule("data_sync_today", {"interval_minutes": 20})
        overrides = job_settings.get_job_schedule_overrides()
        self.assertEqual(overrides["ml_training"], {"hour": 2, "minute": 15})
        self.assertEqual(overrides["data_sync_today"], {"interval_minutes": 20})

    def test_reset_removes_only_that_job_override(self):
        job_settings.update_job_schedule("ml_training", {"hour": 2, "minute": 15})
        job_settings.update_job_schedule("data_sync_today", {"interval_minutes": 20})
        job_settings.reset_job_schedule("ml_training")
        overrides = job_settings.get_job_schedule_overrides()
        self.assertNotIn("ml_training", overrides)
        self.assertIn("data_sync_today", overrides)

    def test_reset_unknown_job_id_raises(self):
        with self.assertRaises(ValueError):
            job_settings.reset_job_schedule("does_not_exist")

    def test_update_rejects_invalid_schedule_and_does_not_persist(self):
        with self.assertRaises(ValueError):
            job_settings.update_job_schedule("ml_training", {"hour": 99, "minute": 0})
        self.assertEqual(job_settings.get_job_schedule_overrides(), {})


class TestResolveJobSchedule(_IsolatedSchedulePath):
    def test_returns_cfg_default_when_no_override(self):
        cfg = _cfg(training_hour=23, training_minute=0)
        resolved = job_settings.resolve_job_schedule("ml_training", cfg=cfg)
        self.assertEqual(resolved, {"hour": 23, "minute": 0})

    def test_returns_saved_override_instead_of_cfg_default(self):
        cfg = _cfg(training_hour=23, training_minute=0)
        job_settings.update_job_schedule("ml_training", {"hour": 6, "minute": 10})
        resolved = job_settings.resolve_job_schedule("ml_training", cfg=cfg)
        self.assertEqual(resolved, {"hour": 6, "minute": 10})

    def test_falls_back_to_cfg_default_after_reset(self):
        cfg = _cfg(training_hour=23, training_minute=0)
        job_settings.update_job_schedule("ml_training", {"hour": 6, "minute": 10})
        job_settings.reset_job_schedule("ml_training")
        resolved = job_settings.resolve_job_schedule("ml_training", cfg=cfg)
        self.assertEqual(resolved, {"hour": 23, "minute": 0})

    def test_unknown_job_id_raises(self):
        with self.assertRaises(ValueError):
            job_settings.resolve_job_schedule("does_not_exist")


class TestListJobDefinitionsIncludesSchedule(_IsolatedSchedulePath):
    def test_rows_include_schedule_kind_and_schedule_and_default_flag(self):
        cfg = _cfg(training_hour=23, training_minute=0)
        rows = job_settings.list_job_definitions(cfg=cfg)
        by_id = {row["job_id"]: row for row in rows}

        training_row = by_id["ml_training"]
        self.assertEqual(training_row["schedule_kind"], "daily")
        self.assertEqual(training_row["schedule"], {"hour": 23, "minute": 0})
        self.assertTrue(training_row["schedule_is_default"])

        live_row = by_id["data_sync_live"]
        self.assertEqual(live_row["schedule_kind"], "interval_seconds")

    def test_schedule_is_default_false_after_override(self):
        cfg = _cfg(training_hour=23, training_minute=0)
        job_settings.update_job_schedule("ml_training", {"hour": 6, "minute": 10})
        rows = job_settings.list_job_definitions(cfg=cfg)
        by_id = {row["job_id"]: row for row in rows}
        self.assertFalse(by_id["ml_training"]["schedule_is_default"])
        self.assertEqual(by_id["ml_training"]["schedule"], {"hour": 6, "minute": 10})
        self.assertTrue(by_id["data_sync_today"]["schedule_is_default"])


if __name__ == "__main__":
    unittest.main()
