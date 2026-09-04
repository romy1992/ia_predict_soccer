import os
import unittest
from unittest import mock

from src.service_ia.config.app_config import load_app_config

_SCHEDULING_ENV_KEYS = [
    "TRAINING_HOUR",
    "TRAINING_MINUTE",
    "SCHEDULER_HOUR",
    "SCHEDULER_MINUTE",
    "DATA_SYNC_INTERVAL_MINUTES",
    "SETTLEMENT_INTERVAL_MINUTES",
    "FUTURE_SYNC_HOUR",
    "FUTURE_SYNC_MINUTE",
]


class TestLoadAppConfigScheduling(unittest.TestCase):
    """OPS-01: `AppConfig` espone orari/intervalli SEPARATI per i data job
    (frequenti) e il training job (indipendente) - mai un unico
    `scheduler_hour` condiviso. Ogni test isola le env var rilevanti (mai
    influenzato da un `properties/config.env` reale gia' popolato)."""

    def _run_with_env(self, overrides: dict):
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in _SCHEDULING_ENV_KEYS:
                os.environ.pop(key, None)
            os.environ.update(overrides)
            return load_app_config()

    def test_default_values_when_env_not_set(self):
        cfg = self._run_with_env({})
        self.assertEqual(cfg.training_hour, 23)
        self.assertEqual(cfg.training_minute, 0)
        self.assertEqual(cfg.data_sync_interval_minutes, 30)
        self.assertEqual(cfg.settlement_interval_minutes, 60)
        self.assertEqual(cfg.future_sync_hour, 4)
        self.assertEqual(cfg.future_sync_minute, 30)

    def test_training_hour_explicit_override(self):
        cfg = self._run_with_env({"TRAINING_HOUR": "5", "TRAINING_MINUTE": "15"})
        self.assertEqual(cfg.training_hour, 5)
        self.assertEqual(cfg.training_minute, 15)

    def test_training_hour_falls_back_to_legacy_scheduler_hour(self):
        # Retrocompatibilita': ambienti gia' configurati con SCHEDULER_HOUR/
        # SCHEDULER_MINUTE (nome storico) continuano a funzionare invariati.
        cfg = self._run_with_env({"SCHEDULER_HOUR": "2", "SCHEDULER_MINUTE": "45"})
        self.assertEqual(cfg.training_hour, 2)
        self.assertEqual(cfg.training_minute, 45)

    def test_training_hour_prefers_explicit_name_over_legacy(self):
        cfg = self._run_with_env({"TRAINING_HOUR": "5", "SCHEDULER_HOUR": "2"})
        self.assertEqual(cfg.training_hour, 5)

    def test_data_sync_interval_override(self):
        cfg = self._run_with_env({"DATA_SYNC_INTERVAL_MINUTES": "15"})
        self.assertEqual(cfg.data_sync_interval_minutes, 15)

    def test_settlement_interval_override(self):
        cfg = self._run_with_env({"SETTLEMENT_INTERVAL_MINUTES": "10"})
        self.assertEqual(cfg.settlement_interval_minutes, 10)

    def test_future_sync_hour_override(self):
        cfg = self._run_with_env({"FUTURE_SYNC_HOUR": "6", "FUTURE_SYNC_MINUTE": "10"})
        self.assertEqual(cfg.future_sync_hour, 6)
        self.assertEqual(cfg.future_sync_minute, 10)

    def test_invalid_value_falls_back_to_default(self):
        cfg = self._run_with_env({"DATA_SYNC_INTERVAL_MINUTES": "not-a-number"})
        self.assertEqual(cfg.data_sync_interval_minutes, 30)

    def test_training_and_data_sync_are_independent_fields(self):
        # Cambiare l'orario di training non deve toccare l'intervallo dei
        # data job (acceptance criteria implicito: job separati/indipendenti).
        cfg = self._run_with_env({"TRAINING_HOUR": "3", "DATA_SYNC_INTERVAL_MINUTES": "20"})
        self.assertEqual(cfg.training_hour, 3)
        self.assertEqual(cfg.data_sync_interval_minutes, 20)


if __name__ == "__main__":
    unittest.main()

