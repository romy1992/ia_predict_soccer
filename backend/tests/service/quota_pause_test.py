"""`sync_job_settings_with_quota`: pausa a quota esaurita e ripristino.

Il bucket e' isolato dalla fixture autouse di `conftest.py`; lo snapshot
quota viene passato esplicitamente, quindi nessuna chiamata al provider.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from src.jobs import job_settings as js


def _esaurita(day_offset: int = 0) -> dict:
    ts = datetime.now(timezone.utc) + timedelta(days=day_offset)
    return {"status_error_message": "limite giornaliero raggiunto", "status_updated_at": ts.isoformat()}


class TestQuotaPause(unittest.TestCase):
    def setUp(self):
        self.sensibili = sorted(js.QUOTA_SENSITIVE_JOB_IDS)
        self.assertTrue(self.sensibili)

    def test_settlement_e_training_non_sono_sensibili(self):
        self.assertNotIn("data_settlement", js.QUOTA_SENSITIVE_JOB_IDS)
        self.assertNotIn("ml_training", js.QUOTA_SENSITIVE_JOB_IDS)

    def test_quota_non_esaurita_non_tocca_nulla(self):
        self.assertFalse(js.sync_job_settings_with_quota({}))
        self.assertEqual(js.get_quota_pause_state(), {})

    def test_pausa_disattiva_solo_i_sensibili_e_salva_baseline(self):
        base = dict(js.get_job_settings())
        self.assertTrue(js.sync_job_settings_with_quota(_esaurita()))
        now = js.get_job_settings()
        for j in self.sensibili:
            self.assertFalse(now[j])
        for j in set(base) - js.QUOTA_SENSITIVE_JOB_IDS:
            self.assertEqual(now[j], base[j])
        self.assertEqual(
            js.get_quota_pause_state()["previous_settings"], {j: base[j] for j in self.sensibili}
        )

    def test_ripristino_riporta_lo_stato_precedente_incluso_job_gia_spento(self):
        spento = self.sensibili[0]
        js.update_job_settings({spento: False})
        base = dict(js.get_job_settings())
        js.sync_job_settings_with_quota(_esaurita())
        self.assertFalse(js.sync_job_settings_with_quota({}))
        after = js.get_job_settings()
        for j in self.sensibili:
            self.assertEqual(after[j], base[j])
        self.assertFalse(after[spento])
        self.assertEqual(js.get_quota_pause_state(), {})

    def test_ricontrollo_riapplica_la_pausa_se_riacceso_a_mano(self):
        js.sync_job_settings_with_quota(_esaurita())
        js.update_job_settings({self.sensibili[0]: True})
        js.sync_job_settings_with_quota(_esaurita())
        self.assertFalse(js.get_job_settings()[self.sensibili[0]])

    def test_snapshot_di_ieri_non_attiva_la_pausa(self):
        self.assertFalse(js.sync_job_settings_with_quota(_esaurita(-1)))

    def test_pausa_di_ieri_non_ripristinata_conserva_la_baseline_originale(self):
        spento = self.sensibili[0]
        js.update_job_settings({spento: False})
        base = dict(js.get_job_settings())
        js.sync_job_settings_with_quota(_esaurita())
        # simula pausa rimasta dal giorno prima, poi quota esaurita di nuovo oggi
        stato = js.get_quota_pause_state()
        stato["paused_date"] = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
        js._write_quota_pause_state(stato)
        js.sync_job_settings_with_quota(_esaurita())
        self.assertEqual(
            js.get_quota_pause_state()["previous_settings"], {j: base[j] for j in self.sensibili}
        )
        js.sync_job_settings_with_quota({})
        after = js.get_job_settings()
        for j in self.sensibili:
            self.assertEqual(after[j], base[j])

    def test_toggle_operatore_durante_la_pausa_viene_ricordato(self):
        accesi = [j for j in self.sensibili if js.get_job_settings()[j]]
        self.assertTrue(accesi)
        js.sync_job_settings_with_quota(_esaurita())
        # l'operatore spegne esplicitamente un job durante la pausa
        js.update_job_settings({accesi[0]: False})
        self.assertTrue(js.remember_operator_toggles_during_pause({accesi[0]: False}))
        js.sync_job_settings_with_quota({})
        self.assertFalse(js.get_job_settings()[accesi[0]])

    def test_toggle_fuori_pausa_non_scrive_baseline(self):
        self.assertFalse(js.remember_operator_toggles_during_pause({self.sensibili[0]: False}))
        self.assertEqual(js.get_quota_pause_state(), {})

    def test_la_lettura_quota_non_chiama_il_provider(self):
        with mock.patch("src.service_ia.pre_processing.api_sports_provider.ApiSportsProvider") as prov:
            js.sync_job_settings_with_quota()
        prov.assert_not_called()


if __name__ == "__main__":
    unittest.main()
