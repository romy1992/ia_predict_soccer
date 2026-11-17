"""`sync_job_settings_with_quota`: pausa a quota esaurita e ripristino.

Il bucket e' isolato dalla fixture autouse di `conftest.py`; lo snapshot
quota viene passato esplicitamente, quindi nessuna chiamata al provider.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from src.jobs import api_quota_state
from src.jobs import job_settings as js
from src.storage import bucket_store
from tests.support.fake_bucket import FakeS3Client


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

    def test_settlement_dichiarato_parziale_nelle_definizioni(self):
        righe = {r["job_id"]: r for r in js.list_job_definitions()}
        self.assertFalse(righe["data_settlement"]["calls_api_sports"])
        self.assertTrue(righe["data_settlement"]["partially_calls_api_sports"])
        self.assertFalse(righe["ml_training"]["partially_calls_api_sports"])
        from src.api.schemas import JobSettingRow
        JobSettingRow(**righe["data_settlement"])

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

    def test_job_aggiunto_durante_la_pausa_torna_al_default_al_reset(self):
        """Deploy di un job sensibile mentre la pausa e' attiva: non e' nella
        baseline, la pausa lo spegne e al reset deve tornare al suo default."""
        js.sync_job_settings_with_quota(_esaurita())
        stato = js.get_quota_pause_state()
        for job_id in ("closing_odds_capture", "data_sync_live"):
            stato["previous_settings"].pop(job_id)
        js._write_quota_pause_state(stato)
        js.update_job_settings({"closing_odds_capture": True, "data_sync_live": True})
        js.sync_job_settings_with_quota(_esaurita())
        self.assertFalse(js.get_job_settings()["closing_odds_capture"])

        js.sync_job_settings_with_quota({})
        after = js.get_job_settings()
        self.assertTrue(after["closing_odds_capture"])
        self.assertEqual(after["data_sync_live"], js.DEFAULT_JOB_SETTINGS["data_sync_live"])

    def test_toggle_fuori_pausa_non_scrive_baseline(self):
        self.assertFalse(js.remember_operator_toggles_during_pause({self.sensibili[0]: False}))
        self.assertEqual(js.get_quota_pause_state(), {})

    def test_la_lettura_quota_non_chiama_il_provider(self):
        with mock.patch("src.service_ia.pre_processing.api_sports_provider.ApiSportsProvider") as prov:
            js.sync_job_settings_with_quota()
        prov.assert_not_called()


class TestQuotaPauseSenzaScrittureDistruttive(unittest.TestCase):
    """Un errore di LETTURA del bucket non deve mai trasformarsi in una
    scrittura che cancella i toggle dell'operatore o la baseline di ripristino.
    Prima ogni lettura fallita ripiegava su un valore vuoto (default / nessuna
    pausa / quota non esaurita) e lo stato veniva riscritto partendo da li'."""

    def setUp(self):
        self.fake = FakeS3Client()
        bucket_store.set_client_for_tests(self.fake, "test-bucket")
        self.sensibili = sorted(js.QUOTA_SENSITIVE_JOB_IDS)
        # Stato dell'operatore DIVERSO dai default: un job non sensibile spento
        # e un job sensibile spento a mano.
        self.spento_a_mano = next(j for j in self.sensibili if js.DEFAULT_JOB_SETTINGS[j])
        js.update_job_settings({"ml_training": False, self.spento_a_mano: False})
        self.stato_operatore = dict(js.get_job_settings())
        self.assertNotEqual(self.stato_operatore, js.DEFAULT_JOB_SETTINGS)

    def _stored(self, key):
        return bucket_store.get_json(key, default=None)

    def test_update_job_settings_con_lettura_fallita_non_scrive_i_default(self):
        self.fake.deny_read_keys.add(js._SETTINGS_KEY)
        with self.assertRaises(Exception):
            js.update_job_settings({"ml_training": True})
        self.fake.deny_read_keys.clear()
        self.assertEqual(js.get_job_settings(), self.stato_operatore)

    def test_impostazioni_illeggibili_a_quota_esaurita_non_vengono_riscritte(self):
        self.fake.deny_read_keys.add(js._SETTINGS_KEY)
        self.assertFalse(js.sync_job_settings_with_quota(_esaurita()))
        self.fake.deny_read_keys.clear()
        # Nessun toggle perso e nessuna baseline "di default" salvata.
        self.assertEqual(js.get_job_settings(), self.stato_operatore)
        self.assertEqual(js.get_quota_pause_state(), {})
        # Al controllo successivo (bucket di nuovo leggibile) la pausa scatta
        # con la baseline VERA dell'operatore.
        self.assertTrue(js.sync_job_settings_with_quota(_esaurita()))
        self.assertEqual(
            js.get_quota_pause_state()["previous_settings"],
            {j: self.stato_operatore[j] for j in self.sensibili},
        )

    def test_pausa_illeggibile_non_sovrascrive_la_baseline_con_lo_stato_spento(self):
        self.assertTrue(js.sync_job_settings_with_quota(_esaurita()))
        baseline = dict(js.get_quota_pause_state()["previous_settings"])
        self.fake.deny_read_keys.add(js._QUOTA_PAUSE_KEY)
        self.assertFalse(js.sync_job_settings_with_quota(_esaurita()))
        self.fake.deny_read_keys.clear()
        self.assertEqual(js.get_quota_pause_state()["previous_settings"], baseline)
        # Reset: si torna esattamente allo stato dell'operatore.
        self.assertFalse(js.sync_job_settings_with_quota({}))
        self.assertEqual(js.get_job_settings(), self.stato_operatore)

    def test_stato_quota_illeggibile_non_toglie_la_pausa(self):
        api_quota_state._write_state(_esaurita())
        self.assertTrue(js.sync_job_settings_with_quota())
        self.fake.deny_read_keys.add(api_quota_state._STATE_KEY)
        self.assertFalse(js.sync_job_settings_with_quota())
        self.fake.deny_read_keys.clear()
        self.assertTrue(js.get_quota_pause_state().get("paused_date"))
        for j in self.sensibili:
            self.assertFalse(js.get_job_settings()[j])

    def test_baseline_non_salvata_non_disattiva_i_job(self):
        originale = bucket_store.put_json

        def put_json(key, payload):
            if key == js._QUOTA_PAUSE_KEY:
                raise RuntimeError("scrittura rifiutata")
            return originale(key, payload)

        with mock.patch.object(bucket_store, "put_json", side_effect=put_json):
            self.assertFalse(js.sync_job_settings_with_quota(_esaurita()))
        self.assertEqual(js.get_job_settings(), self.stato_operatore)

    def test_ripristino_ignora_job_non_piu_esistenti_e_chiude_la_pausa(self):
        self.assertTrue(js.sync_job_settings_with_quota(_esaurita()))
        stato = js.get_quota_pause_state()
        stato["previous_settings"]["job_rimosso_da_un_deploy"] = True
        js._write_quota_pause_state(stato)
        self.assertFalse(js.sync_job_settings_with_quota({}))
        self.assertEqual(js.get_quota_pause_state(), {})
        self.assertEqual(js.get_job_settings(), self.stato_operatore)
        self.assertNotIn("job_rimosso_da_un_deploy", self._stored(js._SETTINGS_KEY))

    def test_job_spento_a_mano_prima_della_pausa_non_viene_riacceso(self):
        self.assertTrue(js.sync_job_settings_with_quota(_esaurita()))
        self.assertFalse(js.sync_job_settings_with_quota({}))
        self.assertFalse(js.get_job_settings()[self.spento_a_mano])
        self.assertFalse(js.get_job_settings()["ml_training"])

    def test_la_sincronizzazione_legge_solo_il_bucket(self):
        """Tre letture al massimo (quota, pausa, impostazioni) e nessuna
        richiesta HTTP: lo stato quota e' quello gia' salvato dalle chiamate
        dati, non un `GET /status` al provider."""
        api_quota_state._write_state(_esaurita())
        js.sync_job_settings_with_quota()  # prima rilevazione: scrive
        prima = self.fake.gets
        with mock.patch("requests.get") as http_get, mock.patch("requests.Session.request") as http_request:
            self.assertTrue(js.sync_job_settings_with_quota())
        http_get.assert_not_called()
        http_request.assert_not_called()
        self.assertEqual(self.fake.gets - prima, 3)

    def test_rilevazioni_concorrenti_non_perdono_la_baseline(self):
        """Piu' thread che rilevano insieme la quota esaurita (job dello stesso
        tick): la baseline resta quella PRIMA della pausa, mai lo stato gia'
        disattivato da un altro thread."""
        import threading

        barriera = threading.Barrier(8)
        errori = []

        def worker():
            try:
                barriera.wait(timeout=5)
                js.sync_job_settings_with_quota(_esaurita())
            except Exception as exc:  # pragma: no cover - solo diagnostica
                errori.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(errori, [])
        self.assertEqual(
            js.get_quota_pause_state()["previous_settings"],
            {j: self.stato_operatore[j] for j in self.sensibili},
        )


if __name__ == "__main__":
    unittest.main()
