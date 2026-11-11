"""Test per `ApiSportsProvider` (bug fix 2026-09-06).

Diagnosticato da `jobs_history.jsonl` + log reali del container `soccer_api`:
API-Sports usa lo STESSO campo `errors` (risposta HTTP 200) sia per la quota
GIORNALIERA esaurita (chiave "requests", permanente fino al reset del
giorno) sia per il rate-limit AL MINUTO (chiave "rateLimit", transitorio -
si risolve da solo). Il codice PRIMA trattava i due casi allo stesso modo
(`ApiSportsQuotaExceededError` fatale), abbandonando per sempre singole
fixture colpite da un blip di rate-limit momentaneo invece di ritentarle,
e loggando "quota giornaliera esaurita" anche quando la quota reale era
quasi intera (es. 5161/7500) - fuorviante e, soprattutto nel loop
per-fixture di `download_import_matches`, mai un vero stop immediato
(vedi `download_match_test.py::TestDownloadImportMatchesQuotaExceededMidway`
per quella parte)."""

import unittest
from unittest import mock

import requests

from src.service_ia.pre_processing.api_sports_provider import (
    ApiSportsCircuitOpenError,
    ApiSportsProvider,
    ApiSportsProviderConfig,
    ApiSportsQuotaExceededError,
    ApiSportsUnavailableError,
)


class TestIsDailyQuotaError(unittest.TestCase):
    def test_requests_key_is_daily_quota_error(self):
        errors = {"requests": "You have reached the request limit for the day, ..."}
        self.assertTrue(ApiSportsProvider._is_daily_quota_error(errors))

    def test_ratelimit_key_alone_is_not_daily_quota_error(self):
        errors = {"rateLimit": "Too many requests. You have exceeded the limit of requests per minute..."}
        self.assertFalse(ApiSportsProvider._is_daily_quota_error(errors))

    def test_ratelimit_and_requests_together_is_daily_quota_error(self):
        # Conservativo: se compare ANCHE "requests" insieme a "rateLimit",
        # trattiamo come quota giornaliera (mai sottovalutare un vero stop).
        errors = {"rateLimit": "...", "requests": "..."}
        self.assertTrue(ApiSportsProvider._is_daily_quota_error(errors))

    def test_other_errors_are_not_daily_quota_error(self):
        self.assertFalse(ApiSportsProvider._is_daily_quota_error({"plan": "Free plans do not have access to this season"}))
        self.assertFalse(ApiSportsProvider._is_daily_quota_error({"token": "Error/Missing application key"}))

    def test_limit_text_under_other_key_is_daily_quota_error(self):
        self.assertTrue(ApiSportsProvider._is_daily_quota_error({"x": "You have reached the request limit for the day"}))

    def test_empty_errors_is_not_daily_quota_error(self):
        self.assertFalse(ApiSportsProvider._is_daily_quota_error({}))
        self.assertFalse(ApiSportsProvider._is_daily_quota_error(None))

    def test_list_errors_is_conservatively_treated_as_daily_quota(self):
        # Formato non documentato/inatteso (lista invece di dict): mai un
        # retry infinito su un errore sconosciuto - trattato come stop.
        self.assertTrue(ApiSportsProvider._is_daily_quota_error(["some error"]))


def _provider() -> ApiSportsProvider:
    """Provider con backoff a zero: i test non devono dormire davvero."""
    return ApiSportsProvider(
        config=ApiSportsProviderConfig(
            base_url="https://esempio.test",
            api_key="chiave-finta",
            max_retries=3,
            retry_backoff_seconds=0,
        )
    )


def _risposta(status_code=200, payload=None, headers=None):
    risposta = mock.Mock(spec=requests.Response)
    risposta.status_code = status_code
    risposta.headers = headers or {}
    risposta.json.return_value = payload if payload is not None else {"response": []}
    return risposta


class TestRequestNonRitornaListaVuotaSuFallimento(unittest.TestCase):
    """Bug fix 2026-09-15. A retry esauriti `request()` faceva `return []`,
    lo STESSO valore che significa "nessuna partita per questa lega/data".
    Il chiamante non poteva distinguere i due casi: `download_import_matches`
    chiudeva la lega senza errori e il job finiva `success` con `failed=0` e
    `errors=[]`, senza traccia di un intero campionato non importato. Le
    partite mai aggiornate restavano a `NS` con kickoff nel passato, cioe' le
    righe che la Dashboard mostrava "In diretta"."""

    def test_errore_di_rete_persistente_solleva(self):
        with mock.patch("requests.get", side_effect=requests.ConnectionError("rete giu'")) as get:
            with self.assertRaises(ApiSportsUnavailableError):
                _provider().request(path="fixtures", params={"league": 135})
        self.assertEqual(get.call_count, 3)

    def test_rate_limit_al_minuto_mai_rientrato_solleva(self):
        risposta = _risposta(payload={"errors": {"rateLimit": "troppe richieste al minuto"}})
        with mock.patch("requests.get", return_value=risposta) as get:
            with self.assertRaises(ApiSportsUnavailableError):
                _provider().request(path="odds", params={"fixture": 1})
        # Ritentato fino a max_retries: il rate-limit al minuto e' transitorio
        # e va ritentato (bug fix 2026-09-06), ma se NON rientra il chiamante
        # deve saperlo invece di ricevere una lista vuota.
        self.assertEqual(get.call_count, 3)

    def test_errore_non_di_quota_solleva_subito_senza_segnare_quota(self):
        risposta = _risposta(payload={"errors": {"plan": "piano senza accesso a questa stagione"}})
        with mock.patch("requests.get", return_value=risposta) as get, \
                mock.patch("src.service_ia.pre_processing.api_sports_provider.record_status_snapshot") as snap:
            with self.assertRaises(ApiSportsUnavailableError):
                _provider().request(path="fixtures", params={"league": 135})
        self.assertEqual(get.call_count, 1)
        snap.assert_not_called()

    def test_5xx_ripetuti_sollevano(self):
        with mock.patch("requests.get", return_value=_risposta(status_code=503)) as get:
            with self.assertRaises(ApiSportsUnavailableError):
                _provider().request(path="fixtures", params={"league": 135})
        self.assertEqual(get.call_count, 3)

    def test_risposta_valida_dopo_un_blip_non_solleva(self):
        """Regressione esplicita: il retry deve continuare a FUNZIONARE. Un
        fallimento singolo seguito da una risposta buona non e' un errore."""
        buona = _risposta(payload={"response": [{"fixture": {"id": 7}}]})
        with mock.patch("requests.get", side_effect=[requests.Timeout("blip"), buona]) as get:
            righe = _provider().request(path="fixtures", params={"league": 135})
        self.assertEqual(righe, [{"fixture": {"id": 7}}])
        self.assertEqual(get.call_count, 2)

    def test_risposta_vuota_legittima_resta_lista_vuota(self):
        """Il caso "davvero nessuna partita" deve restare distinguibile: qui
        non si solleva niente, si ritorna `[]`."""
        with mock.patch("requests.get", return_value=_risposta(payload={"response": []})):
            self.assertEqual(_provider().request(path="fixtures", params={"date": "2026-09-14"}), [])


class TestBackoffEsponenzialeConJitter(unittest.TestCase):
    """Il backoff fra i tentativi e' esponenziale con jitter e tetto massimo:
    prima era lineare (`base * tentativo`) e senza jitter, quindi piu' worker
    che fallivano insieme ritentavano tutti nello stesso istante."""

    def _provider(self, **overrides) -> ApiSportsProvider:
        cfg = dict(
            base_url="https://esempio.test",
            api_key="chiave-finta",
            max_retries=4,
            retry_backoff_seconds=2.0,
            retry_backoff_max_seconds=5.0,
        )
        cfg.update(overrides)
        return ApiSportsProvider(config=ApiSportsProviderConfig(**cfg))

    def test_delay_cresce_in_modo_esponenziale_entro_i_limiti(self):
        provider = self._provider()
        for attempt, step in ((1, 2.0), (2, 4.0), (3, 5.0), (6, 5.0)):
            for _ in range(20):
                delay = provider._retry_delay(attempt)
                self.assertGreaterEqual(delay, step / 2)
                self.assertLessEqual(delay, step)

    def test_backoff_zero_non_dorme_mai(self):
        provider = self._provider(retry_backoff_seconds=0)
        risposta = _risposta(status_code=429, headers={"Retry-After": "30"})
        self.assertEqual(provider._retry_delay(1), 0.0)
        self.assertEqual(provider._retry_delay(3, risposta), 0.0)

    def test_retry_after_del_provider_vince_ma_ha_un_tetto(self):
        provider = self._provider()
        self.assertEqual(provider._retry_delay(1, _risposta(status_code=429, headers={"Retry-After": "12"})), 12.0)
        self.assertEqual(provider._retry_delay(1, _risposta(status_code=429, headers={"Retry-After": "99999"})), 120.0)
        # Valore non numerico (es. data HTTP): ignorato, resta il backoff.
        delay = provider._retry_delay(1, _risposta(status_code=429, headers={"Retry-After": "Wed, 21 Oct 2026"}))
        self.assertLessEqual(delay, 2.0)

    def test_il_retry_dorme_con_il_delay_calcolato(self):
        provider = self._provider(max_retries=3)
        with mock.patch("requests.get", return_value=_risposta(status_code=503)), \
                mock.patch("src.service_ia.pre_processing.api_sports_provider.time.sleep") as sleep:
            with self.assertRaises(ApiSportsUnavailableError):
                provider.request(path="fixtures")
        attese = [call.args[0] for call in sleep.call_args_list]
        self.assertEqual(len(attese), 2)  # nessuna attesa dopo l'ultimo tentativo
        self.assertTrue(1.0 <= attese[0] <= 2.0)
        self.assertTrue(2.0 <= attese[1] <= 4.0)


class TestCircuitBreaker(unittest.TestCase):
    """Dopo N richieste consecutive fallite per indisponibilita' il provider
    smette di essere chiamato per un periodo: un import di 18 campionati con
    API-Sports giu' non deve consumare `max_retries` timeout per ogni lega."""

    def _provider(self, clock, threshold=2) -> ApiSportsProvider:
        provider = ApiSportsProvider(
            config=ApiSportsProviderConfig(
                base_url="https://esempio.test",
                api_key="chiave-finta",
                max_retries=2,
                retry_backoff_seconds=0,
                circuit_failure_threshold=threshold,
                circuit_recovery_seconds=60,
            )
        )
        provider._breaker._clock = clock
        return provider

    def test_si_apre_dopo_la_soglia_e_fallisce_senza_chiamare_il_provider(self):
        now = [1000.0]
        provider = self._provider(lambda: now[0])
        with mock.patch("requests.get", return_value=_risposta(status_code=503)) as get:
            for _ in range(2):
                with self.assertRaises(ApiSportsUnavailableError):
                    provider.request(path="fixtures")
            self.assertEqual(get.call_count, 4)
            self.assertEqual(provider.circuit_state, "open")
            with self.assertRaises(ApiSportsCircuitOpenError):
                provider.request(path="fixtures")
            self.assertEqual(get.call_count, 4)

    def test_circuit_open_e_una_unavailable_per_i_chiamanti_esistenti(self):
        self.assertTrue(issubclass(ApiSportsCircuitOpenError, ApiSportsUnavailableError))

    def test_dopo_il_recovery_una_sonda_riuscita_richiude(self):
        now = [1000.0]
        provider = self._provider(lambda: now[0])
        with mock.patch("requests.get", return_value=_risposta(status_code=503)):
            for _ in range(2):
                with self.assertRaises(ApiSportsUnavailableError):
                    provider.request(path="fixtures")
        now[0] += 61
        self.assertEqual(provider.circuit_state, "half_open")
        buona = _risposta(payload={"response": [{"fixture": {"id": 1}}]})
        with mock.patch("requests.get", return_value=buona):
            self.assertEqual(provider.request(path="fixtures"), [{"fixture": {"id": 1}}])
        self.assertEqual(provider.circuit_state, "closed")

    def test_sonda_fallita_riapre_subito(self):
        now = [1000.0]
        provider = self._provider(lambda: now[0])
        with mock.patch("requests.get", return_value=_risposta(status_code=503)) as get:
            for _ in range(2):
                with self.assertRaises(ApiSportsUnavailableError):
                    provider.request(path="fixtures")
            now[0] += 61
            with self.assertRaises(ApiSportsUnavailableError):
                provider.request(path="fixtures")
            self.assertEqual(provider.circuit_state, "open")
            chiamate = get.call_count
            with self.assertRaises(ApiSportsCircuitOpenError):
                provider.request(path="fixtures")
            self.assertEqual(get.call_count, chiamate)

    def test_un_successo_azzera_il_conteggio_dei_fallimenti(self):
        provider = self._provider(lambda: 1000.0, threshold=2)
        buona = _risposta(payload={"response": []})
        with mock.patch("requests.get", side_effect=[_risposta(status_code=503)] * 2 + [buona] + [_risposta(status_code=503)] * 2):
            with self.assertRaises(ApiSportsUnavailableError):
                provider.request(path="fixtures")
            self.assertEqual(provider.request(path="fixtures"), [])
            with self.assertRaises(ApiSportsUnavailableError):
                provider.request(path="fixtures")
        self.assertEqual(provider.circuit_state, "closed")

    def test_errori_di_richiesta_e_quota_esaurita_non_aprono_il_breaker(self):
        provider = self._provider(lambda: 1000.0, threshold=1)
        rifiuto = _risposta(payload={"errors": {"plan": "stagione non inclusa nel piano"}})
        with mock.patch("requests.get", return_value=rifiuto):
            with self.assertRaises(ApiSportsUnavailableError):
                provider.request(path="fixtures")
        self.assertEqual(provider.circuit_state, "closed")

        quota = _risposta(payload={"errors": {"requests": "You have reached the request limit for the day"}})
        with mock.patch("requests.get", return_value=quota), \
                mock.patch("src.service_ia.pre_processing.api_sports_provider.record_status_snapshot"), \
                mock.patch("src.service_ia.pre_processing.api_sports_provider.record_quota_snapshot"):
            with self.assertRaises(ApiSportsQuotaExceededError):
                provider.request(path="fixtures")
        self.assertEqual(provider.circuit_state, "closed")

    def test_soglia_zero_disattiva_il_breaker(self):
        provider = self._provider(lambda: 1000.0, threshold=0)
        with mock.patch("requests.get", return_value=_risposta(status_code=503)) as get:
            for _ in range(4):
                with self.assertRaises(ApiSportsUnavailableError):
                    provider.request(path="fixtures")
        self.assertEqual(get.call_count, 8)
        self.assertEqual(provider.circuit_state, "closed")


if __name__ == "__main__":
    unittest.main()
