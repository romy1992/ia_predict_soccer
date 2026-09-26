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
    ApiSportsProvider,
    ApiSportsProviderConfig,
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


if __name__ == "__main__":
    unittest.main()
