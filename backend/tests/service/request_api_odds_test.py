"""Test per `base_api_odds` (helper legacy verso The Odds API).

Tre difetti corretti: nessun timeout sulla chiamata HTTP, `ZeroDivisionError`
quando la risposta non porta gli header di utilizzo (`X-Requests-Used` /
`X-Requests-Remaining` assenti o entrambi a zero) e doppio parsing del body.
Il comportamento sui casi gia' gestiti resta identico."""

import importlib
import logging
import unittest
from unittest import mock

from src.service_ia.utility import request_api


def _risposta(status_code=200, payload=None, headers=None):
    risposta = mock.Mock()
    risposta.status_code = status_code
    risposta.headers = headers or {}
    risposta.json.return_value = payload
    return risposta


class TestBaseApiOdds(unittest.TestCase):
    def _call(self, risposta, **kwargs):
        with mock.patch("src.service_ia.config.secret_store.get_secret", return_value="chiave-finta"), \
                mock.patch.object(request_api.requests, "get", return_value=risposta) as get:
            result = request_api.base_api_odds(**kwargs)
        return result, get

    def test_passa_un_timeout_esplicito(self):
        _, get = self._call(_risposta(payload=[{"id": "a"}], headers={"X-Requests-Used": "1", "X-Requests-Remaining": "99"}))
        self.assertEqual(get.call_args.kwargs["timeout"], request_api.API_ODDS_TIMEOUT_SECONDS)
        self.assertGreater(request_api.API_ODDS_TIMEOUT_SECONDS, 0)

    def test_header_di_utilizzo_assenti_non_dividono_per_zero(self):
        result, _ = self._call(_risposta(payload=[{"id": "a"}], headers={}))
        self.assertEqual(result, [{"id": "a"}])

    def test_header_a_zero_non_dividono_per_zero(self):
        result, _ = self._call(
            _risposta(payload=[{"id": "a"}], headers={"X-Requests-Used": "0", "X-Requests-Remaining": "0"})
        )
        self.assertEqual(result, [{"id": "a"}])

    def test_il_body_viene_letto_una_volta_sola(self):
        risposta = _risposta(payload=[{"id": "a"}], headers={"X-Requests-Used": "10", "X-Requests-Remaining": "90"})
        self._call(risposta)
        self.assertEqual(risposta.json.call_count, 1)

    def test_oltre_il_99_percento_si_ferma_come_prima(self):
        risposta = _risposta(payload=[{"id": "a"}], headers={"X-Requests-Used": "995", "X-Requests-Remaining": "5"})
        with self.assertRaises(Exception) as ctx:
            self._call(risposta)
        self.assertIn("Stop API", str(ctx.exception))

    def test_status_non_200_resta_lista_vuota_senza_leggere_il_body(self):
        risposta = _risposta(status_code=500, payload={"message": "errore"}, headers={})
        result, _ = self._call(risposta)
        self.assertEqual(result, [])
        risposta.json.assert_not_called()

    def test_body_vuoto_resta_lista_vuota(self):
        result, _ = self._call(_risposta(payload=None, headers={"X-Requests-Used": "1", "X-Requests-Remaining": "9"}))
        self.assertEqual(result, [])

    def test_errore_di_rete_non_espone_la_chiave_nel_messaggio(self):
        errore = request_api.requests.ConnectionError(
            "Max retries exceeded with url: /v4/sports?apiKey=SEGRETO-DI-PROVA"
        )
        with mock.patch("src.service_ia.config.secret_store.get_secret", return_value="SEGRETO-DI-PROVA"), \
                mock.patch.object(request_api.requests, "get", side_effect=errore):
            with self.assertRaises(request_api.requests.RequestException) as ctx:
                request_api.base_api_odds(path="sports")
        self.assertNotIn("SEGRETO-DI-PROVA", str(ctx.exception))
        self.assertNotIn("SEGRETO-DI-PROVA", repr(ctx.exception.__cause__))


class _Raccoglitore(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messaggi: list[str] = []

    def emit(self, record):
        self.messaggi.append(record.getMessage())


class TestChiaveFuoriDaiLogDiUrllib3(unittest.TestCase):
    """urllib3 logga a DEBUG la riga di richiesta con la query string, quindi
    con `apiKey=<chiave>` (verificato su loopback il 2026-11-17); questo
    modulo imposta il root logger a DEBUG. Il filtro installato all'import
    maschera il valore. Nessuna rete: il record e' emesso a mano con lo stesso
    formato di urllib3."""

    FORMATO_URLLIB3 = '%s://%s:%s "%s %s %s" %s %s'

    def _emetti(self, url: str) -> list[str]:
        logger = logging.getLogger("urllib3.connectionpool")
        raccoglitore = _Raccoglitore()
        livello = logger.level
        logger.addHandler(raccoglitore)
        logger.setLevel(logging.DEBUG)
        try:
            logger.debug(self.FORMATO_URLLIB3, "https", "api.the-odds-api.com", 443, "GET", url, "HTTP/1.1", 200, None)
        finally:
            logger.removeHandler(raccoglitore)
            logger.setLevel(livello)
        return raccoglitore.messaggi

    def test_la_chiave_viene_mascherata_e_il_resto_della_riga_resta(self):
        messaggi = self._emetti("/v4/sports/soccer_epl/odds?regions=eu&apiKey=SEGRETO-DI-PROVA&markets=h2h")
        self.assertEqual(len(messaggi), 1)
        self.assertNotIn("SEGRETO-DI-PROVA", messaggi[0])
        self.assertIn("apiKey=***", messaggi[0])
        self.assertIn("regions=eu", messaggi[0])
        self.assertIn("markets=h2h", messaggi[0])
        self.assertIn('" 200', messaggi[0])

    def test_le_righe_senza_chiave_restano_identiche(self):
        messaggi = self._emetti("/fixtures?league=135&season=2026")
        self.assertEqual(messaggi, ['https://api.the-odds-api.com:443 "GET /fixtures?league=135&season=2026 HTTP/1.1" 200 None'])

    def test_il_filtro_viene_installato_una_volta_sola(self):
        request_api._install_api_key_redaction()
        request_api._install_api_key_redaction()
        filtri = [f for f in logging.getLogger("urllib3.connectionpool").filters
                  if type(f).__name__ == "_RedactApiKeyFilter" and type(f).__module__ == request_api.__name__]
        self.assertEqual(len(filtri), 1)


class TestHelperLegacy(unittest.TestCase):
    """La copia storica `backend/service_ia/utility/request_api.py` aveva lo
    stesso difetto: stessa correzione, stesso test."""

    def test_errore_di_rete_non_espone_la_chiave_e_passa_il_timeout(self):
        legacy = importlib.import_module("service_ia.utility.request_api")
        errore = legacy.requests.ConnectionError("Max retries exceeded with url: /v4/sports?apiKey=SEGRETO-LEGACY")
        with mock.patch.object(legacy, "API_ODDS_KEY", "SEGRETO-LEGACY"), \
                mock.patch.object(legacy.requests, "get", side_effect=errore) as get:
            with self.assertRaises(legacy.requests.RequestException) as ctx:
                legacy.base_api_odds(path="sports")
        self.assertNotIn("SEGRETO-LEGACY", str(ctx.exception))
        self.assertIsNone(ctx.exception.__cause__)
        self.assertEqual(get.call_args.kwargs["timeout"], legacy.API_ODDS_TIMEOUT_SECONDS)
        self.assertTrue(any(type(f).__name__ == "_RedactApiKeyFilter"
                            for f in logging.getLogger("urllib3.connectionpool").filters))


if __name__ == "__main__":
    unittest.main()
