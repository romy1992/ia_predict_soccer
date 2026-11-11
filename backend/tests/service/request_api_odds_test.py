"""Test per `base_api_odds` (helper legacy verso The Odds API).

Tre difetti corretti: nessun timeout sulla chiamata HTTP, `ZeroDivisionError`
quando la risposta non porta gli header di utilizzo (`X-Requests-Used` /
`X-Requests-Remaining` assenti o entrambi a zero) e doppio parsing del body.
Il comportamento sui casi gia' gestiti resta identico."""

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


if __name__ == "__main__":
    unittest.main()
