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

from src.service_ia.pre_processing.api_sports_provider import ApiSportsProvider


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


if __name__ == "__main__":
    unittest.main()

