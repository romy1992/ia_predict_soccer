import unittest
from typing import Any
from unittest import mock

from src.service_ia.model.match import Match, Odds, Statistics
from src.service_ia.pre_processing.settlement_service import (
    NoOpSettlementHook,
    SettlementService,
    apply_settlement_metadata,
    evaluate_match_completeness,
)


class FakeMatchRepo:
    def __init__(self, matches):
        self.matches = matches
        self.saved: list[Match] = []
        self.finestre: list[dict[str, Any]] = []

    def search_filter(self, filters: dict[str, Any]):
        return self.matches

    def search_by_date_window(self, **kwargs):
        """Dal 2026-09-15 `run_settlement` filtra la finestra di date in SQL
        invece di caricare tutti i match finali di tutte le stagioni (44.983
        righe) e scartarne il 99% in Python. Qui si registrano gli argomenti
        ricevuti, cosi' un test puo' verificare che la finestra sia davvero
        arrivata al DB."""
        self.finestre.append(kwargs)
        return self.matches

    def save(self, match: Match):
        self.saved.append(match)


class FakeSnapshotRepo:
    def __init__(self, counts_by_fixture: dict[int, int]):
        self.counts_by_fixture = counts_by_fixture

    def list_for_fixture(self, fixture_id: int):
        count = self.counts_by_fixture.get(int(fixture_id), 0)
        return [object() for _ in range(count)]


class CountingHook(NoOpSettlementHook):
    def __init__(self):
        self.calls = 0

    def on_prediction_settled(self, match: Match, settlement_details: dict[str, Any]) -> None:
        self.calls += 1


def _complete_match() -> Match:
    match = Match(
        id_match_fk="m1",
        id_fixture=1001,
        id_team_home=10,
        id_team_away=20,
        name_home="A",
        name_away="B",
        date_match="2026-09-01T18:00:00+00:00",
        current_league=135,
        season=2026,
        status="FT",
    )
    match.statistics = [
        Statistics(statistics_team_id=10, score_ft=2, score_ht=1),
        Statistics(statistics_team_id=20, score_ft=1, score_ht=0),
    ]
    match.odds = [Odds(h2h={"home_Book": "1.80"})]
    return match


def _incomplete_match() -> Match:
    match = Match(
        id_match_fk="m2",
        id_fixture=1002,
        id_team_home=11,
        id_team_away=21,
        name_home="C",
        name_away="D",
        date_match="2026-09-01T20:00:00+00:00",
        current_league=135,
        season=2026,
        status="FT",
    )
    match.statistics = []
    match.odds = []
    return match


class TestSettlementService(unittest.TestCase):
    def test_evaluate_match_completeness(self):
        complete = evaluate_match_completeness(_complete_match(), snapshot_count=0)
        self.assertEqual(complete["completeness_status"], "complete")

        incomplete = evaluate_match_completeness(_incomplete_match(), snapshot_count=0)
        self.assertEqual(incomplete["completeness_status"], "incomplete")

    def test_apply_settlement_metadata_idempotent(self):
        match = _complete_match()
        details = evaluate_match_completeness(match, snapshot_count=1)

        changed_1, _ = apply_settlement_metadata(match, details, settled_at_iso="2026-09-02T10:00:00+00:00")
        changed_2, _ = apply_settlement_metadata(match, details, settled_at_iso="2026-09-02T10:05:00+00:00")

        self.assertTrue(changed_1)
        self.assertFalse(changed_2)

    def test_run_settlement_updates_then_unchanged(self):
        matches = [_complete_match(), _incomplete_match()]
        match_repo = FakeMatchRepo(matches=matches)
        snapshot_repo = FakeSnapshotRepo(counts_by_fixture={1001: 2, 1002: 0})
        hook = CountingHook()

        service = SettlementService(
            match_repo=match_repo,
            snapshot_repo=snapshot_repo,
            import_runner=lambda **kwargs: {"ok": True, "params": kwargs},
        )

        report_1 = service.run_settlement(
            from_date="2026-09-01",
            to_date="2026-09-01",
            seasons=[2026],
            leagues=[135],
            provider=object(),
            hook=hook,
        )
        report_2 = service.run_settlement(
            from_date="2026-09-01",
            to_date="2026-09-01",
            seasons=[2026],
            leagues=[135],
            provider=object(),
            hook=hook,
        )

        self.assertEqual(report_1["updated"], 2)
        self.assertEqual(report_2["unchanged"], 2)
        self.assertEqual(hook.calls, 1)


class TestSettlementConQuotaEsaurita(unittest.TestCase):
    """`data_settlement` non viene messo in pausa a quota esaurita (deve
    continuare a riconciliare ledger e schedine), ma il suo primo passo e' un
    import da API-Sports. A quota esaurita OGGI quell'import va saltato: la
    riconciliazione prosegue sui dati gia' a DB."""

    def _service(self, *, exhausted, import_calls, match_repo=None):
        def import_runner(**kwargs):
            import_calls.append(kwargs)
            return {"ok": True}

        return SettlementService(
            match_repo=match_repo or FakeMatchRepo(matches=[_complete_match(), _incomplete_match()]),
            snapshot_repo=FakeSnapshotRepo(counts_by_fixture={1001: 2, 1002: 0}),
            import_runner=import_runner,
            quota_exhausted_check=exhausted,
        )

    def test_quota_esaurita_salta_import_ma_riconcilia_dal_db(self):
        import_calls: list[dict[str, Any]] = []
        match_repo = FakeMatchRepo(matches=[_complete_match(), _incomplete_match()])
        service = self._service(exhausted=lambda: True, import_calls=import_calls, match_repo=match_repo)
        hook = CountingHook()

        report = service.run_settlement(
            from_date="2026-09-01", to_date="2026-09-01", seasons=[2026], leagues=[135], hook=hook
        )

        self.assertEqual(import_calls, [])
        self.assertTrue(report["provider_import_skipped"])
        self.assertEqual(
            report["import_report"], {"skipped": True, "reason": "api_sports_quota_exhausted_today"}
        )
        # La parte locale gira comunque: finestra interrogata, match aggiornati, hook chiamato.
        self.assertEqual(len(match_repo.finestre), 1)
        self.assertEqual(report["final_matches_seen"], 2)
        self.assertEqual(report["updated"], 2)
        self.assertEqual(hook.calls, 1)

    def test_quota_esaurita_non_costruisce_il_provider(self):
        import_calls: list[dict[str, Any]] = []
        service = self._service(exhausted=lambda: True, import_calls=import_calls)
        with mock.patch(
            "src.service_ia.pre_processing.settlement_service.get_api_sports_provider"
        ) as get_provider:
            service.run_settlement(from_date="2026-09-01", to_date="2026-09-01")
        get_provider.assert_not_called()

    def test_quota_disponibile_importa_come_prima(self):
        import_calls: list[dict[str, Any]] = []
        service = self._service(exhausted=lambda: False, import_calls=import_calls)
        provider = object()

        report = service.run_settlement(
            from_date="2026-09-01", to_date="2026-09-01", seasons=[2026], leagues=[135], provider=provider
        )

        self.assertEqual(len(import_calls), 1)
        self.assertIs(import_calls[0]["provider"], provider)
        self.assertEqual(import_calls[0]["statuses"], "FT-AET-PEN-ABD")
        self.assertFalse(report["provider_import_skipped"])
        self.assertEqual(report["import_report"], {"ok": True})

    def test_stato_quota_illeggibile_non_blocca_il_settlement(self):
        import_calls: list[dict[str, Any]] = []

        def esplode():
            raise RuntimeError("bucket non raggiungibile")

        service = self._service(exhausted=esplode, import_calls=import_calls)
        report = service.run_settlement(from_date="2026-09-01", to_date="2026-09-01", provider=object())

        self.assertEqual(len(import_calls), 1)
        self.assertFalse(report["provider_import_skipped"])


if __name__ == "__main__":
    unittest.main()
