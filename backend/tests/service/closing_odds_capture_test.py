"""Cattura delle quote di chiusura (`closing_odds_capture`).

DB SQLite in memoria, bucket fake (conftest), provider e repository snapshot
finti: nessuna chiamata reale ad API-Sports ne' a Postgres.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.oracle.backtest.clv import REASON_CLOSING_NOT_AFTER_BET, build_clv_result
from src.oracle.ledger import closing_odds_capture_service as service_module
from src.oracle.ledger.closing_odds_capture_service import (
    ClosingCaptureRegistry,
    ClosingOddsCaptureService,
)
from src.service_ia.model.match import Base, Match, OddsSnapshot, PredictionLedger
from src.service_ia.pre_processing.api_sports_provider import (
    ApiSportsCircuitOpenError,
    ApiSportsQuotaExceededError,
    ApiSportsUnavailableError,
)
from src.storage import bucket_store

KICKOFF = datetime(2026, 11, 17, 20, 0, tzinfo=timezone.utc)


def _payload(update: str | None = "2026-11-17T19:50:00+00:00", home: str = "2.10") -> list[dict]:
    row = {
        "bookmakers": [
            {
                "name": "Bet365",
                "bets": [
                    {
                        "id": 1,
                        "name": "Match Winner",
                        "values": [
                            {"value": "Home", "odd": home},
                            {"value": "Draw", "odd": "3.40"},
                            {"value": "Away", "odd": "3.60"},
                        ],
                    }
                ],
            }
        ]
    }
    if update is not None:
        row["update"] = update
    return [row]


class FakeProvider:
    """`responses[fixture_id]`: payload o eccezione (sempre); `failures_once`:
    eccezioni sollevate una volta ciascuna, poi il payload di default."""

    def __init__(self, responses=None, failures_once=None):
        self.responses = responses or {}
        self.failures_once = {key: list(value) for key, value in (failures_once or {}).items()}
        self.calls: list[int] = []

    def get_fixture_odds(self, fixture_id: int):
        self.calls.append(fixture_id)
        if self.failures_once.get(fixture_id):
            raise self.failures_once[fixture_id].pop(0)
        response = self.responses.get(fixture_id, _payload())
        if isinstance(response, BaseException):
            raise response
        return response


class FakeSnapshotRepo:
    def __init__(self, fail_save: bool = False):
        self.rows: dict[str, OddsSnapshot] = {}
        self.fail_save = fail_save

    def save_many(self, snapshots):
        if self.fail_save:
            raise RuntimeError("db down")
        for snapshot in snapshots:
            self.rows[snapshot.id_snapshot] = snapshot

    def latest_captured_at(self, fixture_id):
        values = [row.captured_at for row in self.rows.values() if row.fixture_id == fixture_id]
        return max(values) if values else None

    def list_for_fixture(self, fixture_id, market=None, period=None, line=None):
        rows = [
            row
            for row in self.rows.values()
            if row.fixture_id == fixture_id
            and (market is None or row.market == market)
            and (period is None or row.period == period)
            and (line is None or row.line == line)
        ]
        return sorted(rows, key=lambda row: row.captured_at)


class Clock:
    def __init__(self, value: datetime):
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def at(self, minutes_before_kickoff: float, kickoff: datetime = KICKOFF) -> "Clock":
        self.value = kickoff - timedelta(minutes=minutes_before_kickoff)
        return self


@pytest.fixture()
def session_factory():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


def _add_match(session_factory, fixture_id: int, kickoff: datetime, status: str = "NS",
               cohort: str | None = "official_paper", decision: str = "PLAY") -> None:
    with session_factory() as session:
        session.add(
            Match(
                id_match_fk=f"m-{fixture_id}",
                id_fixture=fixture_id,
                status=status,
                date_match=kickoff.isoformat(),
            )
        )
        if cohort is not None:
            session.add(
                PredictionLedger(
                    fixture_id=fixture_id,
                    market="h2h",
                    outcome="Home",
                    decision=decision,
                    cohort=cohort,
                    kickoff_at=kickoff,
                    odd=2.0,
                )
            )
        session.commit()


def _service(session_factory, clock, provider=None, repo=None, quota_exhausted=False, registry=None):
    return ClosingOddsCaptureService(
        provider=provider if provider is not None else FakeProvider(),
        snapshot_repo=repo if repo is not None else FakeSnapshotRepo(),
        registry=registry or ClosingCaptureRegistry(),
        session_factory=session_factory,
        now_fn=clock,
        quota_exhausted_fn=lambda: quota_exhausted,
    )


# --- Selezione delle partite (solo DB) -------------------------------------


def test_seleziona_solo_ns_con_giocate_ufficiali_nei_prossimi_15_minuti(session_factory):
    now = KICKOFF - timedelta(minutes=30)
    _add_match(session_factory, 1, now + timedelta(minutes=12), decision="PLAY")
    _add_match(session_factory, 2, now + timedelta(minutes=3), decision="BORDERLINE")
    _add_match(session_factory, 3, now + timedelta(minutes=15), decision="NO BET")
    _add_match(session_factory, 4, now + timedelta(minutes=5), status="FT")
    _add_match(session_factory, 5, now + timedelta(minutes=5), cohort=None)
    _add_match(session_factory, 6, now + timedelta(minutes=5), cohort="manual")
    _add_match(session_factory, 7, now + timedelta(minutes=20))
    _add_match(session_factory, 8, now - timedelta(minutes=1))
    provider = FakeProvider()
    service = _service(session_factory, Clock(now), provider=provider)

    candidates = service.candidates(now)

    assert [c.fixture_id for c in candidates] == [2, 1, 3]
    assert candidates[0].id_match == "m-2"
    assert provider.calls == []


def test_partita_con_piu_giocate_ufficiali_e_una_sola_candidata(session_factory):
    _add_match(session_factory, 1, KICKOFF)
    with session_factory() as session:
        session.add(PredictionLedger(fixture_id=1, market="dc", outcome="1X", decision="NO BET",
                                     cohort="official_paper", kickoff_at=KICKOFF))
        session.commit()
    clock = Clock(KICKOFF).at(12)
    assert len(_service(session_factory, clock).candidates(clock())) == 1


def test_nessuna_candidata_nessuna_lettura_del_registro(session_factory):
    report = _service(session_factory, Clock(KICKOFF).at(12)).capture()
    assert report["candidates"] == 0
    assert report["calls"] == 0


# --- Finestre e deduplica -----------------------------------------------------


def test_al_massimo_una_chiamata_per_momento_e_due_per_partita(session_factory):
    _add_match(session_factory, 1, KICKOFF)
    clock = Clock(KICKOFF)
    provider = FakeProvider()
    repo = FakeSnapshotRepo()

    reports = [
        _service(session_factory, clock.at(minutes), provider=provider, repo=repo).capture()
        for minutes in (14, 11, 8, 4, 1)
    ]

    assert provider.calls == [1, 1]
    assert [r["calls"] for r in reports] == [1, 0, 0, 1, 0]
    assert reports[1]["skipped"]["already_captured"] == 1
    assert reports[2]["skipped"]["outside_window"] == 1
    assert reports[4]["skipped"]["already_captured"] == 1
    assert reports[0]["captures"][0]["moment"] == "early"
    assert reports[3]["captures"][0]["moment"] == "late"
    state = bucket_store.get_json("best_models/closing_odds_capture_state.json")
    (entry,) = state.values()
    assert set(entry["moments"]) == {"early", "late"}
    assert entry["moments"]["early"]["status"] == "ok"


def test_early_saltato_dalla_cadenza_viene_recuperato_fra_10_e_5_minuti(session_factory):
    _add_match(session_factory, 1, KICKOFF)
    clock = Clock(KICKOFF)
    provider = FakeProvider()

    first = _service(session_factory, clock.at(8), provider=provider).capture()
    second = _service(session_factory, clock.at(3), provider=provider).capture()

    assert first["captures"][0]["moment"] == "early"
    assert second["captures"][0]["moment"] == "late"
    assert provider.calls == [1, 1]


def test_partita_rinviata_riapre_le_finestre_al_nuovo_kickoff(session_factory):
    _add_match(session_factory, 1, KICKOFF)
    clock = Clock(KICKOFF)
    provider = FakeProvider()
    _service(session_factory, clock.at(3), provider=provider).capture()

    nuovo = KICKOFF + timedelta(hours=2)
    with session_factory() as session:
        session.query(Match).filter(Match.id_fixture == 1).update({"date_match": nuovo.isoformat()})
        session.commit()
    _service(session_factory, clock.at(3, kickoff=nuovo), provider=provider).capture()

    assert provider.calls == [1, 1]


def test_registro_condiviso_fra_giri_diversi(session_factory):
    _add_match(session_factory, 1, KICKOFF)
    clock = Clock(KICKOFF).at(13)
    provider = FakeProvider()

    _service(session_factory, clock, provider=provider).capture()
    report = _service(session_factory, clock, provider=provider).capture()

    assert provider.calls == [1]
    assert report["skipped"]["already_captured"] == 1


def test_momento_preso_da_un_altro_processo_fra_lettura_e_prenotazione(session_factory):
    _add_match(session_factory, 1, KICKOFF)
    clock = Clock(KICKOFF).at(13)
    provider = FakeProvider()
    registry = ClosingCaptureRegistry()
    service = _service(session_factory, clock, provider=provider, registry=registry)
    original_read = registry.read
    letture = {"n": 0}

    def read_then_other_process_claims():
        state = original_read()
        letture["n"] += 1
        if letture["n"] == 1:
            ClosingCaptureRegistry().claim(service.candidates(clock())[0], "early", clock())
        return state

    registry.read = read_then_other_process_claims
    report = service.capture()

    assert provider.calls == []
    assert report["skipped"]["already_captured"] == 1


def test_voci_vecchie_del_registro_rimosse(session_factory):
    bucket_store.put_json(
        "best_models/closing_odds_capture_state.json",
        {"9|2026-11-10T20:00:00+00:00": {"fixture_id": 9, "kickoff_at": "2026-11-10T20:00:00+00:00", "moments": {"late": {}}}},
    )
    _add_match(session_factory, 1, KICKOFF)
    _service(session_factory, Clock(KICKOFF).at(12)).capture()

    state = bucket_store.get_json("best_models/closing_odds_capture_state.json")
    assert list(state) == [f"1|{KICKOFF.isoformat()}"]


# --- Quota, errori del provider, registro -----------------------------------


def test_quota_esaurita_nessuna_chiamata_e_nessuna_prenotazione(session_factory):
    _add_match(session_factory, 1, KICKOFF)
    _add_match(session_factory, 2, KICKOFF + timedelta(minutes=1))
    provider = FakeProvider()

    report = _service(session_factory, Clock(KICKOFF).at(12), provider=provider, quota_exhausted=True).capture()

    assert provider.calls == []
    assert report["calls"] == 0
    assert report["skipped"]["quota_exhausted"] == 2
    assert bucket_store.get_json("best_models/closing_odds_capture_state.json") is None


def test_quota_esaurita_durante_il_giro_ferma_le_chiamate_successive(session_factory):
    _add_match(session_factory, 1, KICKOFF)
    _add_match(session_factory, 2, KICKOFF + timedelta(minutes=1))
    provider = FakeProvider({1: ApiSportsQuotaExceededError("limit for the day")})

    report = _service(session_factory, Clock(KICKOFF).at(12), provider=provider).capture()

    assert provider.calls == [1]
    assert report["calls"] == 1
    assert report["errors"]["quota_exhausted"] == 1
    assert report["skipped"]["quota_exhausted"] == 1
    assert report["error_details"][0]["fixture_id"] == 1


def test_errore_del_provider_contato_e_momento_non_ripetuto(session_factory):
    _add_match(session_factory, 1, KICKOFF)
    clock = Clock(KICKOFF)
    provider = FakeProvider(failures_once={1: [ApiSportsUnavailableError("503 x3")]})

    first = _service(session_factory, clock.at(14), provider=provider).capture()
    second = _service(session_factory, clock.at(11), provider=provider).capture()

    assert first["errors"]["provider"] == 1
    assert first["calls"] == 1
    assert first["captured"] == 0
    assert second["skipped"]["already_captured"] == 1
    assert provider.calls == [1]


def test_errore_imprevisto_del_provider_non_propagato(session_factory):
    _add_match(session_factory, 1, KICKOFF)
    _add_match(session_factory, 2, KICKOFF + timedelta(minutes=1))
    provider = FakeProvider({1: ValueError("payload strano")})

    report = _service(session_factory, Clock(KICKOFF).at(12), provider=provider).capture()

    assert report["errors"]["provider"] == 1
    assert report["captured"] == 1
    assert provider.calls == [1, 2]


def test_circuit_breaker_aperto_libera_il_momento(session_factory):
    _add_match(session_factory, 1, KICKOFF)
    clock = Clock(KICKOFF)
    provider = FakeProvider(failures_once={1: [ApiSportsCircuitOpenError("aperto")]})

    first = _service(session_factory, clock.at(14), provider=provider).capture()
    second = _service(session_factory, clock.at(11), provider=provider).capture()

    assert first["calls"] == 0
    assert first["errors"]["provider"] == 1
    assert second["captured"] == 1
    assert provider.calls == [1, 1]


def test_errore_di_salvataggio_contato(session_factory):
    _add_match(session_factory, 1, KICKOFF)
    report = _service(session_factory, Clock(KICKOFF).at(12), repo=FakeSnapshotRepo(fail_save=True)).capture()

    assert report["errors"]["save"] == 1
    assert report["snapshots_saved"] == 0


def test_risposta_vuota_contata(session_factory):
    _add_match(session_factory, 1, KICKOFF)
    report = _service(session_factory, Clock(KICKOFF).at(12), provider=FakeProvider({1: []})).capture()

    assert report["calls"] == 1
    assert report["empty_responses"] == 1
    assert report["captured"] == 0


def test_registro_illeggibile_nessuna_chiamata(session_factory):
    _add_match(session_factory, 1, KICKOFF)
    registry = ClosingCaptureRegistry()
    registry.read = lambda: (_ for _ in ()).throw(RuntimeError("bucket giu'"))
    provider = FakeProvider()

    report = _service(session_factory, Clock(KICKOFF).at(12), provider=provider, registry=registry).capture()

    assert provider.calls == []
    assert report["errors"]["registry"] == 1


def test_non_usa_il_lock_degli_import():
    assert not hasattr(service_module, "job_lock")
    assert not hasattr(service_module, "LOCK_IMPORT_MATCH")


# --- Snapshot e chiusura per il CLV -----------------------------------------


def test_snapshot_con_update_del_provider_e_update_invariato_segnalato(session_factory):
    _add_match(session_factory, 1, KICKOFF)
    clock = Clock(KICKOFF)
    repo = FakeSnapshotRepo()
    provider = FakeProvider({1: _payload(update="2026-11-17T19:47:00+00:00")})

    early = _service(session_factory, clock.at(12), provider=provider, repo=repo).capture()
    late = _service(session_factory, clock.at(3), provider=provider, repo=repo).capture()

    assert early["snapshots_saved"] == 3
    assert early["captures"][0]["provider_update_changed"] is True
    assert late["provider_update_unchanged"] == 1
    assert late["captures"][0]["provider_update_changed"] is False
    assert len(repo.rows) == 3
    assert {row.captured_at for row in repo.rows.values()} == {datetime(2026, 11, 17, 19, 47, tzinfo=timezone.utc)}
    assert all(capture["captured_at_before_kickoff"] for capture in early["captures"] + late["captures"])


def _bet_snapshot(captured_at: datetime) -> OddsSnapshot:
    return OddsSnapshot(
        id_snapshot=f"bet-{captured_at.isoformat()}", id_match="m-1", fixture_id=1, bookmaker="Bet365",
        market="h2h", period="full_time", line=None, outcome="Home", odd=2.0, captured_at=captured_at,
        source="api_sports",
    )


def _clv(repo: FakeSnapshotRepo, kickoff: datetime, bet_odds_at: datetime):
    return build_clv_result(
        market="1x2", outcome="Home", odd_at_bet=2.0, p_fair_at_bet=0.45, fixture_id=1, kickoff_at=kickoff,
        snapshot_repo=repo, bet_captured_at=bet_odds_at + timedelta(minutes=2), bet_odds_captured_at=bet_odds_at,
    )


def test_chiusura_valida_per_il_clv_quando_il_provider_aggiorna(session_factory):
    _add_match(session_factory, 1, KICKOFF)
    bet_odds_at = KICKOFF - timedelta(minutes=60)
    repo = FakeSnapshotRepo()
    repo.save_many([_bet_snapshot(bet_odds_at)])
    provider = FakeProvider({1: _payload(update="2026-11-17T19:52:00+00:00", home="1.90")})

    _service(session_factory, Clock(KICKOFF).at(4), provider=provider, repo=repo).capture()
    result = _clv(repo, KICKOFF, bet_odds_at)

    assert result.available is True
    assert result.closing_odd == pytest.approx(1.90)
    assert result.closing_captured_at == "2026-11-17T19:52:00+00:00"
    assert result.closing_minutes_before_kickoff == pytest.approx(8.0)


def test_update_del_provider_invariato_resta_chiusura_non_successiva(session_factory):
    _add_match(session_factory, 1, KICKOFF)
    bet_odds_at = KICKOFF - timedelta(minutes=60)
    repo = FakeSnapshotRepo()
    provider = FakeProvider({1: _payload(update=bet_odds_at.isoformat())})

    report = _service(session_factory, Clock(KICKOFF).at(4), provider=provider, repo=repo).capture()
    result = _clv(repo, KICKOFF, bet_odds_at)

    assert report["captured"] == 1
    assert result.available is False
    assert result.reason_code == REASON_CLOSING_NOT_AFTER_BET


def test_senza_update_del_provider_vale_l_ora_del_download(session_factory):
    kickoff = datetime.now(timezone.utc) + timedelta(minutes=3)
    _add_match(session_factory, 1, kickoff)
    bet_odds_at = kickoff - timedelta(minutes=60)
    repo = FakeSnapshotRepo()
    provider = FakeProvider({1: _payload(update=None)})

    report = _service(session_factory, lambda: datetime.now(timezone.utc), provider=provider, repo=repo).capture()
    result = _clv(repo, kickoff, bet_odds_at)

    assert report["captures"][0]["captured_at_before_kickoff"] is True
    assert result.available is True
    assert 0 < result.closing_minutes_before_kickoff <= 3
