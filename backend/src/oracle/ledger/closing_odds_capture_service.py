"""Quote di chiusura per le partite con giocate ufficiali (job `closing_odds_capture`).

Il CLV ufficiale (`GET /predictions/official/clv`, `src/oracle/backtest/clv.py`)
usa come chiusura, per ogni bookmaker/esito, l'ultimo `OddsSnapshot` con
`captured_at <= kickoff`, e la accetta solo se e' piu' recente della quota
della giocata (`odds_captured_at` nel ledger). Le quote arrivano a DB solo
dagli import (sync di oggi ogni 30 minuti, calendario e "Aggiorna tutto" una
volta al giorno) e la giocata ufficiale viene catturata fino a 60 minuti
prima del kickoff: spesso l'ultimo snapshot prima del kickoff e' proprio
quello della giocata e il CLV resta non misurabile.

Questo servizio scarica `GET /odds?fixture=<id>` vicino al calcio d'inizio,
SOLO per le partite `NS` con almeno una riga nel ledger ufficiale (cohort
`official_paper`, qualunque decisione: PLAY, BORDERLINE, NO BET). La scelta
delle partite e' fatta dal DB, senza chiamate al provider. Al massimo una
chiamata per "momento", quindi al massimo 2 per partita:

- `early`: fra 15 e 10 minuti prima del kickoff. Il job gira ogni 5 minuti
  piu' il battito dello scheduler (30 secondi) e la durata del giro, quindi
  puo' saltare una finestra di 5 minuti: in quel caso il momento viene
  recuperato fra 10 e 5 minuti prima;
- `late`: negli ultimi 5 minuti prima del kickoff.

Cosa e' gia' stato tentato e' scritto in un registro leggero sul bucket
(`ClosingCaptureRegistry`, nessuna tabella nuova) PRIMA della chiamata: gli
snapshot salvati non bastano, perche' il loro `captured_at` e' l'ora di
aggiornamento del provider e non l'ora del download.

`captured_at` degli snapshot e' quello di `map_odds_snapshots`: il campo
`update` della risposta (ora in cui API-Sports ha aggiornato le quote della
partita) o, se manca, l'ora del download. Il download avviene prima del
kickoff, quindi in entrambi i casi lo snapshot e' una chiusura valida
(`captured_at <= kickoff`); con l'ora del download e' sempre piu' recente
della giocata. Se invece il provider non ha aggiornato le quote dopo la
giocata, `update` resta lo stesso: lo snapshot ha la stessa chiave di quello
gia' salvato (l'upsert non aggiunge righe) e il CLV di quella giocata resta
`closing_not_after_bet`, correttamente (non c'e' una quota piu' recente). Il
report lo conta in `provider_update_unchanged`.

Non prende il lock degli import (`job_lock`): salterebbe ogni volta che gira
`data_sync_today`. Le scritture sono l'upsert idempotente di
`OddsSnapshotRepository.save_many` (righe in ordine di chiave), sicuro anche
in parallelo a un import. Gli errori del provider finiscono nel report,
mai in un'eccezione.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from sqlalchemy import select

from src.jobs.api_quota_state import is_quota_exhausted_today
from src.oracle.ledger.official_performance_service import OFFICIAL_COHORT
from src.repository.base.repository_db import SessionLocal
from src.repository.odds_snapshot_repository import OddsSnapshotRepository
from src.service_ia.model.match import Match, PredictionLedger
from src.service_ia.pre_processing.api_sports_provider import (
    ApiSportsCircuitOpenError,
    ApiSportsQuotaExceededError,
)
from src.service_ia.pre_processing.download_match_service import map_odds_snapshots
from src.storage import bucket_store

logger = logging.getLogger(__name__)

# Minuti prima del kickoff.
HORIZON_MINUTES = 15.0
EARLY_WINDOW_START = 15.0
EARLY_WINDOW_END = 10.0
EARLY_RECOVERY_END = 5.0
LATE_WINDOW_START = 5.0

MOMENT_EARLY = "early"
MOMENT_LATE = "late"

SKIP_ALREADY_CAPTURED = "already_captured"
SKIP_OUTSIDE_WINDOW = "outside_window"
SKIP_QUOTA_EXHAUSTED = "quota_exhausted"

WINDOWS_MINUTES_BEFORE_KICKOFF = {
    MOMENT_EARLY: [EARLY_WINDOW_START, EARLY_WINDOW_END],
    "early_recovery": [EARLY_WINDOW_END, EARLY_RECOVERY_END],
    MOMENT_LATE: [LATE_WINDOW_START, 0.0],
}

# Righe del ledger considerate: kickoff da ieri in poi. Una partita rinviata
# riceve nuove righe ufficiali al nuovo kickoff (la chiave di cattura
# contiene il kickoff), quindi non serve guardare piu' indietro.
_LEDGER_LOOKBACK = timedelta(days=1)
_MAX_REPORT_ROWS = 50


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _aware(value)
    try:
        return _aware(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except (TypeError, ValueError):
        return None


def moment_for(minutes_to_kickoff: float, captured: set[str]) -> tuple[Optional[str], Optional[str]]:
    """Momento da catturare adesso, oppure `(None, motivo)`.

    `captured` sono i momenti gia' tentati per questa partita e questo
    kickoff. Fra 10 e 5 minuti prima si cattura solo l'`early` mancato."""
    if minutes_to_kickoff < 0 or minutes_to_kickoff > HORIZON_MINUTES:
        return None, SKIP_OUTSIDE_WINDOW
    if minutes_to_kickoff <= LATE_WINDOW_START:
        if MOMENT_LATE in captured:
            return None, SKIP_ALREADY_CAPTURED
        return MOMENT_LATE, None
    if MOMENT_EARLY not in captured:
        return MOMENT_EARLY, None
    if minutes_to_kickoff >= EARLY_WINDOW_END:
        return None, SKIP_ALREADY_CAPTURED
    return None, SKIP_OUTSIDE_WINDOW


@dataclass(frozen=True)
class ClosingCandidate:
    fixture_id: int
    id_match: str
    kickoff_at: datetime


class ClosingCaptureRegistry:
    """Momenti gia' tentati per partita e kickoff, in un solo JSON sul bucket
    (stesso meccanismo di `job_settings.json`/`api_quota_state.json`).

    Il momento viene scritto PRIMA della chiamata, cosi' un secondo giro (o
    un "Esegui ora" in un altro processo) non lo ripete. Ogni scrittura
    rilegge lo stato: dentro il processo un lock serializza le scritture;
    fra i container `api` e `scheduler` non esiste un lock (il bucket non ha
    compare-and-swap), resta una finestra di pochi istanti. Le voci con
    kickoff piu' vecchio di `retention` vengono rimosse."""

    _LOCK = threading.Lock()

    def __init__(self, key: str = "best_models/closing_odds_capture_state.json", retention: timedelta = timedelta(days=2)):
        self.key = key
        self.retention = retention

    def read(self) -> dict[str, Any]:
        """Lettura STRETTA: un errore del bucket viene propagato (senza
        registro il job non sa cosa ha gia' scaricato e non chiama)."""
        payload = bucket_store.get_json(self.key, default={})
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def entry_key(candidate: ClosingCandidate) -> str:
        return f"{candidate.fixture_id}|{_aware(candidate.kickoff_at).isoformat()}"

    @classmethod
    def moments(cls, state: dict[str, Any], candidate: ClosingCandidate) -> set[str]:
        entry = state.get(cls.entry_key(candidate))
        moments = entry.get("moments") if isinstance(entry, dict) else None
        return set(moments) if isinstance(moments, dict) else set()

    def _prune(self, state: dict[str, Any], now: datetime) -> None:
        limit = now - self.retention
        for key in list(state):
            entry = state.get(key)
            kickoff = _parse_datetime(entry.get("kickoff_at")) if isinstance(entry, dict) else None
            if kickoff is None or kickoff < limit:
                state.pop(key, None)

    def _entry(self, state: dict[str, Any], candidate: ClosingCandidate) -> dict[str, Any]:
        entry = state.get(self.entry_key(candidate))
        moments = entry.get("moments") if isinstance(entry, dict) else None
        return {
            "fixture_id": candidate.fixture_id,
            "kickoff_at": _aware(candidate.kickoff_at).isoformat(),
            "moments": dict(moments) if isinstance(moments, dict) else {},
        }

    def claim(self, candidate: ClosingCandidate, moment: str, now: datetime) -> bool:
        """True se il momento era libero ed e' stato prenotato."""
        with self._LOCK:
            state = self.read()
            entry = self._entry(state, candidate)
            if moment in entry["moments"]:
                return False
            entry["moments"][moment] = {"status": "pending", "attempted_at": now.isoformat()}
            state[self.entry_key(candidate)] = entry
            self._prune(state, now)
            bucket_store.put_json(self.key, state)
            return True

    def record(self, candidate: ClosingCandidate, moment: str, outcome: dict[str, Any]) -> None:
        with self._LOCK:
            state = self.read()
            entry = self._entry(state, candidate)
            entry["moments"][moment] = {**(entry["moments"].get(moment) or {}), **outcome}
            state[self.entry_key(candidate)] = entry
            bucket_store.put_json(self.key, state)

    def release(self, candidate: ClosingCandidate, moment: str) -> None:
        """Libera un momento prenotato per cui nessuna richiesta e' partita."""
        with self._LOCK:
            state = self.read()
            entry = self._entry(state, candidate)
            entry["moments"].pop(moment, None)
            if entry["moments"]:
                state[self.entry_key(candidate)] = entry
            else:
                state.pop(self.entry_key(candidate), None)
            bucket_store.put_json(self.key, state)


def _empty_report() -> dict[str, Any]:
    return {
        "candidates": 0,
        "calls": 0,
        "captured": 0,
        "snapshots_saved": 0,
        "empty_responses": 0,
        "provider_update_unchanged": 0,
        "skipped": {SKIP_ALREADY_CAPTURED: 0, SKIP_OUTSIDE_WINDOW: 0, SKIP_QUOTA_EXHAUSTED: 0},
        "errors": {"provider": 0, "quota_exhausted": 0, "save": 0, "registry": 0},
        "error_details": [],
        "captures": [],
        "horizon_minutes": HORIZON_MINUTES,
        "windows_minutes_before_kickoff": WINDOWS_MINUTES_BEFORE_KICKOFF,
    }


class ClosingOddsCaptureService:
    def __init__(
        self,
        provider: Any = None,
        snapshot_repo: Optional[OddsSnapshotRepository] = None,
        registry: Optional[ClosingCaptureRegistry] = None,
        session_factory: Optional[Callable[[], Any]] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
        quota_exhausted_fn: Optional[Callable[[], bool]] = None,
    ):
        self._provider = provider
        self.snapshot_repo = snapshot_repo or OddsSnapshotRepository()
        self.registry = registry or ClosingCaptureRegistry()
        self.session_factory = session_factory or SessionLocal
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._quota_exhausted_fn = quota_exhausted_fn or is_quota_exhausted_today

    @property
    def provider(self) -> Any:
        if self._provider is None:
            # Stesso provider (retry, circuit breaker, quota) degli altri job.
            from src.service_ia.utility.request_api import get_api_sports_provider

            self._provider = get_api_sports_provider()
        return self._provider

    def _now(self) -> datetime:
        return _aware(self._now_fn())

    def candidates(self, now: datetime) -> list[ClosingCandidate]:
        """Partite `NS` con almeno una riga ufficiale nel ledger e kickoff
        nei prossimi `HORIZON_MINUTES` minuti. Solo DB, ordinate per kickoff
        (le piu' vicine prima: la chiusura e' il momento che conta di piu')."""
        official_fixtures = (
            select(PredictionLedger.fixture_id)
            .where(PredictionLedger.cohort == OFFICIAL_COHORT)
            .where(PredictionLedger.kickoff_at >= now - _LEDGER_LOOKBACK)
            .distinct()
        )
        with self.session_factory() as session:
            rows = session.execute(
                select(Match.id_fixture, Match.id_match_fk, Match.date_match)
                .where(Match.status == "NS")
                .where(Match.id_fixture.in_(official_fixtures))
            ).all()
        upper = now + timedelta(minutes=HORIZON_MINUTES)
        found = []
        for fixture_id, id_match, date_match in rows:
            # `date_match` e' una colonna stringa storica: confronto in Python.
            kickoff = _parse_datetime(date_match)
            if fixture_id is None or kickoff is None or not now <= kickoff <= upper:
                continue
            found.append(ClosingCandidate(fixture_id=int(fixture_id), id_match=str(id_match), kickoff_at=kickoff))
        return sorted(found, key=lambda c: (c.kickoff_at, c.fixture_id))

    def _quota_exhausted(self) -> bool:
        try:
            return bool(self._quota_exhausted_fn())
        except Exception:  # noqa: BLE001 - la lettura gia' non solleva: difesa in piu'
            return False

    @staticmethod
    def _error(report: dict[str, Any], kind: str, candidate: Optional[ClosingCandidate], moment: Optional[str], exc: Any) -> None:
        report["errors"][kind] += 1
        if len(report["error_details"]) < _MAX_REPORT_ROWS:
            report["error_details"].append(
                {
                    "fixture_id": candidate.fixture_id if candidate else None,
                    "moment": moment,
                    "kind": kind,
                    "error": str(exc),
                }
            )

    def capture(self) -> dict[str, Any]:
        report = _empty_report()
        candidates = self.candidates(self._now())
        report["candidates"] = len(candidates)
        if not candidates:
            return report

        try:
            state = self.registry.read()
        except Exception as exc:  # noqa: BLE001
            # Senza registro non si sa cosa e' gia' stato scaricato: nessuna chiamata.
            self._error(report, "registry", None, None, exc)
            return report

        quota_exhausted = self._quota_exhausted()
        for candidate in candidates:
            now = self._now()
            minutes = (candidate.kickoff_at - now).total_seconds() / 60.0
            moment, skip_reason = moment_for(minutes, self.registry.moments(state, candidate))
            if moment is None:
                report["skipped"][skip_reason] += 1
                continue
            if quota_exhausted:
                report["skipped"][SKIP_QUOTA_EXHAUSTED] += 1
                continue
            try:
                claimed = self.registry.claim(candidate, moment, now)
            except Exception as exc:  # noqa: BLE001
                self._error(report, "registry", candidate, moment, exc)
                continue
            if not claimed:
                # Preso nel frattempo da un altro giro/processo.
                report["skipped"][SKIP_ALREADY_CAPTURED] += 1
                continue

            outcome = self._fetch_and_save(report, candidate, moment, minutes)
            if outcome is None:
                self._safe_registry_call(report, candidate, moment, self.registry.release, candidate, moment)
                continue
            if outcome.get("status") == SKIP_QUOTA_EXHAUSTED:
                quota_exhausted = True
            self._safe_registry_call(report, candidate, moment, self.registry.record, candidate, moment, outcome)
        return report

    def _safe_registry_call(self, report, candidate, moment, func, *args) -> None:
        try:
            func(*args)
        except Exception as exc:  # noqa: BLE001
            self._error(report, "registry", candidate, moment, exc)

    def _fetch_and_save(
        self, report: dict[str, Any], candidate: ClosingCandidate, moment: str, minutes: float
    ) -> Optional[dict[str, Any]]:
        """Una chiamata e il salvataggio. `None` se nessuna richiesta e'
        partita (circuit breaker aperto): il momento torna libero."""
        fixture_id = candidate.fixture_id
        try:
            payload = self.provider.get_fixture_odds(fixture_id)
        except ApiSportsCircuitOpenError as exc:
            self._error(report, "provider", candidate, moment, exc)
            return None
        except ApiSportsQuotaExceededError as exc:
            report["calls"] += 1
            self._error(report, "quota_exhausted", candidate, moment, exc)
            return {"status": SKIP_QUOTA_EXHAUSTED, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - ApiSportsUnavailableError e imprevisti
            report["calls"] += 1
            self._error(report, "provider", candidate, moment, exc)
            return {"status": "error", "error": str(exc)}
        report["calls"] += 1

        try:
            snapshots = map_odds_snapshots(
                id_match=candidate.id_match, id_fixture=fixture_id, fixture_bookmakers=payload or []
            )
        except Exception as exc:  # noqa: BLE001 - risposta con una forma inattesa
            self._error(report, "provider", candidate, moment, exc)
            return {"status": "error", "error": str(exc)}
        if not snapshots:
            report["empty_responses"] += 1
            return {"status": "empty", "snapshots": 0}

        captured_at = _aware(snapshots[0].captured_at)
        try:
            previous = self.snapshot_repo.latest_captured_at(fixture_id)
        except Exception:  # noqa: BLE001 - serve solo al report
            previous = None
            update_changed: Optional[bool] = None
        else:
            update_changed = previous is None or captured_at > _aware(previous)

        try:
            self.snapshot_repo.save_many(snapshots)
        except Exception as exc:  # noqa: BLE001
            self._error(report, "save", candidate, moment, exc)
            return {"status": "save_error", "error": str(exc)}

        report["captured"] += 1
        report["snapshots_saved"] += len(snapshots)
        if update_changed is False:
            report["provider_update_unchanged"] += 1
        first = payload[0] if payload and isinstance(payload[0], dict) else {}
        outcome = {
            "status": "ok",
            "snapshots": len(snapshots),
            "captured_at": captured_at.isoformat(),
            "provider_update": first.get("update"),
            "provider_update_changed": update_changed,
        }
        if len(report["captures"]) < _MAX_REPORT_ROWS:
            report["captures"].append(
                {
                    "fixture_id": fixture_id,
                    "moment": moment,
                    "minutes_before_kickoff": round(minutes, 2),
                    "captured_at_before_kickoff": captured_at <= _aware(candidate.kickoff_at),
                    **outcome,
                }
            )
        return outcome
