"""Pipeline dati LIVE (LIVE-01, Fase LIVE ORACLE).

Separata DELIBERATAMENTE dalla pipeline pre-match (`download_match_service.py`
/DATA-03, che scrive su `match`/`statistics`/`odds`): questo modulo scrive
SOLO sulle tabelle dedicate `live_fixture_snapshot`/`live_match_event`/
`live_fixture_stat_snapshot` (`src/data/live/live_models.py`) - "Non
mischiare training pre-match e live" (acceptance criteria).

Riusa `ApiSportsProvider` COSI' COM'E' (nessuna duplicazione della logica di
retry/rate-limit gia' presente in `api_sports_provider.py`): questo modulo
aggiunge solo (1) l'isolamento dell'errore per singola lega/fixture
("Provider errors isolati" - un fallimento non blocca il resto del batch,
stesso principio gia' visto in `download_match_service.py`/DATA-03 e in
`DashboardService._fetch_api_live_fixtures`) e (2) una cache TTL DEDICATA
("Cache/polling controllato") separata da quella di `DashboardService`
(`_api_cache`), cosi' i due moduli restano indipendenti.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from src.data.live.live_models import (
    LiveFixtureSnapshot,
    LiveFixtureStatSnapshot,
    LiveMatchEvent,
    live_event_id,
)
from src.repository.live_data_repository import LiveDataRepository
from src.service_ia.config.app_config import AppConfig, load_app_config
from src.service_ia.pre_processing.api_sports_provider import ApiSportsProvider


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fixture_id_from_raw(raw_fixture: dict[str, Any]) -> Optional[int]:
    fixture = raw_fixture.get("fixture") or {}
    return _safe_int(fixture.get("id"))


def dedupe_fixtures(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stesso principio di `DashboardService._dedupe_api_fixtures`: l'API
    puo' ripetere la stessa fixture (es. per lega multipla/errore provider),
    l'ultima vista vince."""
    container: dict[int, dict[str, Any]] = {}
    for fixture in fixtures:
        fixture_id = fixture_id_from_raw(fixture)
        if fixture_id is None:
            continue
        container[fixture_id] = fixture
    return list(container.values())


def extract_fixture_snapshot(raw_fixture: dict[str, Any], captured_at: Optional[datetime] = None) -> LiveFixtureSnapshot:
    """Mappa UNA fixture grezza (`fixtures?live=all`) in uno snapshot -
    funzione pura, testabile senza DB/provider."""
    fixture_id = fixture_id_from_raw(raw_fixture)
    if fixture_id is None:
        raise ValueError("raw_fixture senza 'fixture.id' valido")

    captured_at = captured_at or datetime.now(timezone.utc)
    fixture = raw_fixture.get("fixture") or {}
    status = fixture.get("status") or {}
    league = raw_fixture.get("league") or {}
    teams = raw_fixture.get("teams") or {}
    goals = raw_fixture.get("goals") or {}

    return LiveFixtureSnapshot(
        fixture_id=fixture_id,
        captured_at=captured_at,
        league_id=_safe_int(league.get("id")),
        status=status.get("short"),
        elapsed_minute=_safe_int(status.get("elapsed")),
        elapsed_extra=_safe_int(status.get("extra")),
        home_team_id=_safe_int((teams.get("home") or {}).get("id")),
        away_team_id=_safe_int((teams.get("away") or {}).get("id")),
        home_goals=_safe_int(goals.get("home")),
        away_goals=_safe_int(goals.get("away")),
        raw_payload=raw_fixture,
        source="api_sports",
    )


def extract_event(fixture_id: int, raw_event: dict[str, Any], captured_at: Optional[datetime] = None) -> LiveMatchEvent:
    """Mappa UN evento grezzo (`fixtures/events`) - funzione pura."""
    captured_at = captured_at or datetime.now(timezone.utc)
    time_block = raw_event.get("time") or {}
    team = raw_event.get("team") or {}
    player = raw_event.get("player") or {}
    assist = raw_event.get("assist") or {}

    elapsed = _safe_int(time_block.get("elapsed"))
    elapsed_extra = _safe_int(time_block.get("extra"))
    event_type = raw_event.get("type")
    event_detail = raw_event.get("detail")
    team_id = _safe_int(team.get("id"))
    player_name = player.get("name")

    event_id = live_event_id(
        fixture_id=fixture_id,
        event_type=event_type,
        detail=event_detail,
        elapsed=elapsed,
        elapsed_extra=elapsed_extra,
        team_id=team_id,
        player_name=player_name,
    )
    return LiveMatchEvent(
        id_event=event_id,
        fixture_id=fixture_id,
        captured_at=captured_at,
        elapsed_minute=elapsed,
        elapsed_extra=elapsed_extra,
        event_type=event_type,
        event_detail=event_detail,
        team_id=team_id,
        team_name=team.get("name"),
        player_name=player_name,
        assist_name=assist.get("name"),
        comments=raw_event.get("comments"),
        source="api_sports",
    )


def extract_stat_snapshot(
    fixture_id: int, raw_stat_entry: dict[str, Any], captured_at: Optional[datetime] = None
) -> LiveFixtureStatSnapshot:
    """Mappa UN blocco statistiche-per-team grezzo (`fixtures/statistics`) -
    funzione pura; `stats` resta il payload grezzo (nessuna normalizzazione:
    scope di LIVE-02/feature store, non di questa pipeline dati)."""
    captured_at = captured_at or datetime.now(timezone.utc)
    team = raw_stat_entry.get("team") or {}
    return LiveFixtureStatSnapshot(
        fixture_id=fixture_id,
        team_id=_safe_int(team.get("id")),
        captured_at=captured_at,
        stats=raw_stat_entry.get("statistics") or [],
        source="api_sports",
    )


class _TtlCache:
    """Cache TTL minimale e DEDICATA a questo modulo (non la stessa istanza/
    classe di `DashboardService._api_cache`): "Cache/polling controllato"
    senza accoppiare i due moduli."""

    def __init__(self, ttl_seconds: int):
        self._ttl_seconds = max(0, int(ttl_seconds))
        self._store: dict[str, tuple[datetime, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if not item:
            return None
        ts, payload = item
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        if age > self._ttl_seconds:
            return None
        return payload

    def set(self, key: str, payload: Any) -> None:
        self._store[key] = (datetime.now(timezone.utc), payload)


class LiveDataService:
    """Orchestratore DB-aware: fetch (isolato per lega/fixture, con cache
    TTL) + persistenza sul dataset live distinto. Dependency injection di
    provider/repository/config (stesso pattern di `DataQualityService`) per
    essere testabile senza rete/DB reali."""

    def __init__(
        self,
        provider: Optional[ApiSportsProvider] = None,
        repository: Optional[LiveDataRepository] = None,
        cfg: Optional[AppConfig] = None,
    ):
        self.cfg = cfg or load_app_config()
        self.provider = provider or ApiSportsProvider()
        self.repository = repository or LiveDataRepository()
        self._cache = _TtlCache(ttl_seconds=self.cfg.live_cache_ttl_seconds)

    def fetch_live_fixtures(
        self, leagues: Optional[list[int]] = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Ritorna (fixtures, errors). Un fallimento del provider finisce in
        `errors` senza propagarsi (acceptance criteria "Provider errors
        isolati"): dal 2026-09-15 la chiamata e' una sola, quindi non c'e'
        piu' un "resto del batch" da salvare come quando il giro era per
        lega, ma il job live non deve comunque fallire in blocco.

        Fix consumo quota 2026-09-15: PRIMA faceva una chiamata
        `fixtures?live=all&league=X` PER OGNI lega censita - 18 chiamate a
        ogni singolo poll, anche quando non si stava giocando NIENTE. Con
        `live_sync_interval_seconds=90` sono ~720 chiamate/ora a stadi vuoti,
        cioe' oltre 17.000 al giorno su un piano da 7.500: il job da solo
        bruciava la quota giornaliera, e a quota esaurita `request()`
        restituisce `[]` in silenzio, quindi TUTTI gli altri import (partite
        di ieri, calendario, quote) tornavano vuoti senza segnalare nulla.
        Misurato in diretta durante la diagnosi: 20 chiamate consumate in 3
        minuti con zero partite live.

        Ora si fa UNA sola chiamata `fixtures?live=all` (l'endpoint
        restituisce tutte le partite in corso nel mondo) e si filtra per lega
        IN MEMORIA: 1 chiamata per poll invece di 18, ~40/ora, con lo stesso
        risultato esatto. Il filtro resta perche' i campionati fuori da
        `cfg.leagues` non ci interessano e non devono finire nelle tabelle
        `live_*`.
        """
        leagues = leagues if leagues is not None else list(self.cfg.leagues or [])
        cache_key = "live_fixtures:" + ",".join(str(item) for item in leagues)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        fixtures: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        try:
            payload = self.provider.get_fixtures(live="all")
        except Exception as exc:
            # Nessun isolamento "per lega" da preservare: e' una chiamata
            # sola, quindi se cade non c'e' un resto del batch da salvare.
            errors.append({"scope": "live_fixtures", "league": None, "error": str(exc)})
            payload = []

        ammesse = {int(item) for item in leagues}
        for fixture in payload or []:
            league_id = _safe_int((fixture.get("league") or {}).get("id"))
            if ammesse and league_id not in ammesse:
                continue
            fixtures.append(fixture)

        result = (dedupe_fixtures(fixtures), errors)
        self._cache.set(cache_key, result)
        return result

    def fetch_fixture_events(self, fixture_id: int) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
        cache_key = f"events:{fixture_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            payload = self.provider.get_fixture_events(fixture_id)
            result: tuple[list[dict[str, Any]], Optional[dict[str, Any]]] = (
                payload if isinstance(payload, list) else [],
                None,
            )
        except Exception as exc:  # provider isolato per fixture
            result = ([], {"scope": "fixture_events", "fixture_id": fixture_id, "error": str(exc)})

        self._cache.set(cache_key, result)
        return result

    def fetch_fixture_statistics(self, fixture_id: int) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
        cache_key = f"stats:{fixture_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            payload = self.provider.get_fixture_statistics(fixture_id)
            result: tuple[list[dict[str, Any]], Optional[dict[str, Any]]] = (
                payload if isinstance(payload, list) else [],
                None,
            )
        except Exception as exc:  # provider isolato per fixture
            result = ([], {"scope": "fixture_statistics", "fixture_id": fixture_id, "error": str(exc)})

        self._cache.set(cache_key, result)
        return result

    def sync_live_data(
        self,
        leagues: Optional[list[int]] = None,
        include_events: bool = True,
        include_statistics: bool = True,
    ) -> dict[str, Any]:
        """Orchestratore principale (invocato dal job schedulato/manuale,
        `live_sync_job.py`): scopre le fixture attualmente live, salva uno
        snapshot per ciascuna + eventi/statistiche associati. Un errore su
        UNA fixture (o lega) finisce in `errors` senza bloccare le altre -
        stesso stile "report" di `download_import_matches`/DATA-03."""
        captured_at = datetime.now(timezone.utc)
        fixtures, errors = self.fetch_live_fixtures(leagues=leagues)
        errors = list(errors)

        fixture_snapshots: list[LiveFixtureSnapshot] = []
        event_rows: list[LiveMatchEvent] = []
        stat_rows: list[LiveFixtureStatSnapshot] = []

        for raw_fixture in fixtures:
            fixture_id = fixture_id_from_raw(raw_fixture)
            if fixture_id is None:
                errors.append({"scope": "fixture_snapshot", "fixture_id": None, "error": "fixture.id mancante"})
                continue

            try:
                fixture_snapshots.append(extract_fixture_snapshot(raw_fixture, captured_at=captured_at))
            except Exception as exc:
                errors.append({"scope": "fixture_snapshot", "fixture_id": fixture_id, "error": str(exc)})
                continue

            if include_events:
                events_payload, event_error = self.fetch_fixture_events(fixture_id)
                if event_error:
                    errors.append(event_error)
                for raw_event in events_payload:
                    try:
                        event_rows.append(extract_event(fixture_id, raw_event, captured_at=captured_at))
                    except Exception as exc:
                        errors.append({"scope": "event", "fixture_id": fixture_id, "error": str(exc)})

            if include_statistics:
                stats_payload, stats_error = self.fetch_fixture_statistics(fixture_id)
                if stats_error:
                    errors.append(stats_error)
                for raw_stat_entry in stats_payload:
                    try:
                        stat_rows.append(extract_stat_snapshot(fixture_id, raw_stat_entry, captured_at=captured_at))
                    except Exception as exc:
                        errors.append({"scope": "stat_snapshot", "fixture_id": fixture_id, "error": str(exc)})

        self.repository.save_fixture_snapshots(fixture_snapshots)
        self.repository.save_events(event_rows)
        self.repository.save_stat_snapshots(stat_rows)

        return {
            "captured_at": captured_at.isoformat(),
            "fixtures_live": len(fixture_snapshots),
            "events_saved": len(event_rows),
            "stat_snapshots_saved": len(stat_rows),
            "errors": errors,
        }
