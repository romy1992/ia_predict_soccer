"""Test per LIVE-01 (Pipeline dati live separata).

Quattro livelli, stesso stile del resto del progetto:
1. `TestPureMapping`: funzioni pure di mapping (nessun DB/rete).
2. `TestLiveDataServiceFetch`: `LiveDataService.fetch_*` con provider
   mockato - isolamento errori per lega/fixture + cache TTL dedicata.
3. `TestLiveDataServiceSync`: orchestratore `sync_live_data` con provider e
   repository mockati - verifica cosa viene persistito e come gli errori
   parziali finiscono nel report senza bloccare il resto del batch.
4. `TestLiveDataRepositoryWithDb`: vero SQLite in-memory (stesso pattern di
   `prediction_ledger_test.py`/`crud_repository_test.py`) - esercita
   DAVVERO le query SQLAlchemy, incluso l'upsert idempotente degli eventi.
5. `TestRunManualLiveSync`: job manuale con `JobHistory`/`LiveDataService`
   mockati (successo e fallimento).
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.data.live.live_models as live_models  # noqa: F401  (registra le tabelle live su Base.metadata)
from src.data.live.live_models import Base, LiveFixtureSnapshot, LiveMatchEvent, live_event_id
from src.data.live.live_data_service import (
    LiveDataService,
    dedupe_fixtures,
    extract_event,
    extract_fixture_snapshot,
    extract_stat_snapshot,
    fixture_id_from_raw,
)
from src.data.live.live_sync_job import run_manual_live_sync
from src.repository.live_data_repository import LiveDataRepository
from src.service_ia.config.app_config import AppConfig


def _cfg(**overrides) -> AppConfig:
    base = dict(
        leagues=[39],
        seasons=[2026],
        data_sync_interval_minutes=30,
        settlement_interval_minutes=60,
        future_sync_hour=4,
        future_sync_minute=30,
        daily_refresh_hour=5,
        daily_refresh_minute=0,
        daily_refresh_days_ahead=7,
        training_hour=23,
        training_minute=0,
        live_sync_interval_seconds=90,
        live_cache_ttl_seconds=20,
        api_sports_daily_limit=7500,
        database_url="sqlite://",
        database_schema="public",
    )
    base.update(overrides)
    return AppConfig(**base)


def _raw_fixture(fixture_id=100, status="1H", elapsed=34, home_goals=1, away_goals=0, league_id=39):
    return {
        "fixture": {"id": fixture_id, "status": {"long": "First Half", "short": status, "elapsed": elapsed, "extra": None}},
        "league": {"id": league_id, "name": "Premier League"},
        "teams": {"home": {"id": 33, "name": "Home FC"}, "away": {"id": 34, "name": "Away FC"}},
        "goals": {"home": home_goals, "away": away_goals},
    }


def _raw_event(elapsed=34, event_type="Goal", detail="Normal Goal", team_id=33, player_name="Player X"):
    return {
        "time": {"elapsed": elapsed, "extra": None},
        "team": {"id": team_id, "name": "Home FC"},
        "player": {"id": 1, "name": player_name},
        "assist": {"id": None, "name": None},
        "type": event_type,
        "detail": detail,
        "comments": None,
    }


def _raw_stat_entry(team_id=33):
    return {
        "team": {"id": team_id, "name": "Home FC"},
        "statistics": [{"type": "Ball Possession", "value": "55%"}, {"type": "Shots on Goal", "value": 3}],
    }


class TestPureMapping(unittest.TestCase):
    def test_fixture_id_from_raw(self):
        self.assertEqual(fixture_id_from_raw(_raw_fixture(fixture_id=777)), 777)
        self.assertIsNone(fixture_id_from_raw({"fixture": {}}))
        self.assertIsNone(fixture_id_from_raw({}))

    def test_dedupe_fixtures_keeps_last_seen_per_id(self):
        first = _raw_fixture(fixture_id=1, status="1H")
        duplicate = _raw_fixture(fixture_id=1, status="2H")
        other = _raw_fixture(fixture_id=2, status="1H")
        deduped = dedupe_fixtures([first, duplicate, other])
        self.assertEqual(len(deduped), 2)
        by_id = {fixture_id_from_raw(f): f for f in deduped}
        self.assertEqual(by_id[1]["fixture"]["status"]["short"], "2H")

    def test_extract_fixture_snapshot_maps_all_fields(self):
        captured_at = datetime.now(timezone.utc)
        snapshot = extract_fixture_snapshot(_raw_fixture(fixture_id=100, status="1H", elapsed=34), captured_at=captured_at)
        self.assertEqual(snapshot.fixture_id, 100)
        self.assertEqual(snapshot.status, "1H")
        self.assertEqual(snapshot.elapsed_minute, 34)
        self.assertEqual(snapshot.league_id, 39)
        self.assertEqual(snapshot.home_team_id, 33)
        self.assertEqual(snapshot.away_team_id, 34)
        self.assertEqual(snapshot.home_goals, 1)
        self.assertEqual(snapshot.away_goals, 0)
        self.assertEqual(snapshot.captured_at, captured_at)
        self.assertIsNotNone(snapshot.raw_payload)

    def test_extract_fixture_snapshot_requires_fixture_id(self):
        with self.assertRaises(ValueError):
            extract_fixture_snapshot({"fixture": {}})

    def test_extract_event_maps_fields_and_id_is_deterministic(self):
        raw = _raw_event(elapsed=34, event_type="Goal", detail="Normal Goal", team_id=33, player_name="Player X")
        event_a = extract_event(100, raw)
        event_b = extract_event(100, raw)
        self.assertEqual(event_a.id_event, event_b.id_event)
        self.assertEqual(event_a.fixture_id, 100)
        self.assertEqual(event_a.elapsed_minute, 34)
        self.assertEqual(event_a.event_type, "Goal")
        self.assertEqual(event_a.event_detail, "Normal Goal")
        self.assertEqual(event_a.team_id, 33)
        self.assertEqual(event_a.player_name, "Player X")

    def test_live_event_id_differs_for_different_events(self):
        id_goal = live_event_id(100, "Goal", "Normal Goal", 34, None, 33, "Player X")
        id_card = live_event_id(100, "Card", "Yellow Card", 34, None, 33, "Player X")
        id_other_fixture = live_event_id(101, "Goal", "Normal Goal", 34, None, 33, "Player X")
        self.assertNotEqual(id_goal, id_card)
        self.assertNotEqual(id_goal, id_other_fixture)

    def test_extract_stat_snapshot_keeps_raw_statistics(self):
        snapshot = extract_stat_snapshot(100, _raw_stat_entry(team_id=33))
        self.assertEqual(snapshot.fixture_id, 100)
        self.assertEqual(snapshot.team_id, 33)
        self.assertEqual(len(snapshot.stats), 2)


class TestLiveDataServiceFetch(unittest.TestCase):
    """Acceptance criteria "Provider errors isolati" + "Cache/polling
    controllato" - provider mockato, nessuna rete/DB reale."""

    def test_fetch_live_fixtures_isolates_error_per_league(self):
        provider = mock.Mock()

        def fake_get_fixtures(**params):
            if params.get("league") == 39:
                raise ConnectionError("boom")
            return [_raw_fixture(fixture_id=2)]

        provider.get_fixtures.side_effect = fake_get_fixtures
        service = LiveDataService(provider=provider, repository=mock.Mock(), cfg=_cfg(leagues=[39, 140]))

        fixtures, errors = service.fetch_live_fixtures()

        self.assertEqual(len(fixtures), 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["league"], 39)
        self.assertEqual(errors[0]["scope"], "league_fixtures")

    def test_fetch_live_fixtures_uses_cache_within_ttl(self):
        provider = mock.Mock()
        provider.get_fixtures.return_value = [_raw_fixture(fixture_id=1)]
        service = LiveDataService(provider=provider, repository=mock.Mock(), cfg=_cfg(leagues=[39]))

        service.fetch_live_fixtures()
        service.fetch_live_fixtures()

        self.assertEqual(provider.get_fixtures.call_count, 1)

    def test_fetch_fixture_events_isolates_error_and_caches(self):
        provider = mock.Mock()
        provider.get_fixture_events.side_effect = TimeoutError("slow")
        service = LiveDataService(provider=provider, repository=mock.Mock(), cfg=_cfg())

        events, error = service.fetch_fixture_events(100)

        self.assertEqual(events, [])
        self.assertIsNotNone(error)
        self.assertEqual(error["scope"], "fixture_events")
        self.assertEqual(error["fixture_id"], 100)

        # Seconda chiamata entro il TTL: nessuna nuova invocazione al provider.
        service.fetch_fixture_events(100)
        self.assertEqual(provider.get_fixture_events.call_count, 1)

    def test_fetch_fixture_statistics_isolates_error(self):
        provider = mock.Mock()
        provider.get_fixture_statistics.side_effect = ValueError("bad payload")
        service = LiveDataService(provider=provider, repository=mock.Mock(), cfg=_cfg())

        stats, error = service.fetch_fixture_statistics(100)

        self.assertEqual(stats, [])
        self.assertEqual(error["scope"], "fixture_statistics")


class TestLiveDataServiceSync(unittest.TestCase):
    """Orchestratore `sync_live_data`: una fixture che fallisce (eventi o
    statistiche) NON deve bloccare le altre ("Provider errors isolati")."""

    def test_sync_live_data_saves_snapshots_events_and_stats(self):
        provider = mock.Mock()
        provider.get_fixtures.return_value = [_raw_fixture(fixture_id=100)]
        provider.get_fixture_events.return_value = [_raw_event()]
        provider.get_fixture_statistics.return_value = [_raw_stat_entry()]
        repository = mock.Mock()
        service = LiveDataService(provider=provider, repository=repository, cfg=_cfg(leagues=[39]))

        report = service.sync_live_data()

        self.assertEqual(report["fixtures_live"], 1)
        self.assertEqual(report["events_saved"], 1)
        self.assertEqual(report["stat_snapshots_saved"], 1)
        self.assertEqual(report["errors"], [])
        repository.save_fixture_snapshots.assert_called_once()
        repository.save_events.assert_called_once()
        repository.save_stat_snapshots.assert_called_once()

    def test_sync_live_data_isolates_error_on_single_fixture(self):
        provider = mock.Mock()
        provider.get_fixtures.return_value = [_raw_fixture(fixture_id=100), _raw_fixture(fixture_id=200)]

        def fake_events(fixture_id):
            if fixture_id == 100:
                raise RuntimeError("provider down")
            return [_raw_event()]

        provider.get_fixture_events.side_effect = fake_events
        provider.get_fixture_statistics.return_value = []
        repository = mock.Mock()
        service = LiveDataService(provider=provider, repository=repository, cfg=_cfg(leagues=[39]))

        report = service.sync_live_data(include_statistics=False)

        # Entrambe le fixture restano nello snapshot (l'errore riguarda SOLO gli eventi di 100).
        self.assertEqual(report["fixtures_live"], 2)
        self.assertEqual(report["events_saved"], 1)
        self.assertEqual(len(report["errors"]), 1)
        self.assertEqual(report["errors"][0]["fixture_id"], 100)

    def test_sync_live_data_reports_league_errors_without_crashing(self):
        provider = mock.Mock()
        provider.get_fixtures.side_effect = ConnectionError("down")
        repository = mock.Mock()
        service = LiveDataService(provider=provider, repository=repository, cfg=_cfg(leagues=[39]))

        report = service.sync_live_data()

        self.assertEqual(report["fixtures_live"], 0)
        self.assertEqual(len(report["errors"]), 1)
        repository.save_fixture_snapshots.assert_called_once_with([])


def _make_in_memory_engine():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return engine


class TestLiveDataRepositoryWithDb(unittest.TestCase):
    """Vero SQLite in-memory (non mock) - stesso principio di
    `crud_repository_test.py`/`prediction_ledger_test.py`."""

    def setUp(self):
        self.engine = _make_in_memory_engine()
        self.session_factory = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        self._session_local_patch = mock.patch(
            "src.repository.live_data_repository.SessionLocal", new=self.session_factory
        )
        self._session_local_patch.start()
        self.addCleanup(self._session_local_patch.stop)
        self.repo = LiveDataRepository()

    def test_save_and_list_fixture_snapshots(self):
        snap1 = extract_fixture_snapshot(_raw_fixture(fixture_id=100, status="1H"))
        self.repo.save_fixture_snapshots([snap1])

        rows = self.repo.list_fixture_snapshots(100)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, "1H")

    def test_latest_fixture_snapshots_keeps_most_recent_per_fixture(self):
        older = extract_fixture_snapshot(
            _raw_fixture(fixture_id=100, status="1H"), captured_at=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc)
        )
        newer = extract_fixture_snapshot(
            _raw_fixture(fixture_id=100, status="2H"), captured_at=datetime(2026, 9, 4, 10, 30, tzinfo=timezone.utc)
        )
        self.repo.save_fixture_snapshots([older, newer])

        latest = self.repo.latest_fixture_snapshots()
        self.assertEqual(latest[100].status, "2H")

    def test_list_active_fixture_ids_excludes_final_status(self):
        live = extract_fixture_snapshot(_raw_fixture(fixture_id=100, status="1H"))
        finished = extract_fixture_snapshot(_raw_fixture(fixture_id=200, status="FT"))
        self.repo.save_fixture_snapshots([live, finished])

        active = self.repo.list_active_fixture_ids()
        self.assertEqual(active, [100])

    def test_save_events_is_idempotent_on_repeated_poll(self):
        """Stesso evento osservato in due poll successivi (l'API ritorna
        SEMPRE l'elenco completo) -> id deterministico -> `merge` aggiorna
        la stessa riga invece di duplicarla ("Persistenza idempotente")."""
        raw = _raw_event(elapsed=34)
        event_poll_1 = extract_event(100, raw, captured_at=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc))
        event_poll_2 = extract_event(100, raw, captured_at=datetime(2026, 9, 4, 10, 1, tzinfo=timezone.utc))

        self.repo.save_events([event_poll_1])
        self.repo.save_events([event_poll_2])

        events = self.repo.list_events_for_fixture(100)
        self.assertEqual(len(events), 1)

    def test_list_events_for_fixture_orders_by_elapsed_minute(self):
        late = extract_event(100, _raw_event(elapsed=80, event_type="Card", detail="Yellow Card"))
        early = extract_event(100, _raw_event(elapsed=10, event_type="Goal", detail="Normal Goal"))
        self.repo.save_events([late, early])

        events = self.repo.list_events_for_fixture(100)
        self.assertEqual([event.elapsed_minute for event in events], [10, 80])

    def test_save_and_list_stat_snapshots_per_team(self):
        home_stat = extract_stat_snapshot(100, _raw_stat_entry(team_id=33))
        away_stat = extract_stat_snapshot(100, _raw_stat_entry(team_id=34))
        self.repo.save_stat_snapshots([home_stat, away_stat])

        latest = self.repo.latest_stat_snapshots_for_fixture(100)
        team_ids = {row.team_id for row in latest}
        self.assertEqual(team_ids, {33, 34})


class TestRunManualLiveSync(unittest.TestCase):
    """Job manuale (`src/data/live/live_sync_job.py`) - `JobHistory` e
    `LiveDataService` mockati, stesso stile try/except del resto degli
    altri job (`run_manual_today_update`/`run_manual_settlement`)."""

    @mock.patch("src.data.live.live_sync_job.load_app_config")
    @mock.patch("src.data.live.live_sync_job.LiveDataService")
    @mock.patch("src.data.live.live_sync_job.JobHistory")
    def test_success_marks_job_success_with_report(self, mock_history_cls, mock_service_cls, mock_load_cfg):
        mock_load_cfg.return_value = _cfg(leagues=[39])
        mock_history = mock_history_cls.return_value
        mock_history.create_job.return_value = {"job_id": "job-1"}
        mock_service = mock_service_cls.return_value
        mock_service.sync_live_data.return_value = {
            "captured_at": "2026-09-04T10:00:00+00:00",
            "fixtures_live": 2,
            "events_saved": 1,
            "stat_snapshots_saved": 2,
            "errors": [],
        }

        report = run_manual_live_sync()

        self.assertEqual(report["fixtures_live"], 2)
        self.assertEqual(report["job_id"], "job-1")
        mock_history.mark_success.assert_called_once()
        mock_history.mark_failed.assert_not_called()

    @mock.patch("src.data.live.live_sync_job.load_app_config")
    @mock.patch("src.data.live.live_sync_job.LiveDataService")
    @mock.patch("src.data.live.live_sync_job.JobHistory")
    def test_failure_marks_job_failed_and_reraises(self, mock_history_cls, mock_service_cls, mock_load_cfg):
        mock_load_cfg.return_value = _cfg(leagues=[39])
        mock_history = mock_history_cls.return_value
        mock_history.create_job.return_value = {"job_id": "job-2"}
        mock_service = mock_service_cls.return_value
        mock_service.sync_live_data.side_effect = RuntimeError("provider down")

        with self.assertRaises(RuntimeError):
            run_manual_live_sync()

        mock_history.mark_failed.assert_called_once()
        mock_history.mark_success.assert_not_called()


if __name__ == "__main__":
    unittest.main()

