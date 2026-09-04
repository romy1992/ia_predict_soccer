"""Test per LIVE-02 (Feature store live).

Due livelli, stesso stile di `live_data_service_test.py` (LIVE-01):
1. `TestPureFeatureExtractors`: funzioni pure di estrazione feature (nessun
   DB) - minute/scoreline/cards/stat/prior.
2. `TestLiveFeatureStoreWithDb`: vero SQLite in-memory (non mock) - esercita
   `LiveFeatureStore` end-to-end, incluso il filtro point-in-time su `as_of`
   (acceptance criteria "Ogni feature live timestamped": niente eventi/
   statistiche/prior con timestamp successivo ad `as_of`).
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.data.live.live_models as live_models  # noqa: F401  (registra le tabelle live su Base.metadata)
from src.data.live.live_models import Base, LiveFixtureSnapshot, LiveFixtureStatSnapshot, LiveMatchEvent
from src.ml.live.live_feature_store import (
    LiveFeatureRow,
    LiveFeatureStore,
    card_features_from_events,
    minute_features,
    pre_match_prior_features,
    scoreline_features,
    stat_features,
)
from src.repository.live_data_repository import LiveDataRepository
from src.repository.prediction_ledger_repository import PredictionLedgerRepository
from src.service_ia.model.match import PredictionLedger

T0 = datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc)


def _snapshot(
    fixture_id=100,
    captured_at=T0,
    minute=30,
    extra=None,
    status="1H",
    home_goals=1,
    away_goals=0,
    home_team_id=33,
    away_team_id=34,
):
    return LiveFixtureSnapshot(
        fixture_id=fixture_id,
        captured_at=captured_at,
        status=status,
        elapsed_minute=minute,
        elapsed_extra=extra,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_goals=home_goals,
        away_goals=away_goals,
    )


def _card_event(fixture_id=100, captured_at=T0, minute=30, team_id=33, detail="Yellow Card", id_event=None):
    return LiveMatchEvent(
        id_event=id_event or f"card-{team_id}-{minute}-{detail}-{captured_at.isoformat()}",
        fixture_id=fixture_id,
        captured_at=captured_at,
        elapsed_minute=minute,
        event_type="Card",
        event_detail=detail,
        team_id=team_id,
    )


def _stat_snapshot(fixture_id=100, captured_at=T0, team_id=33, stats=None):
    return LiveFixtureStatSnapshot(
        id_snapshot=f"stat-{team_id}-{captured_at.isoformat()}",
        fixture_id=fixture_id,
        team_id=team_id,
        captured_at=captured_at,
        stats=stats if stats is not None else [{"type": "Shots on Goal", "value": 3}],
    )


class TestPureFeatureExtractors(unittest.TestCase):
    def test_minute_features_none_snapshot(self):
        features = minute_features(None)
        self.assertEqual(features, {"minute": None, "minute_extra": None, "minute_total": None, "status": None})

    def test_minute_features_with_extra_time(self):
        features = minute_features(_snapshot(minute=45, extra=3))
        self.assertEqual(features["minute"], 45)
        self.assertEqual(features["minute_extra"], 3)
        self.assertEqual(features["minute_total"], 48)
        self.assertEqual(features["status"], "1H")

    def test_scoreline_features(self):
        features = scoreline_features(_snapshot(home_goals=2, away_goals=1))
        self.assertEqual(features, {"home_goals": 2, "away_goals": 1, "goal_diff": 1, "total_goals": 3})

    def test_scoreline_features_none_snapshot(self):
        features = scoreline_features(None)
        self.assertIsNone(features["home_goals"])
        self.assertIsNone(features["goal_diff"])

    def test_card_features_counts_per_team_and_ignores_unknown_team(self):
        events = [
            _card_event(team_id=33, detail="Yellow Card", minute=10),
            _card_event(team_id=33, detail="Yellow Card", minute=20),
            _card_event(team_id=34, detail="Red Card", minute=30),
            _card_event(team_id=999, detail="Yellow Card", minute=40),  # team sconosciuto -> ignorato
            _card_event(team_id=33, detail="Second Yellow Card", minute=50),
        ]
        features = card_features_from_events(events, home_team_id=33, away_team_id=34)
        self.assertEqual(features["yellow_cards_home"], 2)
        self.assertEqual(features["yellow_cards_away"], 0)
        self.assertEqual(features["red_cards_home"], 1)  # second yellow conta come rosso
        self.assertEqual(features["red_cards_away"], 1)
        self.assertEqual(features["yellow_cards_diff"], 2)
        self.assertEqual(features["red_cards_diff"], 0)

    def test_stat_features_only_includes_available_values(self):
        home_stat = _stat_snapshot(
            team_id=33,
            stats=[
                {"type": "Shots on Goal", "value": 5},
                {"type": "Total Shots", "value": 10},
                {"type": "expected_goals", "value": None},  # non disponibile -> omessa
                {"type": "Ball Possession", "value": "60%"},
            ],
        )
        away_stat = _stat_snapshot(
            team_id=34,
            stats=[
                {"type": "Shots on Goal", "value": 2},
                {"type": "expected_goals", "value": "0.8"},  # solo away la fornisce
            ],
        )
        features = stat_features([home_stat, away_stat], home_team_id=33, away_team_id=34)

        self.assertEqual(features["shots_on_goal_home"], 5.0)
        self.assertEqual(features["shots_on_goal_away"], 2.0)
        self.assertEqual(features["shots_on_goal_diff"], 3.0)
        self.assertEqual(features["total_shots_home"], 10.0)
        self.assertNotIn("total_shots_away", features)
        self.assertNotIn("total_shots_diff", features)
        self.assertEqual(features["ball_possession_pct_home"], 60.0)
        # expected_goals: home None (omessa), away disponibile (0.8) -> nessun diff (manca l'home)
        self.assertNotIn("expected_goals_home", features)
        self.assertEqual(features["expected_goals_away"], 0.8)
        self.assertNotIn("expected_goals_diff", features)

    def test_stat_features_empty_when_no_snapshots(self):
        self.assertEqual(stat_features([], home_team_id=33, away_team_id=34), {})

    def test_pre_match_prior_features_keeps_latest_per_market_outcome(self):
        older = PredictionLedger(
            fixture_id=1, market="h2h", outcome="Home", decision="PLAY", stake=1.0,
            p_model=0.5, p_market_fair=0.45, prob_edge=0.05,
            created_at=T0 - timedelta(hours=2),
        )
        newer = PredictionLedger(
            fixture_id=1, market="h2h", outcome="Home", decision="PLAY", stake=1.0,
            p_model=0.6, p_market_fair=0.5, prob_edge=0.1,
            created_at=T0 - timedelta(hours=1),
        )
        other_market = PredictionLedger(
            fixture_id=1, market="under_over_2_5", outcome="Over", decision="PLAY", stake=1.0,
            p_model=0.7, created_at=T0 - timedelta(hours=1),
        )
        features = pre_match_prior_features([older, newer, other_market])

        self.assertEqual(features["prior__h2h__home__p_model"], 0.6)
        self.assertEqual(features["prior__h2h__home__p_market_fair"], 0.5)
        self.assertEqual(features["prior__h2h__home__prob_edge"], 0.1)
        self.assertEqual(features["prior__under_over_2_5__over__p_model"], 0.7)
        self.assertNotIn("prior__under_over_2_5__over__p_market_fair", features)


def _make_in_memory_engine():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return engine


class TestLiveFeatureStoreWithDb(unittest.TestCase):
    """Vero SQLite in-memory - stesso principio di `TestLiveDataRepositoryWithDb`."""

    def setUp(self):
        self.engine = _make_in_memory_engine()
        self.session_factory = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)

        self._live_patch = mock.patch(
            "src.repository.live_data_repository.SessionLocal", new=self.session_factory
        )
        self._ledger_patch = mock.patch(
            "src.repository.prediction_ledger_repository.SessionLocal", new=self.session_factory
        )
        self._live_patch.start()
        self._ledger_patch.start()
        self.addCleanup(self._live_patch.stop)
        self.addCleanup(self._ledger_patch.stop)

        self.live_repo = LiveDataRepository()
        self.ledger_repo = PredictionLedgerRepository()
        self.store = LiveFeatureStore(live_repository=self.live_repo, prediction_repository=self.ledger_repo)

    def test_build_feature_row_returns_none_when_no_snapshot_available_yet(self):
        self.live_repo.save_fixture_snapshots([_snapshot(captured_at=T0)])

        row = self.store.build_feature_row(fixture_id=100, as_of=T0 - timedelta(minutes=5))
        self.assertIsNone(row)

    def test_build_feature_row_assembles_minute_scoreline_cards_and_stats(self):
        self.live_repo.save_fixture_snapshots([_snapshot(captured_at=T0, minute=30, home_goals=1, away_goals=0)])
        self.live_repo.save_events(
            [
                _card_event(captured_at=T0, minute=20, team_id=33, detail="Yellow Card"),
                _card_event(captured_at=T0, minute=25, team_id=34, detail="Yellow Card", id_event="ev-2"),
            ]
        )
        self.live_repo.save_stat_snapshots(
            [
                _stat_snapshot(captured_at=T0, team_id=33, stats=[{"type": "Shots on Goal", "value": 4}]),
                _stat_snapshot(captured_at=T0, team_id=34, stats=[{"type": "Shots on Goal", "value": 2}]),
            ]
        )

        row = self.store.build_feature_row(fixture_id=100, as_of=T0)
        self.assertIsInstance(row, LiveFeatureRow)
        self.assertEqual(row.fixture_id, 100)
        self.assertEqual(row.as_of, T0.isoformat())
        self.assertEqual(row.features["minute"], 30)
        self.assertEqual(row.features["home_goals"], 1)
        self.assertEqual(row.features["goal_diff"], 1)
        self.assertEqual(row.features["yellow_cards_home"], 1)
        self.assertEqual(row.features["yellow_cards_away"], 1)
        self.assertEqual(row.features["shots_on_goal_home"], 4.0)
        self.assertEqual(row.features["shots_on_goal_away"], 2.0)
        self.assertIsNotNone(row.feature_available_at_max)

    def test_build_feature_row_excludes_future_events_and_stats(self):
        """Acceptance criteria "Ogni feature live timestamped": un evento/
        statistica registrati DOPO `as_of` non devono comparire nella riga."""
        self.live_repo.save_fixture_snapshots(
            [
                _snapshot(captured_at=T0, minute=10, home_goals=0, away_goals=0),
                _snapshot(captured_at=T0 + timedelta(minutes=50), minute=60, home_goals=1, away_goals=0),
            ]
        )
        self.live_repo.save_events(
            [
                _card_event(captured_at=T0, minute=5, team_id=33, detail="Yellow Card"),
                _card_event(
                    captured_at=T0 + timedelta(minutes=50), minute=58, team_id=33, detail="Yellow Card", id_event="ev-late"
                ),
            ]
        )

        as_of_first_half = T0 + timedelta(minutes=1)
        row = self.store.build_feature_row(fixture_id=100, as_of=as_of_first_half)

        self.assertEqual(row.features["minute"], 10)  # NON lo snapshot al minuto 60
        self.assertEqual(row.features["yellow_cards_home"], 1)  # NON il secondo cartellino (futuro)
        self.assertEqual(row.features["home_goals"], 0)

    def test_build_feature_row_includes_pre_match_prior(self):
        self.live_repo.save_fixture_snapshots([_snapshot(captured_at=T0)])
        self.ledger_repo.save(
            PredictionLedger(
                fixture_id=100, market="h2h", outcome="Home", decision="PLAY", stake=1.0,
                p_model=0.55, p_market_fair=0.5, prob_edge=0.05,
                created_at=T0 - timedelta(hours=3),
            )
        )

        row = self.store.build_feature_row(fixture_id=100, as_of=T0)
        self.assertEqual(row.features["prior__h2h__home__p_model"], 0.55)
        self.assertEqual(row.features["prior__h2h__home__p_market_fair"], 0.5)

    def test_build_feature_history_returns_one_row_per_snapshot(self):
        self.live_repo.save_fixture_snapshots(
            [
                _snapshot(captured_at=T0, minute=10),
                _snapshot(captured_at=T0 + timedelta(minutes=20), minute=30),
                _snapshot(captured_at=T0 + timedelta(minutes=40), minute=50),
            ]
        )

        rows = self.store.build_feature_history(fixture_id=100)
        self.assertEqual(len(rows), 3)
        self.assertEqual([row.features["minute"] for row in rows], [10, 30, 50])

    def test_assert_no_leakage_true_for_history(self):
        self.live_repo.save_fixture_snapshots(
            [_snapshot(captured_at=T0, minute=10), _snapshot(captured_at=T0 + timedelta(minutes=20), minute=30)]
        )
        rows = self.store.build_feature_history(fixture_id=100)
        self.assertTrue(LiveFeatureStore.assert_no_leakage(rows))

    def test_assert_no_leakage_false_when_tampered(self):
        row = LiveFeatureRow(
            fixture_id=100,
            as_of=T0.isoformat(),
            feature_available_at_max=(T0 + timedelta(minutes=5)).isoformat(),
        )
        self.assertFalse(LiveFeatureStore.assert_no_leakage([row]))


if __name__ == "__main__":
    unittest.main()

