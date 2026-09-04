"""Test per BET-06 (Prediction Ledger / Paper Betting).

Due livelli, stesso stile del resto della Fase BETTING:
1. `TestPredictionLedgerPure*`: logica pura (nessun DB), stesso stile di
   `fair_odds_engine.py`/`value_engine.py`.
2. `TestPredictionLedgerServiceWithDb`: orchestrazione DB con un vero
   engine SQLite in-memory (non mock), stesso pattern gia' validato in
   `tests/service/crud_repository_test.py` — esercita DAVVERO
   `PredictionLedgerRepository`/`MatchRepository` con SQLAlchemy reale.
"""

from __future__ import annotations

import dataclasses
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.oracle.backtest.betting_backtester import bet_profit as backtest_bet_profit, BacktestBet
from src.oracle.decision_engine.decision_policy import evaluate_decision
from src.oracle.ledger.ledger_service import PredictionLedgerService
from src.oracle.ledger.prediction_ledger import (
    SETTLED,
    VOID_MISSING_ODD,
    VOID_NO_RESULT,
    build_prediction_record,
    compute_paper_pnl,
    resolve_actual_outcome,
    settle_prediction_record,
)
from src.repository.base.crud_repository import CrudRepository
from src.repository.match_repository import MatchRepository
from src.repository.prediction_ledger_repository import PredictionLedgerRepository
from src.service_ia.model.match import Base, Match, PredictionLedger, Statistics


class TestBuildPredictionRecord(unittest.TestCase):
    def test_build_prediction_record_reuses_decision_fields(self):
        decision = evaluate_decision(
            market="h2h", outcome="Home", p_model=0.7, p_market_fair=0.55, odd=2.1, samples=5
        )
        record = build_prediction_record(
            fixture_id=123, decision=decision, model_run_id="run_1", model_name="model_x", stake=2.0
        )

        self.assertEqual(record.fixture_id, 123)
        self.assertEqual(record.market, "h2h")
        self.assertEqual(record.outcome, "Home")
        self.assertEqual(record.p_model, 0.7)
        self.assertEqual(record.p_market_fair, 0.55)
        self.assertEqual(record.odd, 2.1)
        self.assertAlmostEqual(record.fair_odd, 1.0 / 0.55)
        self.assertEqual(record.prob_edge, decision.prob_edge)
        self.assertEqual(record.ev, decision.ev)
        self.assertEqual(record.decision, decision.decision)
        self.assertEqual(record.policy_version, decision.policy_version)
        self.assertEqual(record.model_run_id, "run_1")
        self.assertEqual(record.model_name, "model_x")
        self.assertEqual(record.stake, 2.0)
        self.assertIsNotNone(record.created_at)
        # Immutabilita' logica: nessun campo di settlement popolato alla creazione.
        self.assertFalse(record.is_settled)
        self.assertIsNone(record.won)
        self.assertIsNone(record.pnl)
        self.assertIsNone(record.settlement_status)


class TestResolveActualOutcome(unittest.TestCase):
    def test_h2h_home_win(self):
        stat_home = {"score_ft": 2}
        stat_away = {"score_ft": 1}
        self.assertEqual(resolve_actual_outcome("h2h", stat_home, stat_away), "Home")

    def test_h2h_away_win(self):
        stat_home = {"score_ft": 0}
        stat_away = {"score_ft": 3}
        self.assertEqual(resolve_actual_outcome("h2h", stat_home, stat_away), "Away")

    def test_under_over_2_5_over(self):
        stat_home = {"score_ft": 2}
        stat_away = {"score_ft": 1}
        self.assertEqual(resolve_actual_outcome("under_over_2_5", stat_home, stat_away), "Over 2.5")

    def test_under_over_2_5_under(self):
        stat_home = {"score_ft": 1}
        stat_away = {"score_ft": 0}
        self.assertEqual(resolve_actual_outcome("under_over_2_5", stat_home, stat_away), "Under 2.5")

    def test_goal_no_goal_yes(self):
        stat_home = {"score_ft": 1}
        stat_away = {"score_ft": 1}
        self.assertEqual(resolve_actual_outcome("goal_no_goal", stat_home, stat_away), "Yes")

    def test_dc(self):
        stat_home = {"score_ft": 1}
        stat_away = {"score_ft": 1}
        self.assertEqual(resolve_actual_outcome("dc", stat_home, stat_away), "1X")

    def test_none_when_scores_missing(self):
        self.assertIsNone(resolve_actual_outcome("h2h", {"score_ft": None}, {"score_ft": 1}))
        self.assertIsNone(resolve_actual_outcome("h2h", None, None))

    def test_none_when_market_unsupported(self):
        self.assertIsNone(resolve_actual_outcome("not_a_market", {"score_ft": 1}, {"score_ft": 0}))


class TestComputePaperPnl(unittest.TestCase):
    def test_matches_betting_backtester_formula(self):
        """Coerenza numerica ESPLICITA con `bet_profit` (BET-03): stessa
        formula, due implementazioni indipendenti (modulo ledger
        deliberatamente disaccoppiato dal backtest storico)."""
        for odd, won, stake in [(2.5, True, 1.0), (1.8, False, 1.0), (3.2, True, 2.0)]:
            expected = backtest_bet_profit(
                BacktestBet(
                    market="h2h", outcome="Home", p_model=0.6, p_market_fair=0.5, odd=odd,
                    prob_edge=0.1, ev=0.1, decision="PLAY", policy_version="v1", won=won,
                ),
                stake=stake,
            )
            self.assertAlmostEqual(compute_paper_pnl(odd=odd, won=won, stake=stake), expected)

    def test_none_when_odd_or_won_missing(self):
        self.assertIsNone(compute_paper_pnl(odd=None, won=True, stake=1.0))
        self.assertIsNone(compute_paper_pnl(odd=2.0, won=None, stake=1.0))

    def test_none_when_odd_not_positive(self):
        self.assertIsNone(compute_paper_pnl(odd=0.0, won=True, stake=1.0))


class TestSettlePredictionRecord(unittest.TestCase):
    def _record(self, **overrides):
        decision = evaluate_decision(market="h2h", outcome="Home", p_model=0.7, p_market_fair=0.5, odd=2.0, samples=5)
        record = build_prediction_record(fixture_id=1, decision=decision, model_run_id="run_1", stake=1.0)
        return dataclasses.replace(record, **overrides) if overrides else record

    def test_settled_won(self):
        record = self._record()
        settled = settle_prediction_record(record=record, actual_outcome="Home")

        self.assertTrue(settled.is_settled)
        self.assertEqual(settled.settlement_status, SETTLED)
        self.assertTrue(settled.won)
        self.assertEqual(settled.actual_outcome, "Home")
        self.assertAlmostEqual(settled.pnl, 1.0)  # stake 1.0 * (odd 2.0 - 1)

    def test_settled_lost(self):
        record = self._record()
        settled = settle_prediction_record(record=record, actual_outcome="Away")

        self.assertEqual(settled.settlement_status, SETTLED)
        self.assertFalse(settled.won)
        self.assertAlmostEqual(settled.pnl, -1.0)

    def test_void_missing_odd(self):
        record = self._record(odd=None)
        settled = settle_prediction_record(record=record, actual_outcome="Home")

        self.assertEqual(settled.settlement_status, VOID_MISSING_ODD)
        self.assertTrue(settled.won)  # l'esito resta calcolabile (utile per hit-rate)
        self.assertIsNone(settled.pnl)

    def test_void_no_result(self):
        record = self._record()
        settled = settle_prediction_record(record=record, actual_outcome=None)

        self.assertEqual(settled.settlement_status, VOID_NO_RESULT)
        self.assertIsNone(settled.won)
        self.assertIsNone(settled.pnl)

    def test_does_not_mutate_original_record(self):
        """Immutabilita' logica (acceptance criteria BET-06): l'oggetto
        PredictionRecord originale passato a `settle_prediction_record`
        resta ESATTAMENTE invariato dopo la chiamata."""
        record = self._record()
        original_snapshot = dataclasses.replace(record)

        settled = settle_prediction_record(record=record, actual_outcome="Home")

        self.assertEqual(record, original_snapshot)
        self.assertIsNot(settled, record)
        self.assertFalse(record.is_settled)


def _make_in_memory_engine():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return engine


def _add_match(session, *, fixture_id: int, status: str, home_score, away_score, id_team_home=100, id_team_away=200):
    match_id = str(uuid.uuid4())
    match = Match(
        id_match_fk=match_id,
        id_fixture=fixture_id,
        status=status,
        id_team_home=id_team_home,
        id_team_away=id_team_away,
    )
    session.add(match)
    if home_score is not None and away_score is not None:
        session.add(Statistics(id_statistics_fk=str(uuid.uuid4()), id_match=match_id, statistics_team_id=id_team_home, score_ft=home_score))
        session.add(Statistics(id_statistics_fk=str(uuid.uuid4()), id_match=match_id, statistics_team_id=id_team_away, score_ft=away_score))
    session.commit()
    return match_id


class TestPredictionLedgerServiceWithDb(unittest.TestCase):
    """Esercita DAVVERO SQLAlchemy (SQLite in-memory condiviso tra
    MatchRepository e PredictionLedgerRepository via StaticPool), stesso
    principio di `crud_repository_test.py`: nessun mock sulla query."""

    def setUp(self):
        self.engine = _make_in_memory_engine()
        self.session_factory = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)

        self.match_repo = MatchRepository()
        self.match_repo.session.close()
        self.match_repo.session = self.session_factory()

        self._session_local_patch = mock.patch(
            "src.repository.prediction_ledger_repository.SessionLocal", new=self.session_factory
        )
        self._session_local_patch.start()
        self.addCleanup(self._session_local_patch.stop)

        self.ledger_repo = PredictionLedgerRepository()
        self.service = PredictionLedgerService(repo=self.ledger_repo, match_repo=self.match_repo)

    def _decision(self, market="h2h", outcome="Home", odd=2.0):
        return evaluate_decision(market=market, outcome=outcome, p_model=0.7, p_market_fair=0.5, odd=odd, samples=5)

    def test_log_prediction_creates_row(self):
        row = self.service.log_prediction(fixture_id=42, decision=self._decision(), model_run_id="run_1")

        self.assertIsNotNone(row.id_prediction)
        self.assertEqual(row.fixture_id, 42)
        self.assertEqual(row.market, "h2h")
        self.assertEqual(row.outcome, "Home")
        self.assertFalse(row.is_settled)

    def test_log_prediction_is_idempotent_with_dedupe(self):
        first = self.service.log_prediction(fixture_id=42, decision=self._decision(), model_run_id="run_1")
        second = self.service.log_prediction(fixture_id=42, decision=self._decision(), model_run_id="run_1")

        self.assertEqual(first.id_prediction, second.id_prediction)
        rows = self.ledger_repo.list_all()
        self.assertEqual(len(rows), 1)

    def test_log_prediction_without_dedupe_creates_new_row(self):
        self.service.log_prediction(fixture_id=42, decision=self._decision(), model_run_id="run_1")
        self.service.log_prediction(fixture_id=42, decision=self._decision(), model_run_id="run_1", dedupe=False)

        self.assertEqual(len(self.ledger_repo.list_all()), 2)

    def test_settle_prediction_marks_won_and_computes_pnl(self):
        _add_match(self.match_repo.session, fixture_id=555, status="FT", home_score=2, away_score=1)
        row = self.service.log_prediction(fixture_id=555, decision=self._decision(market="h2h", outcome="Home", odd=2.0))

        settled = self.service.settle_prediction(row.id_prediction)

        self.assertTrue(settled.is_settled)
        self.assertEqual(settled.actual_outcome, "Home")
        self.assertTrue(settled.won)
        self.assertAlmostEqual(settled.pnl, 1.0)
        # Immutabilita': i campi originali non sono cambiati dal settlement.
        self.assertEqual(settled.market, "h2h")
        self.assertEqual(settled.outcome, "Home")
        self.assertEqual(settled.odd, 2.0)

    def test_settle_prediction_lost(self):
        _add_match(self.match_repo.session, fixture_id=556, status="FT", home_score=0, away_score=1)
        row = self.service.log_prediction(fixture_id=556, decision=self._decision(market="h2h", outcome="Home", odd=2.0))

        settled = self.service.settle_prediction(row.id_prediction)

        self.assertFalse(settled.won)
        self.assertAlmostEqual(settled.pnl, -1.0)

    def test_settle_pending_skips_non_final_match(self):
        _add_match(self.match_repo.session, fixture_id=557, status="NS", home_score=None, away_score=None)
        row = self.service.log_prediction(
            fixture_id=557,
            decision=self._decision(),
            kickoff_at=datetime.now(timezone.utc) - timedelta(hours=3),
        )

        report = self.service.settle_pending()

        self.assertEqual(report["still_pending"], 1)
        self.assertEqual(report["settled"], 0)
        refreshed = self.ledger_repo.get_by_id(row.id_prediction)
        self.assertFalse(refreshed.is_settled)

    def test_settle_pending_settles_final_match(self):
        _add_match(self.match_repo.session, fixture_id=558, status="FT", home_score=2, away_score=0)
        self.service.log_prediction(
            fixture_id=558,
            decision=self._decision(market="h2h", outcome="Home", odd=1.9),
            kickoff_at=datetime.now(timezone.utc) - timedelta(hours=3),
        )

        report = self.service.settle_pending()

        self.assertEqual(report["settled"], 1)
        self.assertEqual(report["still_pending"], 0)

    def test_settle_prediction_void_when_match_missing_statistics(self):
        """Match concluso ma senza statistiche complete: settlement
        esplicito 'void_no_result', mai un esito inventato."""
        _add_match(self.match_repo.session, fixture_id=559, status="FT", home_score=None, away_score=None)
        row = self.service.log_prediction(fixture_id=559, decision=self._decision())

        settled = self.service.settle_prediction(row.id_prediction)

        self.assertEqual(settled.settlement_status, VOID_NO_RESULT)
        self.assertIsNone(settled.won)

    def test_raw_pnl_summary_and_paper_pnl_report(self):
        _add_match(self.match_repo.session, fixture_id=560, status="FT", home_score=2, away_score=0)
        _add_match(self.match_repo.session, fixture_id=561, status="FT", home_score=0, away_score=1)

        row_win = self.service.log_prediction(fixture_id=560, decision=self._decision(market="h2h", outcome="Home", odd=2.0))
        row_lose = self.service.log_prediction(fixture_id=561, decision=self._decision(market="h2h", outcome="Home", odd=1.5))
        self.service.settle_prediction(row_win.id_prediction)
        self.service.settle_prediction(row_lose.id_prediction)

        summary = self.service.raw_pnl_summary(market="h2h")
        self.assertEqual(summary["settled_total"], 2)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertAlmostEqual(summary["total_pnl"], 1.0 + (-1.0))

        report = self.service.paper_pnl_report(market="h2h")
        self.assertEqual(report.overall.bets, 2)
        self.assertEqual(report.overall.wins, 1)

    def test_list_all_since_filters_by_created_at(self):
        """OPS-03: estensione opzionale (default `None` = comportamento
        INVARIATO, verificato dagli altri test di questa classe che non
        passano mai `since`) - con `since` esplicito, solo le righe con
        `created_at >= since` sono ritornate."""
        old_row = PredictionLedger(
            fixture_id=900, market="h2h", outcome="Home", decision="PLAY", stake=1.0,
            created_at=datetime.now(timezone.utc) - timedelta(days=40),
        )
        recent_row = PredictionLedger(
            fixture_id=901, market="h2h", outcome="Home", decision="PLAY", stake=1.0,
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        self.ledger_repo.save(old_row)
        self.ledger_repo.save(recent_row)

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        rows = self.ledger_repo.list_all(market="h2h", since=cutoff)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].fixture_id, 901)


if __name__ == "__main__":
    unittest.main()
