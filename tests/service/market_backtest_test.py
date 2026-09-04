import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np
import pandas as pd

from src.oracle.backtest.market_backtest import (
    _canonical_outcome_for_prediction,
    _outcome_odds_rows_from_raw_market_odds,
    build_backtest_bets_from_oof,
    build_backtest_dataset,
    run_market_backtest,
    run_market_backtest_from_db,
)
from src.oracle.value_engine.value_engine import PLAY, BORDERLINE


def _simple_match(fixture_id, date_iso, market_odds, home_goals, away_goals, market="under_over_2_5"):
    return {
        "id_fixture": fixture_id,
        "season": 2025,
        "status": "FT",
        "date_match": date_iso,
        "current_league": 39,
        "id_team_home": 100,
        "id_team_away": 200,
        "mean_statistics": [
            {"id_team": 100, "mean_Shots on Goal": 6.0},
            {"id_team": 200, "mean_Shots on Goal": 4.0},
        ],
        "statistics": [
            {"statistics_team_id": 100, "score_ft": home_goals},
            {"statistics_team_id": 200, "score_ft": away_goals},
        ],
        "odds": [{market: market_odds}],
    }


def _make_totals_match(fixture_id: int, date: datetime, home_rating: float, away_rating: float, rng: np.random.RandomState) -> dict:
    home_lambda = max(0.2, 1.5 + 0.6 * home_rating)
    away_lambda = max(0.2, 1.2 + 0.6 * away_rating)
    home_goals = int(rng.poisson(home_lambda))
    away_goals = int(rng.poisson(away_lambda))

    return {
        "id_fixture": fixture_id,
        "season": 2025,
        "status": "FT",
        "date_match": date.isoformat(),
        "current_league": 39,
        "id_team_home": 100,
        "id_team_away": 200,
        "mean_statistics": [
            {"id_team": 100, "mean_Shots on Goal": 5.0 + home_rating, "mean_Corner Kicks": 6.0},
            {"id_team": 200, "mean_Shots on Goal": 5.0 + away_rating, "mean_Corner Kicks": 4.0},
        ],
        "statistics": [
            {"statistics_team_id": 100, "score_ft": home_goals},
            {"statistics_team_id": 200, "score_ft": away_goals},
        ],
        "odds": [
            {
                "under_over_2_5": {
                    "over 2.5_bookA": 1.9,
                    "under 2.5_bookA": 1.9,
                    "over 2.5_bookB": 1.95,
                    "under 2.5_bookB": 1.85,
                }
            }
        ],
    }


def _synthetic_totals_matches(n: int = 300, seed: int = 5) -> list[dict]:
    rng = np.random.RandomState(seed)
    base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    matches = []
    for i in range(n):
        home_rating = float(rng.normal(scale=0.9))
        away_rating = float(rng.normal(scale=0.9))
        matches.append(_make_totals_match(2000 + i, base_date + timedelta(days=i), home_rating, away_rating, rng))
    return matches


class TestOutcomeOddsRowsFromRawMarketOdds(unittest.TestCase):
    def test_groups_and_averages_by_outcome(self):
        rows = _outcome_odds_rows_from_raw_market_odds(
            {"over 2.5_bookA": 1.9, "over 2.5_bookB": 2.1, "under 2.5_bookA": 1.9}
        )
        by_outcome = {row["outcome"]: row for row in rows}
        self.assertAlmostEqual(by_outcome["over 2.5"]["avg_odd"], 2.0, places=9)
        self.assertEqual(by_outcome["over 2.5"]["bookmakers"], 2)
        self.assertEqual(by_outcome["under 2.5"]["bookmakers"], 1)

    def test_ignores_non_positive_or_invalid_odds(self):
        rows = _outcome_odds_rows_from_raw_market_odds({"over 2.5_bookA": 0.0, "under 2.5_bookA": "n/a"})
        self.assertEqual(rows, [])

    def test_empty_or_none_input_returns_empty_list(self):
        self.assertEqual(_outcome_odds_rows_from_raw_market_odds(None), [])
        self.assertEqual(_outcome_odds_rows_from_raw_market_odds({}), [])


class TestCanonicalOutcomeForPrediction(unittest.TestCase):
    def test_known_markets(self):
        self.assertEqual(_canonical_outcome_for_prediction("h2h", 1), "Home")
        self.assertEqual(_canonical_outcome_for_prediction("h2h", 0), "Away")
        self.assertEqual(_canonical_outcome_for_prediction("goal_no_goal", 1), "Yes")
        self.assertEqual(_canonical_outcome_for_prediction("goal_no_goal", 0), "No")
        self.assertEqual(_canonical_outcome_for_prediction("dc", 1), "1X")
        self.assertEqual(_canonical_outcome_for_prediction("dc", 0), "X2")
        self.assertEqual(_canonical_outcome_for_prediction("under_over_2_5", 1), "Over 2.5")
        self.assertEqual(_canonical_outcome_for_prediction("under_over_2_5", 0), "Under 2.5")
        self.assertEqual(_canonical_outcome_for_prediction("corners", 1), "Over 9.5")
        self.assertEqual(_canonical_outcome_for_prediction("cards", 0), "Under 4.5")

    def test_unsupported_market_raises(self):
        with self.assertRaises(ValueError):
            _canonical_outcome_for_prediction("not_a_market", 1)


class TestBuildBacktestDataset(unittest.TestCase):
    def test_unsupported_market_raises(self):
        with self.assertRaises(ValueError):
            build_backtest_dataset(matches=[], market="not_a_market")

    def test_empty_matches_returns_empty_frame(self):
        frame, odds_map = build_backtest_dataset(matches=[], market="under_over_2_5")
        self.assertTrue(frame.empty)
        self.assertEqual(odds_map, {})

    def test_frame_is_sorted_by_prediction_at_and_odds_are_captured_per_fixture(self):
        matches = [
            _simple_match(1002, "2026-01-02T18:00:00+00:00", {"over 2.5_bookA": 2.10, "under 2.5_bookA": 1.70}, 0, 0),
            _simple_match(1001, "2026-01-01T18:00:00+00:00", {"over 2.5_bookA": 1.90, "under 2.5_bookA": 1.90}, 2, 1),
        ]
        frame, odds_map = build_backtest_dataset(matches=matches, market="under_over_2_5")

        self.assertEqual(len(frame), 2)
        self.assertEqual(list(frame["id_fixture"]), [1001, 1002])  # riordinato per data, non per ordine di input
        self.assertAlmostEqual(
            next(r["avg_odd"] for r in odds_map[1001] if r["outcome"] == "over 2.5"), 1.90, places=6
        )
        self.assertAlmostEqual(
            next(r["avg_odd"] for r in odds_map[1002] if r["outcome"] == "over 2.5"), 2.10, places=6
        )


class TestBuildBacktestBetsFromOof(unittest.TestCase):
    def test_deterministic_oof_produces_expected_bets_and_odds(self):
        matches = [
            _simple_match(1001, "2026-01-01T18:00:00+00:00", {"over 2.5_bookA": 1.90, "under 2.5_bookA": 1.90}, 2, 1),
            _simple_match(1002, "2026-01-02T18:00:00+00:00", {"over 2.5_bookA": 2.10, "under 2.5_bookA": 1.70}, 0, 0),
        ]
        frame, odds_map = build_backtest_dataset(matches=matches, market="under_over_2_5")
        oof = pd.DataFrame({"index": [0, 1], "probability": [0.90, 0.10], "y_true": [1, 0]})

        bets = build_backtest_bets_from_oof(
            frame=frame, odds_rows_by_fixture=odds_map, oof=oof, market="under_over_2_5"
        )

        self.assertEqual(len(bets), 2)
        bet_by_fixture = {bet.fixture_id: bet for bet in bets}

        first = bet_by_fixture[1001]
        self.assertEqual(first.outcome, "Over 2.5")
        self.assertAlmostEqual(first.odd, 1.90, places=6)
        self.assertAlmostEqual(first.p_model, 0.90, places=6)
        self.assertTrue(first.won)  # y_true=1, prediction=1

        second = bet_by_fixture[1002]
        self.assertEqual(second.outcome, "Under 2.5")
        self.assertAlmostEqual(second.odd, 1.70, places=6)
        self.assertAlmostEqual(second.p_model, 0.90, places=6)  # 1 - 0.10
        self.assertTrue(second.won)  # y_true=0, prediction=0

    def test_missing_quoted_line_falls_back_to_no_bet_not_a_wrong_odd(self):
        # Solo la linea 9.5 e' quotata correttamente per il fixture 1: se il
        # modello "punta" su una riga non quotata, la bet resta NO BET
        # (BET-02) invece di sostituire silenziosamente un'altra quota.
        matches = [
            _simple_match(
                1, "2026-01-01T18:00:00+00:00", {"over 10.5_bookA": 2.0, "under 10.5_bookA": 1.7}, 6, 5, market="corners"
            )
        ]
        frame, odds_map = build_backtest_dataset(matches=matches, market="corners")
        oof = pd.DataFrame({"index": [0], "probability": [0.9], "y_true": [1]})

        bets = build_backtest_bets_from_oof(frame=frame, odds_rows_by_fixture=odds_map, oof=oof, market="corners")
        self.assertEqual(len(bets), 1)
        self.assertEqual(bets[0].outcome, "Over 9.5")
        self.assertIsNone(bets[0].odd)
        self.assertEqual(bets[0].decision, "NO BET")


class TestRunMarketBacktest(unittest.TestCase):
    def test_empty_matches_are_skipped_without_raising(self):
        result = run_market_backtest(matches=[], market="under_over_2_5")
        self.assertEqual(result.status, "skipped_no_data")
        self.assertEqual(result.rows, 0)
        self.assertIsNone(result.report)

    def test_full_pipeline_on_synthetic_matches_is_internally_consistent(self):
        matches = _synthetic_totals_matches(n=300, seed=5)
        result = run_market_backtest(
            matches=matches,
            market="under_over_2_5",
            include_decisions=frozenset({PLAY, BORDERLINE}),
            persist=False,
        )

        self.assertEqual(result.status, "backtested")
        self.assertGreater(result.rows, 0)
        self.assertIsNotNone(result.report)

        report = result.report
        self.assertGreater(report.total_bets_considered, 0)
        self.assertEqual(report.placed_bets + report.skipped_bets, report.total_bets_considered)
        self.assertEqual(report.placed_bets, report.overall.bets)

        if report.overall.bets > 0:
            self.assertGreaterEqual(report.overall.hit_rate, 0.0)
            self.assertLessEqual(report.overall.hit_rate, 1.0)
            self.assertGreater(report.overall.avg_odds, 1.0)
            self.assertGreaterEqual(report.overall.max_drawdown, 0.0)
            self.assertAlmostEqual(report.overall.total_staked, report.overall.bets * report.stake_per_bet, places=6)
            # Nessun mercato diverso da quello richiesto puo' comparire nel breakdown.
            self.assertEqual(set(report.by_market.keys()), {"under_over_2_5"})

    def test_unknown_market_raises(self):
        with self.assertRaises(ValueError):
            run_market_backtest(matches=[{"id_fixture": 1}], market="not_a_market")


class TestRunMarketBacktestFromDb(unittest.TestCase):
    def test_empty_db_result_is_skipped_without_raising(self):
        with mock.patch("src.oracle.backtest.market_backtest.MatchRepository") as repo_cls:
            repo_cls.return_value.search_filter.return_value = []
            result = run_market_backtest_from_db(market="under_over_2_5")

        self.assertEqual(result.status, "skipped_no_data")


if __name__ == "__main__":
    unittest.main()
