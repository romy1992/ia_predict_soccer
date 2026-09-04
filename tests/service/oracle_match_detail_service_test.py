import unittest
from unittest import mock

from src.api.oracle_match_detail_service import OracleMatchDetailService


def _decision_card(market, value_label, probability=0.6):
    return {
        "market": market,
        "pick": "Home",
        "value_label": value_label,
        "predicted_probability": probability,
        "odd": 1.8,
        "fair_odd": 1.9,
        "edge": 0.02,
        "ev": 0.05,
    }


class TestOracleMatchDetailService(unittest.TestCase):
    @staticmethod
    def _service():
        return OracleMatchDetailService(
            dashboard_service=mock.Mock(),
            match_repo=mock.Mock(),
            team_strength_expert=mock.Mock(),
            goal_distribution_expert=mock.Mock(),
            odds_snapshot_repo=mock.Mock(),
        )

    def test_full_happy_path_all_sections_available(self):
        service = self._service()
        service.dashboard_service.get_match_detail.return_value = {
            "fixture": {"fixture_id": 10, "home": "Inter", "away": "Milan"},
            "timeline": [{"minute": "10'"}],
            "odds_summary": {"h2h": []},
            "bookmaker_baseline": {"markets": {}, "generated": True},
            "odds_updated_at": "2026-01-01T00:00:00+00:00",
            "decision_cards": [
                _decision_card("h2h", "PLAY"),
                _decision_card("dc", "NO BET"),
                _decision_card("goal_no_goal", "BORDERLINE"),
            ],
            "predictions": {},
            "model_markets": ["h2h", "dc", "goal_no_goal"],
        }

        match_row = mock.Mock(id_team_home=1, id_team_away=2, date_match="2026-01-05T18:00:00+00:00")
        service.match_repo.filter_by.return_value.first.return_value = match_row
        service.match_repo.search_filter.return_value = ["orm_row"]

        service.team_strength_expert.current_ratings.return_value = {
            1: {"team_attack_rating": 1.6, "team_defense_rating": 0.9, "rating_version": "v1"},
            2: {"team_attack_rating": 1.1, "team_defense_rating": 1.3, "rating_version": "v1"},
        }
        service.goal_distribution_expert.estimate_lambdas_from_ratings.return_value = (1.4, 1.0)
        service.goal_distribution_expert.build_expert_output.return_value = {
            "expert_version": "gd1",
            "home_lambda": 1.4,
            "away_lambda": 1.0,
            "total_lambda": 2.4,
            "over_under": {"over_2_5": 0.45},
            "over_under_from_score_matrix": {"over_2_5": 0.44},
            "score_matrix": {0: {0: 0.1, 1: 0.05}, 1: {0: 0.08, 1: 0.04}},
        }

        service.odds_snapshot_repo.opening_latest_closing.return_value = [
            {
                "bookmaker": "BookA",
                "market": "h2h",
                "outcome": "Home",
                "opening": {"odd": 1.9},
                "latest": {"odd": 1.8},
                "closing": None,
            }
        ]

        with mock.patch(
            "src.api.oracle_match_detail_service.convert_orm_match_to_dict",
            return_value=[{"date_match": "2026-01-01T00:00:00+00:00", "id_team_home": 1, "id_team_away": 2, "status": "FT"}],
        ), mock.patch("src.api.oracle_match_detail_service.build_model_consensus_for_fixture") as consensus_mock:
            consensus_mock.return_value = mock.Mock(
                experts=[{"expert_name": "market_odds"}],
                oracle_final={"probability": 0.6, "source": "simple_consensus_mean"},
                consensus={"agreement_level": "high"},
                warnings=[],
            )
            payload = service.build_oracle_match_detail(fixture_id=10)

        self.assertEqual(payload["fixture_id"], 10)
        self.assertEqual(payload["overview"]["fixture"]["home"], "Inter")
        self.assertEqual(len(payload["probabilities"]), 3)
        self.assertEqual({c["market"] for c in payload["value_bets"]}, {"h2h", "goal_no_goal"})
        self.assertIsNotNone(payload["team_strength"])
        self.assertEqual(payload["team_strength"]["home"]["team_attack_rating"], 1.6)
        self.assertIsNotNone(payload["expected_goals"])
        self.assertEqual(payload["expected_goals"]["home_lambda"], 1.4)
        self.assertNotIn("score_matrix", payload["expected_goals"])
        self.assertEqual(payload["score_matrix"]["0"]["1"], 0.05)
        self.assertTrue(payload["odds_movement"])
        self.assertIn("h2h", payload["model_consensus"])
        self.assertEqual(payload["model_consensus"]["h2h"]["oracle_final"]["probability"], 0.6)
        self.assertEqual(payload["warnings"], [])

    def test_missing_fixture_reports_warning_but_does_not_crash(self):
        service = self._service()
        service.dashboard_service.get_match_detail.return_value = {
            "fixture": None,
            "timeline": [],
            "odds_summary": {},
            "bookmaker_baseline": {},
            "decision_cards": [],
            "predictions": {},
            "model_markets": [],
        }
        service.match_repo.filter_by.return_value.first.return_value = None
        service.odds_snapshot_repo.opening_latest_closing.return_value = []

        payload = service.build_oracle_match_detail(fixture_id=999)

        self.assertIsNone(payload["overview"]["fixture"])
        self.assertIn("fixture_not_found", payload["warnings"])
        self.assertIn("team_strength_not_available", payload["warnings"])
        self.assertIn("expected_goals_not_available", payload["warnings"])
        self.assertIn("odds_movement_not_available", payload["warnings"])
        self.assertIn("model_consensus_not_available", payload["warnings"])
        self.assertIsNone(payload["team_strength"])
        self.assertIsNone(payload["score_matrix"])

    def test_team_strength_returns_none_without_team_ids(self):
        service = self._service()
        self.assertIsNone(service._team_strength_for_teams(None, 2, before="2026-01-01T00:00:00+00:00"))
        self.assertIsNone(service._team_strength_for_teams(1, None, before="2026-01-01T00:00:00+00:00"))

    def test_team_strength_excludes_matches_after_cutoff(self):
        service = self._service()
        service.match_repo.search_filter.return_value = ["a", "b"]
        service.team_strength_expert.current_ratings.return_value = {
            1: {"team_attack_rating": 1.0},
            2: {"team_attack_rating": 1.0},
        }
        with mock.patch(
            "src.api.oracle_match_detail_service.convert_orm_match_to_dict",
            return_value=[
                {"date_match": "2025-01-01T00:00:00+00:00"},  # prima del cutoff -> incluso
                {"date_match": "2026-06-01T00:00:00+00:00"},  # dopo il cutoff -> escluso (no leakage)
            ],
        ):
            result = service._team_strength_for_teams(1, 2, before="2026-01-01T00:00:00+00:00")

        self.assertIsNotNone(result)
        self.assertEqual(result["sample_matches"], 1)

    def test_model_consensus_partial_failure_does_not_block_other_markets(self):
        service = self._service()
        with mock.patch("src.api.oracle_match_detail_service.build_model_consensus_for_fixture") as consensus_mock:
            def side_effect(market, fixture_id):
                if market == "h2h":
                    raise RuntimeError("boom")
                return mock.Mock(experts=[], oracle_final=None, consensus={}, warnings=[])

            consensus_mock.side_effect = side_effect

            result = service._model_consensus_by_market(fixture_id=1, model_markets=["h2h", "dc"])

        self.assertNotIn("h2h", result)
        self.assertIn("dc", result)


if __name__ == "__main__":
    unittest.main()
