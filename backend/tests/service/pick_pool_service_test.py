import unittest
from datetime import date
from unittest import mock

from src.oracle.betslip.pick_pool import PickPoolPolicy
from src.oracle.betslip.pick_pool_service import PickPoolService, candidate_from_decision_card


def _decision_card(market="h2h", pick="Home", value_label="PLAY", odd=2.0, ev=0.08, edge=0.05, run_id="run-1"):
    return {
        "market": market,
        "pick": pick,
        "value_label": value_label,
        "predicted_probability": 0.65,
        "bookmaker_fair_probability": 0.55,
        "odd": odd,
        "fair_odd": 1.8,
        "edge": edge,
        "ev": ev,
        "bookmakers_count": 6,
        "run_id": run_id,
        "model_name": "logistic",
        "policy_version": "decision_policy_v1",
    }


class TestCandidateFromDecisionCard(unittest.TestCase):
    def test_maps_all_fields_without_recomputing(self):
        card = _decision_card()
        candidate = candidate_from_decision_card(fixture_id=101, kickoff_at="2026-09-05T18:00:00+00:00", card=card)

        self.assertEqual(candidate.fixture_id, 101)
        self.assertEqual(candidate.market, "h2h")
        self.assertEqual(candidate.outcome, "Home")
        self.assertEqual(candidate.decision, "PLAY")
        self.assertEqual(candidate.odd, 2.0)
        self.assertEqual(candidate.fair_odd, 1.8)
        self.assertEqual(candidate.ev, 0.08)
        self.assertEqual(candidate.prob_edge, 0.05)
        self.assertEqual(candidate.samples, 6)
        self.assertEqual(candidate.model_run_id, "run-1")
        self.assertEqual(candidate.kickoff_at, "2026-09-05T18:00:00+00:00")


class TestPickPoolService(unittest.TestCase):
    def _service_with_day_rows(self, rows):
        service = PickPoolService(dashboard_service=mock.Mock())
        service.dashboard_service.get_day_matches.return_value = mock.Mock(rows=rows)
        return service

    def test_candidates_for_day_maps_decision_cards(self):
        rows = [
            {
                "fixture_id": 1,
                "datetime": "2026-09-05T18:00:00+00:00",
                "decision_cards": [_decision_card(market="h2h"), _decision_card(market="goal_no_goal", pick="Yes")],
            },
            {"fixture_id": 2, "datetime": "2026-09-05T20:00:00+00:00", "decision_cards": []},
        ]
        service = self._service_with_day_rows(rows)

        candidates = service.candidates_for_day(target_date=date(2026, 9, 5))

        self.assertEqual(len(candidates), 2)
        self.assertEqual({c.market for c in candidates}, {"h2h", "goal_no_goal"})
        self.assertTrue(all(c.fixture_id == 1 for c in candidates))

    def test_candidates_for_day_skips_rows_without_fixture_id(self):
        rows = [{"fixture_id": None, "decision_cards": [_decision_card()]}]
        service = self._service_with_day_rows(rows)

        candidates = service.candidates_for_day(target_date=date(2026, 9, 5))
        self.assertEqual(candidates, [])

    def test_build_pool_for_day_applies_policy_end_to_end(self):
        rows = [
            {
                "fixture_id": 1,
                "datetime": "2026-09-05T18:00:00+00:00",
                "decision_cards": [
                    _decision_card(market="h2h", value_label="PLAY", odd=2.0, ev=0.10, run_id="a"),
                    _decision_card(market="dc", value_label="BORDERLINE", odd=1.3, ev=0.01, run_id="b"),
                ],
            },
            {
                "fixture_id": 2,
                "datetime": "2026-09-05T20:00:00+00:00",
                "decision_cards": [_decision_card(market="h2h", value_label="NO BET", odd=1.1, ev=-0.05, run_id="c")],
            },
        ]
        service = self._service_with_day_rows(rows)

        result = service.build_pool_for_day(target_date=date(2026, 9, 5))

        self.assertEqual(len(result.picks), 1)
        self.assertEqual(result.picks[0].candidate.model_run_id, "a")
        self.assertEqual(result.policy_version, "pick_pool_policy_v1")

    def test_build_pool_for_day_with_custom_policy(self):
        rows = [
            {
                "fixture_id": 1,
                "datetime": "2026-09-05T18:00:00+00:00",
                "decision_cards": [
                    _decision_card(market="h2h", value_label="PLAY", odd=2.0, ev=0.10, run_id="a"),
                    _decision_card(market="dc", value_label="BORDERLINE", odd=1.3, ev=0.01, run_id="b"),
                ],
            },
        ]
        service = self._service_with_day_rows(rows)
        policy = PickPoolPolicy.with_overrides(include_borderline=True, min_odd=1.2)

        result = service.build_pool_for_day(target_date=date(2026, 9, 5), policy=policy)

        self.assertEqual(len(result.picks), 2)
        self.assertEqual(result.policy_version, policy.version)


if __name__ == "__main__":
    unittest.main()
