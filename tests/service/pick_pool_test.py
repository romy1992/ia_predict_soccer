import unittest

from src.oracle.betslip.pick_pool import (
    DEFAULT_PICK_POOL_POLICY,
    CandidatePick,
    PickPoolPolicy,
    build_pick_pool,
)


def _candidate(
    fixture_id=1,
    market="h2h",
    outcome="Home",
    decision="PLAY",
    odd=2.0,
    ev=0.1,
    prob_edge=0.05,
    samples=5,
    model_run_id="run-1",
):
    return CandidatePick(
        fixture_id=fixture_id,
        market=market,
        outcome=outcome,
        decision=decision,
        odd=odd,
        ev=ev,
        prob_edge=prob_edge,
        samples=samples,
        model_run_id=model_run_id,
    )


class TestPickPoolPolicy(unittest.TestCase):
    def test_default_policy_only_play(self):
        self.assertEqual(DEFAULT_PICK_POOL_POLICY.allowed_decisions, frozenset({"PLAY"}))

    def test_min_odd_greater_than_max_odd_raises(self):
        with self.assertRaises(ValueError):
            PickPoolPolicy(min_odd=2.0, max_odd=1.5)

    def test_negative_min_samples_raises(self):
        with self.assertRaises(ValueError):
            PickPoolPolicy(min_samples=-1)

    def test_with_overrides_returns_default_instance_when_no_change(self):
        policy = PickPoolPolicy.with_overrides()
        self.assertIs(policy, DEFAULT_PICK_POOL_POLICY)

    def test_with_overrides_returns_custom_version_when_changed(self):
        policy = PickPoolPolicy.with_overrides(min_odd=1.5, include_borderline=True)
        self.assertNotEqual(policy.version, DEFAULT_PICK_POOL_POLICY.version)
        self.assertTrue(policy.version.startswith("pick_pool_policy_custom_"))
        self.assertEqual(policy.allowed_decisions, frozenset({"PLAY", "BORDERLINE"}))

    def test_with_overrides_deterministic_for_same_params(self):
        policy_a = PickPoolPolicy.with_overrides(min_odd=1.5, min_ev=0.02)
        policy_b = PickPoolPolicy.with_overrides(min_odd=1.5, min_ev=0.02)
        self.assertEqual(policy_a.version, policy_b.version)

    def test_with_overrides_different_params_different_version(self):
        policy_a = PickPoolPolicy.with_overrides(min_odd=1.5)
        policy_b = PickPoolPolicy.with_overrides(min_odd=1.8)
        self.assertNotEqual(policy_a.version, policy_b.version)


class TestBuildPickPool(unittest.TestCase):
    def test_only_play_included_by_default(self):
        candidates = [
            _candidate(fixture_id=1, decision="PLAY"),
            _candidate(fixture_id=2, decision="BORDERLINE"),
            _candidate(fixture_id=3, decision="NO BET"),
        ]
        result = build_pick_pool(candidates)

        self.assertEqual(len(result.picks), 1)
        self.assertEqual(result.picks[0].candidate.fixture_id, 1)
        self.assertEqual(len(result.excluded), 2)
        reasons = {e.candidate.fixture_id: e.exclusion_reason for e in result.excluded}
        self.assertTrue(reasons[2].startswith("decision_not_allowed"))
        self.assertTrue(reasons[3].startswith("decision_not_allowed"))

    def test_include_borderline_when_configured(self):
        candidates = [
            _candidate(fixture_id=1, decision="PLAY"),
            _candidate(fixture_id=2, decision="BORDERLINE"),
            _candidate(fixture_id=3, decision="NO BET"),
        ]
        policy = PickPoolPolicy.with_overrides(include_borderline=True)
        result = build_pick_pool(candidates, policy=policy)

        included_fixtures = {p.candidate.fixture_id for p in result.picks}
        self.assertEqual(included_fixtures, {1, 2})
        self.assertEqual(len(result.excluded), 1)

    def test_min_odd_and_max_odd_filters(self):
        candidates = [
            _candidate(fixture_id=1, odd=1.2),  # sotto min_odd
            _candidate(fixture_id=2, odd=1.8),  # ok
            _candidate(fixture_id=3, odd=5.0),  # sopra max_odd
        ]
        policy = PickPoolPolicy.with_overrides(min_odd=1.5, max_odd=3.0)
        result = build_pick_pool(candidates, policy=policy)

        self.assertEqual([p.candidate.fixture_id for p in result.picks], [2])
        reasons = {e.candidate.fixture_id: e.exclusion_reason for e in result.excluded}
        self.assertTrue(reasons[1].startswith("odd_below_min"))
        self.assertTrue(reasons[3].startswith("odd_above_max"))

    def test_min_ev_filter(self):
        candidates = [
            _candidate(fixture_id=1, ev=0.01),
            _candidate(fixture_id=2, ev=0.10),
        ]
        policy = PickPoolPolicy.with_overrides(min_ev=0.05)
        result = build_pick_pool(candidates, policy=policy)

        self.assertEqual([p.candidate.fixture_id for p in result.picks], [2])
        self.assertTrue(result.excluded[0].exclusion_reason.startswith("ev_below_min"))

    def test_missing_odd_excluded(self):
        candidates = [_candidate(fixture_id=1, odd=None)]
        result = build_pick_pool(candidates)

        self.assertEqual(result.picks, [])
        self.assertEqual(result.excluded[0].exclusion_reason, "odd_missing")

    def test_one_selection_per_fixture_market_keeps_best_ev(self):
        candidates = [
            _candidate(fixture_id=1, market="under_over_2_5", outcome="Over 2.5", ev=0.05, model_run_id="a"),
            _candidate(fixture_id=1, market="under_over_2_5", outcome="Under 2.5", ev=0.20, model_run_id="b"),
        ]
        result = build_pick_pool(candidates)

        self.assertEqual(len(result.picks), 1)
        self.assertEqual(result.picks[0].candidate.model_run_id, "b")
        self.assertEqual(len(result.excluded), 1)
        self.assertTrue(result.excluded[0].exclusion_reason.startswith("superseded_same_fixture_market"))
        self.assertEqual(result.excluded[0].candidate.model_run_id, "a")

    def test_one_selection_per_fixture_market_prefers_play_over_borderline(self):
        candidates = [
            _candidate(fixture_id=1, market="h2h", outcome="Home", decision="BORDERLINE", ev=0.30, model_run_id="a"),
            _candidate(fixture_id=1, market="h2h", outcome="Away", decision="PLAY", ev=0.05, model_run_id="b"),
        ]
        policy = PickPoolPolicy.with_overrides(include_borderline=True)
        result = build_pick_pool(candidates, policy=policy)

        self.assertEqual(len(result.picks), 1)
        self.assertEqual(result.picks[0].candidate.model_run_id, "b")  # PLAY vince nonostante EV piu' basso

    def test_deterministic_ordering_by_ev_desc(self):
        candidates = [
            _candidate(fixture_id=1, market="h2h", ev=0.05),
            _candidate(fixture_id=2, market="h2h", ev=0.20),
            _candidate(fixture_id=3, market="h2h", ev=0.10),
        ]
        result = build_pick_pool(candidates)
        self.assertEqual([p.candidate.fixture_id for p in result.picks], [2, 3, 1])

    def test_pool_id_stable_regardless_of_input_order(self):
        candidates_a = [_candidate(fixture_id=1, ev=0.1), _candidate(fixture_id=2, ev=0.2)]
        candidates_b = list(reversed(candidates_a))

        result_a = build_pick_pool(candidates_a)
        result_b = build_pick_pool(candidates_b)
        self.assertEqual(result_a.pool_id, result_b.pool_id)

    def test_pool_id_changes_when_picks_differ(self):
        result_a = build_pick_pool([_candidate(fixture_id=1)])
        result_b = build_pick_pool([_candidate(fixture_id=2)])
        self.assertNotEqual(result_a.pool_id, result_b.pool_id)

    def test_policy_version_is_reported_on_result_and_picks(self):
        policy = PickPoolPolicy.with_overrides(min_odd=1.5)
        result = build_pick_pool([_candidate(fixture_id=1, odd=2.0)], policy=policy)
        self.assertEqual(result.policy_version, policy.version)
        self.assertEqual(result.picks[0].pool_policy_version, policy.version)


if __name__ == "__main__":
    unittest.main()


