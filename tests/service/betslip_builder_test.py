import unittest

from src.oracle.betslip.betslip_builder import (
    AGGRESSIVE_PROFILE,
    BALANCED_PROFILE,
    BetslipDiversificationPolicy,
    DEFAULT_SLIP_PROFILES,
    SAFE_PROFILE,
    SlipProfile,
    generate_betslips,
)
from src.oracle.betslip.correlation_engine import CorrelationRuleSet
from src.oracle.betslip.pick_pool import CandidatePick


RELAXED_DIVERSIFICATION = BetslipDiversificationPolicy(
    version="test_relaxed_diversification",
    max_candidates_per_family=20,
    max_overlap_ratio=1.0,
)


def _pick(fixture_id=1, market="h2h", outcome="Home", odd=2.0, p_model=0.6, ev=0.2, decision="PLAY"):
    return CandidatePick(
        fixture_id=fixture_id,
        market=market,
        outcome=outcome,
        decision=decision,
        odd=odd,
        p_model=p_model,
        ev=ev,
    )


class TestSlipProfileValidation(unittest.TestCase):
    def test_min_legs_below_two_raises(self):
        with self.assertRaises(ValueError):
            SlipProfile(name="X", version="v1", min_legs=1, max_legs=2)

    def test_max_legs_below_min_legs_raises(self):
        with self.assertRaises(ValueError):
            SlipProfile(name="X", version="v1", min_legs=3, max_legs=2)

    def test_max_legs_above_four_raises(self):
        with self.assertRaises(ValueError):
            SlipProfile(name="X", version="v1", min_legs=2, max_legs=5)

    def test_negative_max_penalty_pairs_raises(self):
        with self.assertRaises(ValueError):
            SlipProfile(name="X", version="v1", min_legs=2, max_legs=4, max_penalty_pairs=-1)

    def test_default_profiles_distinct(self):
        # Acceptance criteria "Tre profili distinti".
        names = {p.name for p in DEFAULT_SLIP_PROFILES}
        versions = {p.version for p in DEFAULT_SLIP_PROFILES}
        self.assertEqual(names, {"SAFE", "BALANCED", "AGGRESSIVE"})
        self.assertEqual(len(versions), 3)
        self.assertLess(SAFE_PROFILE.max_penalty_pairs, BALANCED_PROFILE.max_penalty_pairs)
        self.assertLess(BALANCED_PROFILE.max_penalty_pairs, AGGRESSIVE_PROFILE.max_penalty_pairs)
        self.assertGreater(SAFE_PROFILE.min_leg_probability, AGGRESSIVE_PROFILE.min_leg_probability)


class TestGenerateBetslipsBasics(unittest.TestCase):
    def test_no_candidates_returns_empty_profiles(self):
        result = generate_betslips([])
        self.assertEqual(result.pool_considered, 0)
        for profile in DEFAULT_SLIP_PROFILES:
            self.assertEqual(result.profiles[profile.name], [])

    def test_missing_odd_or_p_model_excluded_from_pool(self):
        candidates = [
            _pick(fixture_id=1, odd=None),
            _pick(fixture_id=2, p_model=None),
            _pick(fixture_id=3, odd=2.0, p_model=0.6),
        ]
        result = generate_betslips(candidates)
        self.assertEqual(result.pool_considered, 1)

    def test_two_independent_picks_generate_safe_slip(self):
        candidates = [
            _pick(fixture_id=1, market="h2h", outcome="Home", odd=1.8, p_model=0.60),
            _pick(fixture_id=2, market="goal_no_goal", outcome="Yes", odd=1.9, p_model=0.58),
        ]
        result = generate_betslips(candidates)
        safe_slips = result.profiles["SAFE"]
        self.assertEqual(len(safe_slips), 1)
        slip = safe_slips[0]
        self.assertEqual(slip.n_legs, 2)
        self.assertEqual(slip.risk_label, "LOW")
        self.assertAlmostEqual(slip.combined_odd, 1.8 * 1.9)
        self.assertAlmostEqual(slip.naive_probability, 0.60 * 0.58)
        self.assertAlmostEqual(slip.adjusted_probability, 0.60 * 0.58)
        self.assertEqual(slip.penalty_pairs, 0)
        self.assertAlmostEqual(slip.combined_ev, slip.adjusted_probability * slip.combined_odd - 1.0)
        self.assertAlmostEqual(slip.risk_score, 1.0 - slip.adjusted_probability)

    def test_low_probability_picks_excluded_from_safe_but_present_in_aggressive(self):
        candidates = [
            _pick(fixture_id=1, market="h2h", outcome="Home", odd=3.5, p_model=0.34),
            _pick(fixture_id=2, market="h2h", outcome="Away", odd=3.2, p_model=0.32),
            _pick(fixture_id=3, market="goal_no_goal", outcome="Yes", odd=3.0, p_model=0.31),
        ]
        result = generate_betslips(candidates)
        self.assertEqual(result.profiles["SAFE"], [])  # p_model < 0.55 (soglia SAFE)
        self.assertGreaterEqual(len(result.profiles["AGGRESSIVE"]), 1)

    def test_high_odd_leg_excluded_from_safe(self):
        candidates = [
            _pick(fixture_id=1, market="h2h", outcome="Home", odd=6.0, p_model=0.60),
            _pick(fixture_id=2, market="h2h", outcome="Away", odd=1.5, p_model=0.65),
        ]
        result = generate_betslips(candidates)
        self.assertEqual(result.profiles["SAFE"], [])  # odd 6.0 > max_leg_odd SAFE (2.50)

    def test_borderline_and_no_bet_never_enter_a_slip(self):
        candidates = [
            _pick(fixture_id=1, decision="PLAY"),
            _pick(fixture_id=2, decision="BORDERLINE"),
            _pick(fixture_id=3, decision="NO BET"),
        ]
        result = generate_betslips(candidates)
        self.assertEqual(result.pool_considered, 1)
        self.assertTrue(all(not slips for slips in result.profiles.values()))

    def test_exploration_can_generate_borderline_and_no_bet_without_promoting_them(self):
        candidates = [
            _pick(fixture_id=1, market="h2h", decision="PLAY"),
            _pick(fixture_id=2, market="goal_no_goal", decision="BORDERLINE"),
            _pick(fixture_id=3, market="under_over_2_5", decision="NO BET"),
        ]
        borderline = generate_betslips(
            candidates,
            allowed_decisions=frozenset({"PLAY", "BORDERLINE"}),
        )
        no_bet = generate_betslips(
            candidates,
            allowed_decisions=frozenset({"PLAY", "BORDERLINE", "NO BET"}),
        )

        self.assertTrue(
            any(
                slip.situation == "BORDERLINE"
                for slips in borderline.profiles.values()
                for slip in slips
            )
        )
        self.assertTrue(
            any(
                slip.situation == "NO BET"
                for slips in no_bet.profiles.values()
                for slip in slips
            )
        )

    def test_safe_slip_prioritizes_distinct_market_families(self):
        candidates = [
            _pick(fixture_id=1, market="under_over_2_5", outcome="Under 2.5"),
            _pick(fixture_id=2, market="under_over_3_5", outcome="Under 3.5"),
            _pick(fixture_id=3, market="goal_no_goal", outcome="No"),
        ]
        result = generate_betslips(candidates)

        self.assertTrue(result.profiles["SAFE"])
        first = result.profiles["SAFE"][0]
        self.assertFalse(
            all(leg.market.startswith("under_over_") for leg in first.legs)
        )
        self.assertTrue(
            all(
                slip.diversification_policy_version
                == "betslip_diversification_v2_soft_fallback"
                for slip in result.profiles["SAFE"]
            )
        )

    def test_single_market_family_uses_soft_fallback_instead_of_zero_slips(self):
        candidates = [
            _pick(
                fixture_id=index,
                market="under_over_3_5",
                outcome="Under 3.5",
                odd=1.8,
                p_model=0.60,
            )
            for index in range(1, 15)
        ]

        result = generate_betslips(candidates)
        total = sum(len(slips) for slips in result.profiles.values())

        self.assertGreaterEqual(total, 10)
        self.assertTrue(
            any("limited_market_diversification" in item for item in result.warnings)
        )

    def test_combined_value_metrics_use_adjusted_probability(self):
        candidates = [
            _pick(fixture_id=1, market="h2h", odd=1.8, p_model=0.60),
            _pick(fixture_id=2, market="goal_no_goal", odd=1.5, p_model=0.70),
        ]
        slip = generate_betslips(candidates).profiles["SAFE"][0]
        self.assertAlmostEqual(slip.combined_odd, 2.70)
        self.assertAlmostEqual(slip.naive_probability, 0.42)
        self.assertAlmostEqual(slip.adjusted_probability, 0.42)
        self.assertAlmostEqual(slip.combined_model_void_odd, 1 / 0.42)
        self.assertAlmostEqual(slip.combined_edge_absolute, 2.70 - (1 / 0.42))
        self.assertAlmostEqual(slip.combined_expected_roi, 0.42 * 2.70 - 1)
        self.assertAlmostEqual(slip.combined_expected_roi_percent, (0.42 * 2.70 - 1) * 100)
        self.assertEqual(slip.situation, "PLAY")


class TestCorrelationLimitsPerProfile(unittest.TestCase):
    def test_exclude_pair_never_generated_in_any_profile(self):
        # "Over 2.5" e "Under 1.5" sono EXCLUDE (contraddizione logica): mai
        # proposte insieme, in nessun profilo (acceptance criteria "Limiti
        # correlazione" + vincolo assoluto documentato nel modulo).
        candidates = [
            _pick(fixture_id=1, market="under_over_2_5", outcome="Over 2.5", odd=1.5, p_model=0.60),
            _pick(fixture_id=1, market="under_over_1_5", outcome="Under 1.5", odd=1.4, p_model=0.55),
        ]
        result = generate_betslips(candidates)
        for profile_name, slips in result.profiles.items():
            for slip in slips:
                legs_key = {(leg.market, leg.outcome) for leg in slip.legs}
                self.assertFalse(
                    {("under_over_2_5", "Over 2.5"), ("under_over_1_5", "Under 1.5")}.issubset(legs_key),
                    f"Combinazione EXCLUDE trovata nel profilo {profile_name}",
                )

    def test_same_fixture_pair_excluded_from_all_profiles_by_default(self):
        # Nested totals same-match: "Over 2.5" + "Over 3.5" -> PENALTY.
        # Probabilita'/quote scelte per superare comunque la soglia leg-level
        # di SAFE, cosi' l'esclusione e' dovuta SOLO a max_penalty_pairs=0.
        candidates = [
            _pick(fixture_id=1, market="under_over_2_5", outcome="Over 2.5", odd=1.6, p_model=0.60),
            _pick(fixture_id=1, market="under_over_3_5", outcome="Over 3.5", odd=2.4, p_model=0.56),
        ]
        result = generate_betslips(candidates)

        def _has_pair(slips):
            return any(len(s.legs) == 2 and s.penalty_pairs >= 1 for s in slips)

        self.assertFalse(_has_pair(result.profiles["SAFE"]))
        self.assertFalse(_has_pair(result.profiles["BALANCED"]))
        self.assertFalse(_has_pair(result.profiles["AGGRESSIVE"]))

    def test_penalty_pair_adjusted_probability_uses_min_not_product(self):
        candidates = [
            _pick(fixture_id=1, market="under_over_2_5", outcome="Over 2.5", odd=1.6, p_model=0.60),
            _pick(fixture_id=1, market="under_over_3_5", outcome="Over 3.5", odd=2.2, p_model=0.45),
        ]
        result = generate_betslips(
            candidates,
            one_pick_per_fixture=False,
            diversification_policy=RELAXED_DIVERSIFICATION,
        )
        balanced = [s for s in result.profiles["BALANCED"] if s.n_legs == 2]
        self.assertEqual(len(balanced), 1)
        slip = balanced[0]
        self.assertAlmostEqual(slip.adjusted_probability, 0.45)  # min(0.60, 0.45)
        self.assertNotAlmostEqual(slip.adjusted_probability, slip.naive_probability)
        self.assertAlmostEqual(slip.combined_ev, 0.45 * (1.6 * 2.2) - 1.0)


class TestRankingDeterminism(unittest.TestCase):
    def test_slips_sorted_by_ev_desc(self):
        candidates = [
            _pick(fixture_id=1, market="h2h", outcome="Home", odd=1.5, p_model=0.60),
            _pick(fixture_id=2, market="goal_no_goal", outcome="Yes", odd=1.5, p_model=0.60),
            _pick(fixture_id=3, market="under_over_2_5", outcome="Over 2.5", odd=3.0, p_model=0.60),
        ]
        result = generate_betslips(candidates, max_slips_per_profile=10)
        aggressive = result.profiles["AGGRESSIVE"]
        two_leg_slips = [s for s in aggressive if s.n_legs == 2]
        evs = [s.combined_ev for s in two_leg_slips]
        self.assertEqual(evs, sorted(evs, reverse=True))

    def test_deterministic_across_runs(self):
        candidates = [
            _pick(fixture_id=1, market="h2h", outcome="Home", odd=1.8, p_model=0.55),
            _pick(fixture_id=2, market="h2h", outcome="Away", odd=2.1, p_model=0.50),
            _pick(fixture_id=3, market="goal_no_goal", outcome="Yes", odd=1.9, p_model=0.52),
        ]
        result_a = generate_betslips(list(candidates))
        result_b = generate_betslips(list(reversed(candidates)))
        ids_a = [s.slip_id for s in result_a.profiles["AGGRESSIVE"]]
        ids_b = [s.slip_id for s in result_b.profiles["AGGRESSIVE"]]
        self.assertEqual(ids_a, ids_b)

    def test_max_slips_per_profile_truncates(self):
        candidates = [
            _pick(fixture_id=i, market="h2h", outcome="Home", odd=1.5 + i * 0.1, p_model=0.55)
            for i in range(1, 7)
        ]
        result = generate_betslips(candidates, max_slips_per_profile=3)
        for slips in result.profiles.values():
            self.assertLessEqual(len(slips), 3)


class TestExplanationAndTraceability(unittest.TestCase):
    def test_explanation_mentions_legs_and_odd(self):
        candidates = [
            _pick(fixture_id=1, market="h2h", outcome="Home", odd=1.8, p_model=0.60),
            _pick(fixture_id=2, market="goal_no_goal", outcome="Yes", odd=1.9, p_model=0.58),
        ]
        result = generate_betslips(candidates)
        slip = result.profiles["SAFE"][0]
        self.assertIn("2 eventi", slip.explanation)
        self.assertIn("SAFE", slip.explanation)
        self.assertIn(f"{slip.combined_odd:.2f}", slip.explanation)

    def test_pool_truncation_is_reported_in_warnings(self):
        candidates = [
            _pick(fixture_id=i, market="h2h", outcome="Home", odd=1.8, p_model=0.60)
            for i in range(1, 20)
        ]
        result = generate_betslips(candidates, max_pool_size=5)
        self.assertTrue(any("pool_truncated" in w for w in result.warnings))

    def test_slip_id_stable_regardless_of_leg_order(self):
        legs_a = [
            _pick(fixture_id=1, market="h2h", outcome="Home", odd=1.8, p_model=0.60),
            _pick(fixture_id=2, market="goal_no_goal", outcome="Yes", odd=1.9, p_model=0.58),
        ]
        legs_b = list(reversed(legs_a))
        result_a = generate_betslips(legs_a)
        result_b = generate_betslips(legs_b)
        self.assertEqual(result_a.profiles["SAFE"][0].slip_id, result_b.profiles["SAFE"][0].slip_id)

    def test_correlation_ruleset_version_reported(self):
        candidates = [
            _pick(fixture_id=1, market="h2h", outcome="Home", odd=1.8, p_model=0.60),
            _pick(fixture_id=2, market="goal_no_goal", outcome="Yes", odd=1.9, p_model=0.58),
        ]
        result = generate_betslips(candidates)
        self.assertEqual(result.correlation_ruleset_version, "correlation_ruleset_v1")
        self.assertEqual(result.profiles["SAFE"][0].correlation_ruleset_version, "correlation_ruleset_v1")


class TestCustomRuleset(unittest.TestCase):
    def test_custom_ruleset_changes_generated_slips(self):
        candidates = [
            _pick(fixture_id=1, market="under_over_1_5", outcome="Over 1.5", odd=1.5, p_model=0.60),
            _pick(fixture_id=1, market="under_over_2_5", outcome="Under 2.5", odd=1.6, p_model=0.55),
        ]
        default_result = generate_betslips(
            candidates,
            one_pick_per_fixture=False,
            diversification_policy=RELAXED_DIVERSIFICATION,
        )
        tighter = CorrelationRuleSet(version="correlation_ruleset_test_tight", totals_narrow_band_max_gap=0.0)
        custom_result = generate_betslips(
            candidates,
            ruleset=tighter,
            one_pick_per_fixture=False,
            diversification_policy=RELAXED_DIVERSIFICATION,
        )

        default_two_leg = [s for s in default_result.profiles["BALANCED"] if s.n_legs == 2]
        custom_two_leg = [s for s in custom_result.profiles["BALANCED"] if s.n_legs == 2]
        # Default: banda stretta (scarto 1.0) -> PENALTY, ammesso in BALANCED.
        self.assertEqual(len(default_two_leg), 1)
        self.assertEqual(default_two_leg[0].penalty_pairs, 1)
        # Ruleset piu' tollerante (gap massimo 0.0): la stessa coppia diventa INDEPENDENT.
        self.assertEqual(len(custom_two_leg), 1)
        self.assertEqual(custom_two_leg[0].penalty_pairs, 0)


if __name__ == "__main__":
    unittest.main()



