import unittest

from src.oracle.betslip.correlation_engine import (
    CORRELATION_RULE_CATALOG,
    DEFAULT_CORRELATION_RULESET,
    EXCLUDE,
    INDEPENDENT,
    PENALTY,
    CorrelationRuleSet,
    SemanticOutcome,
    classify_outcome,
    classify_pick_pair,
    candidates_from_pool_picks,
    evaluate_combination,
    evaluate_same_match_correlations,
    naive_independent_probability,
)
from src.oracle.betslip.pick_pool import CandidatePick, PoolPick


def _pick(fixture_id=1, market="h2h", outcome="Home", p_model=0.6):
    return CandidatePick(fixture_id=fixture_id, market=market, outcome=outcome, decision="PLAY", p_model=p_model)


class TestClassifyOutcome(unittest.TestCase):
    def test_btts_yes_recognized(self):
        self.assertEqual(classify_outcome("goal_no_goal", "Yes"), SemanticOutcome(family="BTTS", tag="YES"))
        self.assertEqual(classify_outcome("goal_no_goal", "Goal"), SemanticOutcome(family="BTTS", tag="YES"))
        self.assertEqual(classify_outcome("btts", "yes"), SemanticOutcome(family="BTTS", tag="YES"))

    def test_btts_no_recognized(self):
        self.assertEqual(classify_outcome("goal_no_goal", "No"), SemanticOutcome(family="BTTS", tag="NO"))
        self.assertEqual(classify_outcome("goal_no_goal", "No Goal"), SemanticOutcome(family="BTTS", tag="NO"))

    def test_match_result_recognized(self):
        self.assertEqual(classify_outcome("h2h", "Home"), SemanticOutcome(family="MATCH_RESULT", tag="HOME"))
        self.assertEqual(classify_outcome("h2h", "Draw"), SemanticOutcome(family="MATCH_RESULT", tag="DRAW"))
        self.assertEqual(classify_outcome("h2h", "Away"), SemanticOutcome(family="MATCH_RESULT", tag="AWAY"))

    def test_match_result_team_name_not_recognized(self):
        # Limite noto e documentato: un nome squadra non e' riconosciuto (fail-open).
        self.assertIsNone(classify_outcome("h2h", "Juventus"))

    def test_double_chance_recognized(self):
        self.assertEqual(classify_outcome("dc", "Home/Draw"), SemanticOutcome(family="DOUBLE_CHANCE", tag="HOME_DRAW"))
        self.assertEqual(classify_outcome("dc", "Draw/Away"), SemanticOutcome(family="DOUBLE_CHANCE", tag="DRAW_AWAY"))
        self.assertEqual(classify_outcome("dc", "1X"), SemanticOutcome(family="DOUBLE_CHANCE", tag="HOME_DRAW"))

    def test_totals_recognized(self):
        self.assertEqual(
            classify_outcome("under_over_2_5", "Over 2.5"),
            SemanticOutcome(family="TOTALS", tag="OVER", threshold=2.5),
        )
        self.assertEqual(
            classify_outcome("under_over_1_5", "Under 1.5"),
            SemanticOutcome(family="TOTALS", tag="UNDER", threshold=1.5),
        )

    def test_corners_and_cards_not_treated_as_totals(self):
        # "corners"/"cards" usano testo Over/Under ma NON riguardano i gol:
        # non devono mai essere classificati come TOTALS.
        self.assertIsNone(classify_outcome("corners", "Over 9.5"))
        self.assertIsNone(classify_outcome("cards", "Under 4.5"))

    def test_unrecognized_market_returns_none(self):
        self.assertIsNone(classify_outcome("something_else", "whatever"))


class TestSameMarketRules(unittest.TestCase):
    def test_different_fixture_always_independent(self):
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="h2h", outcome="Home"),
            _pick(fixture_id=2, market="under_over_1_5", outcome="Under 1.5"),
        )
        self.assertEqual(finding.severity, INDEPENDENT)
        self.assertEqual(finding.rule, "different_fixture")

    def test_duplicate_pick_excluded(self):
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="h2h", outcome="Home"),
            _pick(fixture_id=1, market="h2h", outcome="home"),
        )
        self.assertEqual(finding.severity, EXCLUDE)
        self.assertEqual(finding.rule, "duplicate_pick")

    def test_same_market_different_outcome_mutually_exclusive(self):
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="h2h", outcome="Home"),
            _pick(fixture_id=1, market="h2h", outcome="Away"),
        )
        self.assertEqual(finding.severity, EXCLUDE)
        self.assertEqual(finding.rule, "same_market_mutually_exclusive")


class TestTotalsCorrelation(unittest.TestCase):
    def test_nested_same_direction_over_penalized(self):
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="under_over_2_5", outcome="Over 2.5"),
            _pick(fixture_id=1, market="under_over_3_5", outcome="Over 3.5"),
        )
        self.assertEqual(finding.severity, PENALTY)
        self.assertEqual(finding.rule, "totals_nested_same_direction")

    def test_nested_same_direction_under_penalized(self):
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="under_over_1_5", outcome="Under 1.5"),
            _pick(fixture_id=1, market="under_over_2_5", outcome="Under 2.5"),
        )
        self.assertEqual(finding.severity, PENALTY)
        self.assertEqual(finding.rule, "totals_nested_same_direction")

    def test_contradiction_excluded(self):
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="under_over_2_5", outcome="Over 2.5"),
            _pick(fixture_id=1, market="under_over_1_5", outcome="Under 1.5"),
        )
        self.assertEqual(finding.severity, EXCLUDE)
        self.assertEqual(finding.rule, "totals_contradiction")

    def test_narrow_band_penalized(self):
        # Over 1.5 & Under 2.5 -> scarto 1.0 <= default 1.0 => PENALTY
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="under_over_1_5", outcome="Over 1.5"),
            _pick(fixture_id=1, market="under_over_2_5", outcome="Under 2.5"),
        )
        self.assertEqual(finding.severity, PENALTY)
        self.assertEqual(finding.rule, "totals_narrow_band")

    def test_wide_band_independent(self):
        # Over 1.5 & Under 4.5 -> scarto 3.0 > 1.0 => INDEPENDENT
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="under_over_1_5", outcome="Over 1.5"),
            _pick(fixture_id=1, market="under_over_4_5", outcome="Under 4.5"),
        )
        self.assertEqual(finding.severity, INDEPENDENT)
        self.assertEqual(finding.rule, "totals_wide_band")


class TestBttsCorrelation(unittest.TestCase):
    def test_btts_yes_under_1_5_excluded(self):
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="goal_no_goal", outcome="Yes"),
            _pick(fixture_id=1, market="under_over_1_5", outcome="Under 1.5"),
        )
        self.assertEqual(finding.severity, EXCLUDE)
        self.assertEqual(finding.rule, "btts_totals_contradiction")

    def test_btts_yes_over_1_5_implied(self):
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="goal_no_goal", outcome="Yes"),
            _pick(fixture_id=1, market="under_over_1_5", outcome="Over 1.5"),
        )
        self.assertEqual(finding.severity, PENALTY)
        self.assertEqual(finding.rule, "btts_totals_implied")

    def test_btts_yes_over_2_5_narrow_band(self):
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="goal_no_goal", outcome="Yes"),
            _pick(fixture_id=1, market="under_over_2_5", outcome="Over 2.5"),
        )
        self.assertEqual(finding.severity, PENALTY)
        self.assertEqual(finding.rule, "btts_totals_narrow_band")

    def test_btts_yes_over_3_5_wide_band_independent(self):
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="goal_no_goal", outcome="Yes"),
            _pick(fixture_id=1, market="under_over_3_5", outcome="Over 3.5"),
        )
        self.assertEqual(finding.severity, INDEPENDENT)

    def test_btts_no_never_flagged_against_totals(self):
        finding_over = classify_pick_pair(
            _pick(fixture_id=1, market="goal_no_goal", outcome="No"),
            _pick(fixture_id=1, market="under_over_4_5", outcome="Over 4.5"),
        )
        finding_under = classify_pick_pair(
            _pick(fixture_id=1, market="goal_no_goal", outcome="No"),
            _pick(fixture_id=1, market="under_over_1_5", outcome="Under 1.5"),
        )
        self.assertEqual(finding_over.severity, INDEPENDENT)
        self.assertEqual(finding_under.severity, INDEPENDENT)


class TestMatchResultDoubleChanceCorrelation(unittest.TestCase):
    def test_home_implies_home_draw(self):
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="h2h", outcome="Home"),
            _pick(fixture_id=1, market="dc", outcome="Home/Draw"),
        )
        self.assertEqual(finding.severity, PENALTY)
        self.assertEqual(finding.rule, "match_result_double_chance_implied")

    def test_away_excludes_home_draw(self):
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="h2h", outcome="Away"),
            _pick(fixture_id=1, market="dc", outcome="Home/Draw"),
        )
        self.assertEqual(finding.severity, EXCLUDE)
        self.assertEqual(finding.rule, "match_result_double_chance_exclusive")

    def test_draw_implies_draw_away(self):
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="h2h", outcome="Draw"),
            _pick(fixture_id=1, market="dc", outcome="Draw/Away"),
        )
        self.assertEqual(finding.severity, PENALTY)

    def test_draw_excludes_home_away(self):
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="h2h", outcome="Draw"),
            _pick(fixture_id=1, market="dc", outcome="Home/Away"),
        )
        self.assertEqual(finding.severity, EXCLUDE)


class TestUnrelatedFamiliesIndependent(unittest.TestCase):
    def test_match_result_vs_totals_independent(self):
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="h2h", outcome="Home"),
            _pick(fixture_id=1, market="under_over_2_5", outcome="Over 2.5"),
        )
        self.assertEqual(finding.severity, INDEPENDENT)
        self.assertEqual(finding.rule, "no_rule")

    def test_match_result_vs_btts_independent(self):
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="h2h", outcome="Draw"),
            _pick(fixture_id=1, market="goal_no_goal", outcome="Yes"),
        )
        self.assertEqual(finding.severity, INDEPENDENT)

    def test_double_chance_vs_totals_independent(self):
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="dc", outcome="Home/Draw"),
            _pick(fixture_id=1, market="under_over_2_5", outcome="Over 2.5"),
        )
        self.assertEqual(finding.severity, INDEPENDENT)

    def test_unrecognized_outcome_independent(self):
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="h2h", outcome="Juventus"),
            _pick(fixture_id=1, market="goal_no_goal", outcome="Yes"),
        )
        self.assertEqual(finding.severity, INDEPENDENT)
        self.assertEqual(finding.rule, "no_rule")


class TestEvaluateSameMatchCorrelations(unittest.TestCase):
    def test_cross_fixture_pairs_never_flagged(self):
        picks = [
            _pick(fixture_id=1, market="under_over_2_5", outcome="Over 2.5"),
            _pick(fixture_id=2, market="under_over_1_5", outcome="Under 1.5"),
        ]
        report = evaluate_same_match_correlations(picks)
        self.assertEqual(report.findings, [])
        self.assertTrue(report.is_valid)

    def test_same_fixture_contradiction_reported_as_excluded(self):
        picks = [
            _pick(fixture_id=1, market="under_over_2_5", outcome="Over 2.5"),
            _pick(fixture_id=1, market="under_over_1_5", outcome="Under 1.5"),
        ]
        report = evaluate_same_match_correlations(picks)
        self.assertEqual(len(report.findings), 1)
        self.assertFalse(report.is_valid)
        self.assertEqual(len(report.excluded), 1)

    def test_penalized_pairs_do_not_invalidate_combination(self):
        picks = [
            _pick(fixture_id=1, market="h2h", outcome="Home"),
            _pick(fixture_id=1, market="dc", outcome="Home/Draw"),
        ]
        report = evaluate_same_match_correlations(picks)
        self.assertTrue(report.is_valid)
        self.assertEqual(len(report.penalized), 1)

    def test_ruleset_version_reported(self):
        report = evaluate_same_match_correlations([_pick(fixture_id=1)])
        self.assertEqual(report.ruleset_version, DEFAULT_CORRELATION_RULESET.version)


class TestEvaluateCombination(unittest.TestCase):
    def test_independent_picks_adjusted_equals_naive(self):
        picks = [
            _pick(fixture_id=1, market="h2h", outcome="Home", p_model=0.5),
            _pick(fixture_id=2, market="h2h", outcome="Away", p_model=0.4),
        ]
        result = evaluate_combination(picks)
        self.assertTrue(result.is_valid)
        self.assertAlmostEqual(result.naive_probability, 0.2)
        self.assertAlmostEqual(result.adjusted_probability, 0.2)

    def test_correlated_pair_not_treated_as_independent(self):
        # Over 3.5 implica Over 2.5: prodotto ingenuo sottostima la verita'.
        picks = [
            _pick(fixture_id=1, market="under_over_2_5", outcome="Over 2.5", p_model=0.5),
            _pick(fixture_id=1, market="under_over_3_5", outcome="Over 3.5", p_model=0.3),
        ]
        result = evaluate_combination(picks)
        self.assertTrue(result.is_valid)
        self.assertAlmostEqual(result.naive_probability, 0.15)
        self.assertAlmostEqual(result.adjusted_probability, 0.3)
        self.assertNotAlmostEqual(result.adjusted_probability, result.naive_probability)

    def test_contradiction_makes_combination_invalid_with_zero_probability(self):
        picks = [
            _pick(fixture_id=1, market="under_over_2_5", outcome="Over 2.5", p_model=0.5),
            _pick(fixture_id=1, market="under_over_1_5", outcome="Under 1.5", p_model=0.3),
        ]
        result = evaluate_combination(picks)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.adjusted_probability, 0.0)
        self.assertAlmostEqual(result.naive_probability, 0.15)

    def test_missing_p_model_makes_probabilities_none_but_still_classifies(self):
        picks = [
            _pick(fixture_id=1, market="h2h", outcome="Home", p_model=None),
            _pick(fixture_id=1, market="dc", outcome="Home/Draw", p_model=0.7),
        ]
        result = evaluate_combination(picks)
        self.assertIsNone(result.naive_probability)
        self.assertIsNone(result.adjusted_probability)
        self.assertEqual(len(result.findings), 1)

    def test_three_picks_two_correlated_one_independent(self):
        picks = [
            _pick(fixture_id=1, market="under_over_2_5", outcome="Over 2.5", p_model=0.5),
            _pick(fixture_id=1, market="under_over_3_5", outcome="Over 3.5", p_model=0.3),
            _pick(fixture_id=2, market="h2h", outcome="Home", p_model=0.6),
        ]
        result = evaluate_combination(picks)
        self.assertTrue(result.is_valid)
        self.assertAlmostEqual(result.naive_probability, 0.5 * 0.3 * 0.6)
        self.assertAlmostEqual(result.adjusted_probability, 0.3 * 0.6)


class TestCorrelationRuleSetVersioning(unittest.TestCase):
    def test_default_version(self):
        self.assertEqual(DEFAULT_CORRELATION_RULESET.version, "correlation_ruleset_v1")

    def test_custom_narrow_band_changes_classification(self):
        tighter = CorrelationRuleSet(version="correlation_ruleset_test_tight", totals_narrow_band_max_gap=0.5)
        finding = classify_pick_pair(
            _pick(fixture_id=1, market="under_over_1_5", outcome="Over 1.5"),
            _pick(fixture_id=1, market="under_over_2_5", outcome="Under 2.5"),
            ruleset=tighter,
        )
        # Con banda 0.5 (invece del default 1.0) lo scarto di 1.0 non e' piu' "stretto".
        self.assertEqual(finding.severity, INDEPENDENT)

    def test_rule_catalog_is_documented_and_non_empty(self):
        self.assertGreater(len(CORRELATION_RULE_CATALOG), 0)
        rule_ids = {r.rule_id for r in CORRELATION_RULE_CATALOG}
        self.assertIn("totals_contradiction", rule_ids)
        self.assertIn("btts_totals_implied", rule_ids)
        self.assertIn("match_result_double_chance_exclusive", rule_ids)


class TestCandidatesFromPoolPicks(unittest.TestCase):
    def test_extracts_candidate_from_pool_pick(self):
        candidate = _pick(fixture_id=1, market="h2h", outcome="Home")
        pool_pick = PoolPick(candidate=candidate, inclusion_reason="passed_policy:x", pool_policy_version="x")
        result = candidates_from_pool_picks([pool_pick])
        self.assertEqual(result, [candidate])


class TestNaiveIndependentProbability(unittest.TestCase):
    def test_product_of_p_model(self):
        picks = [_pick(p_model=0.5), _pick(p_model=0.4)]
        self.assertAlmostEqual(naive_independent_probability(picks), 0.2)

    def test_none_when_any_p_model_missing(self):
        picks = [_pick(p_model=0.5), _pick(p_model=None)]
        self.assertIsNone(naive_independent_probability(picks))

    def test_none_for_empty_list(self):
        self.assertIsNone(naive_independent_probability([]))


if __name__ == "__main__":
    unittest.main()

