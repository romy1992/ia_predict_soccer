import unittest

from src.ml.registry.promotion_policy import (
    ComparisonResult,
    PromotionGateThresholds,
    PromotionPolicy,
    compare_candidate_to_production,
    derive_comparable_score,
    evaluate_metrics_gate,
    evaluate_promotion,
)


GOOD_METRICS = {"log_loss": 0.55, "brier": 0.18, "ece": 0.05, "auc": 0.68, "sample_size": 500}


class TestDeriveComparableScore(unittest.TestCase):
    def test_prefers_selection_score_when_present(self):
        score, method = derive_comparable_score({"selection_score": 0.71, "log_loss": 99.0})
        self.assertEqual(score, 0.71)
        self.assertEqual(method, "selection_score")

    def test_falls_back_to_partial_composite(self):
        score, method = derive_comparable_score({"log_loss": 0.6, "brier": 0.2})
        self.assertIsNotNone(score)
        self.assertEqual(method, "derived_partial_composite")

    def test_prefers_post_calibration_metrics_over_pre(self):
        score_post, _ = derive_comparable_score({"post_log_loss": 0.5, "post_brier": 0.15})
        score_pre, _ = derive_comparable_score({"post_log_loss": 0.5, "post_brier": 0.15, "pre_log_loss": 5.0, "pre_brier": 5.0})
        # pre_* non deve mai essere usato quando post_* e' disponibile
        self.assertEqual(score_post, score_pre)

    def test_unavailable_when_no_usable_metric(self):
        score, method = derive_comparable_score({"unrelated": 1})
        self.assertIsNone(score)
        self.assertEqual(method, "unavailable")

    def test_unavailable_when_metrics_none(self):
        score, method = derive_comparable_score(None)
        self.assertIsNone(score)
        self.assertEqual(method, "unavailable")


class TestEvaluateMetricsGate(unittest.TestCase):
    def test_passes_with_good_metrics(self):
        result = evaluate_metrics_gate(GOOD_METRICS, PromotionGateThresholds())
        self.assertTrue(result.passed)
        self.assertEqual(result.blocking_reasons, [])

    def test_fails_on_log_loss_above_threshold(self):
        metrics = {**GOOD_METRICS, "log_loss": 5.0}
        result = evaluate_metrics_gate(metrics, PromotionGateThresholds())
        self.assertFalse(result.passed)
        self.assertTrue(any("log_loss" in reason for reason in result.blocking_reasons))

    def test_fails_on_brier_above_threshold(self):
        metrics = {**GOOD_METRICS, "brier": 0.9}
        result = evaluate_metrics_gate(metrics, PromotionGateThresholds())
        self.assertFalse(result.passed)
        self.assertTrue(any("brier" in reason for reason in result.blocking_reasons))

    def test_fails_on_ece_above_threshold(self):
        metrics = {**GOOD_METRICS, "ece": 0.9}
        result = evaluate_metrics_gate(metrics, PromotionGateThresholds())
        self.assertFalse(result.passed)
        self.assertTrue(any("ece" in reason for reason in result.blocking_reasons))

    def test_fails_on_auc_below_threshold(self):
        metrics = {**GOOD_METRICS, "auc": 0.2}
        result = evaluate_metrics_gate(metrics, PromotionGateThresholds())
        self.assertFalse(result.passed)
        self.assertTrue(any("auc" in reason for reason in result.blocking_reasons))

    def test_fails_on_sample_size_below_threshold(self):
        metrics = {**GOOD_METRICS, "sample_size": 5}
        result = evaluate_metrics_gate(metrics, PromotionGateThresholds())
        self.assertFalse(result.passed)
        self.assertTrue(any("sample_size" in reason for reason in result.blocking_reasons))

    def test_sample_size_fallback_to_rows_key(self):
        metrics = {"log_loss": 0.5, "brier": 0.1, "rows": 500}
        result = evaluate_metrics_gate(metrics, PromotionGateThresholds(max_ece=None, min_auc=None))
        self.assertTrue(result.passed)

    def test_missing_single_metric_does_not_block_when_others_pass(self):
        # ece assente: check "metric_not_available", non bloccante da solo.
        metrics = {"log_loss": 0.5, "brier": 0.1, "auc": 0.7, "sample_size": 100}
        result = evaluate_metrics_gate(metrics, PromotionGateThresholds())
        self.assertTrue(result.passed)
        ece_check = next(c for c in result.checks if c.name == "ece")
        self.assertTrue(ece_check.passed)
        self.assertEqual(ece_check.detail, "metric_not_available")

    def test_fails_when_no_metrics_at_all_and_required(self):
        result = evaluate_metrics_gate({}, PromotionGateThresholds())
        self.assertFalse(result.passed)
        self.assertIn("nessuna metrica valutabile per il gate (metrics mancanti o incomplete)", result.blocking_reasons)

    def test_passes_when_no_metrics_but_not_required(self):
        gate = PromotionGateThresholds(
            max_log_loss=None, max_brier=None, max_ece=None, min_auc=None, min_sample_size=0,
            require_at_least_one_metric=False,
        )
        result = evaluate_metrics_gate({}, gate)
        self.assertTrue(result.passed)

    def test_disabled_threshold_skips_check_entirely(self):
        gate = PromotionGateThresholds(max_log_loss=None)
        metrics = {"log_loss": 999.0, "brier": 0.1, "auc": 0.7, "sample_size": 100}
        result = evaluate_metrics_gate(metrics, gate)
        self.assertTrue(result.passed)
        self.assertFalse(any(c.name == "log_loss" for c in result.checks))


class TestCompareCandidateToProduction(unittest.TestCase):
    def test_no_production_baseline_allowed_by_default(self):
        result = compare_candidate_to_production(GOOD_METRICS, None, PromotionPolicy())
        self.assertTrue(result.passed)
        self.assertEqual(result.method, "no_production_baseline")

    def test_no_production_baseline_denied_when_policy_requires_one(self):
        policy = PromotionPolicy(allow_promotion_without_production_baseline=False)
        result = compare_candidate_to_production(GOOD_METRICS, None, policy)
        self.assertFalse(result.passed)

    def test_candidate_better_than_production_passes(self):
        candidate = {"selection_score": 0.8}
        production = {"selection_score": 0.6}
        result = compare_candidate_to_production(candidate, production, PromotionPolicy())
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.delta, 0.2)
        self.assertEqual(result.method, "selection_score")

    def test_candidate_worse_than_production_fails(self):
        candidate = {"selection_score": 0.5}
        production = {"selection_score": 0.6}
        result = compare_candidate_to_production(candidate, production, PromotionPolicy())
        self.assertFalse(result.passed)

    def test_min_improvement_over_production_enforced(self):
        candidate = {"selection_score": 0.61}
        production = {"selection_score": 0.6}
        policy = PromotionPolicy(min_improvement_over_production=0.05)
        result = compare_candidate_to_production(candidate, production, policy)
        self.assertFalse(result.passed)  # 0.01 di delta < 0.05 richiesto

    def test_unavailable_when_scores_cannot_be_derived(self):
        result = compare_candidate_to_production({"unrelated": 1}, {"unrelated": 2}, PromotionPolicy())
        self.assertFalse(result.passed)
        self.assertEqual(result.method, "unavailable")


class TestEvaluatePromotion(unittest.TestCase):
    def test_champion_stage_skips_comparison(self):
        evaluation = evaluate_promotion(GOOD_METRICS, {"selection_score": 999}, to_stage="champion")
        self.assertIsNone(evaluation.comparison)
        self.assertTrue(evaluation.allowed)

    def test_production_stage_blocks_on_failed_comparison_even_if_gate_passes(self):
        candidate = {**GOOD_METRICS, "selection_score": 0.3}
        production = {"selection_score": 0.9}
        evaluation = evaluate_promotion(candidate, production, to_stage="production")
        self.assertTrue(evaluation.gate.passed)
        self.assertFalse(evaluation.comparison.passed)
        self.assertFalse(evaluation.allowed)

    def test_allowed_true_when_gate_and_comparison_both_pass(self):
        candidate = {**GOOD_METRICS, "selection_score": 0.9}
        production = {"selection_score": 0.5}
        evaluation = evaluate_promotion(candidate, production, to_stage="production")
        self.assertTrue(evaluation.allowed)

    def test_allowed_false_when_gate_fails_regardless_of_comparison(self):
        candidate = {"selection_score": 0.99, "log_loss": 50.0}
        evaluation = evaluate_promotion(candidate, None, to_stage="production")
        self.assertFalse(evaluation.gate.passed)
        self.assertFalse(evaluation.allowed)

    def test_to_dict_is_json_friendly(self):
        evaluation = evaluate_promotion(GOOD_METRICS, None, to_stage="production")
        payload = evaluation.to_dict()
        self.assertIn("allowed", payload)
        self.assertIn("gate", payload)
        self.assertIsInstance(payload["gate"]["checks"], list)


if __name__ == "__main__":
    unittest.main()

