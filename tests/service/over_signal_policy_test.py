import unittest

from src.oracle.decision_engine.over_signal_policy import (
    OVER_SIGNAL_POLICY_VERSION,
    OVER_SIGNAL_THRESHOLDS,
    evaluate_over_signal,
)


class TestOverSignalThresholds(unittest.TestCase):
    def test_covers_all_four_under_over_markets(self):
        self.assertEqual(
            set(OVER_SIGNAL_THRESHOLDS.keys()),
            {"under_over_1_5", "under_over_2_5", "under_over_3_5", "under_over_4_5"},
        )

    def test_thresholds_decrease_as_over_becomes_rarer(self):
        # under_over_4_5 ha la base rate Over piu' bassa (13.5%): la soglia
        # di precisione deve essere la piu' bassa tra le 4 (coerente con
        # find_betting_thresholds.py, non un valore arbitrario).
        t = {m: spec.probability_threshold for m, spec in OVER_SIGNAL_THRESHOLDS.items()}
        self.assertGreater(t["under_over_1_5"], t["under_over_2_5"])
        self.assertGreater(t["under_over_2_5"], t["under_over_3_5"])
        self.assertGreater(t["under_over_3_5"], t["under_over_4_5"])


class TestEvaluateOverSignal(unittest.TestCase):
    def test_none_for_unsupported_market(self):
        self.assertIsNone(evaluate_over_signal(market="goal_no_goal", p_over=0.9))
        self.assertIsNone(evaluate_over_signal(market="h2h", p_over=0.9))

    def test_none_when_probability_missing(self):
        self.assertIsNone(evaluate_over_signal(market="under_over_1_5", p_over=None))

    def test_signal_true_when_probability_above_threshold(self):
        result = evaluate_over_signal(market="under_over_1_5", p_over=0.80)
        self.assertIsNotNone(result)
        self.assertTrue(result["signal"])
        self.assertEqual(result["policy_version"], OVER_SIGNAL_POLICY_VERSION)

    def test_signal_false_when_probability_below_threshold(self):
        result = evaluate_over_signal(market="under_over_1_5", p_over=0.60)
        self.assertFalse(result["signal"])

    def test_signal_can_be_true_below_0_5_on_a_low_base_rate_market(self):
        # under_over_4_5: soglia 0.1776, ben sotto 0.5 - il segnale e'
        # indipendente dal pick mostrato (che a 0.5 direbbe ancora "Under").
        result = evaluate_over_signal(market="under_over_4_5", p_over=0.30)
        self.assertTrue(result["signal"])
        self.assertLess(result["threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
