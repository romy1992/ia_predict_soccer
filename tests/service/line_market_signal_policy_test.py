import unittest

from src.oracle.decision_engine.line_market_signal_policy import (
    LINE_MARKET_SIGNAL_POLICY_VERSION,
    LINE_MARKET_SIGNAL_THRESHOLDS,
    evaluate_line_market_signal,
)


class TestLineMarketSignalThresholds(unittest.TestCase):
    def test_covers_all_eight_lines(self):
        self.assertEqual(
            set(LINE_MARKET_SIGNAL_THRESHOLDS.keys()),
            {
                "corners_line_8_5",
                "corners_line_9_5",
                "corners_line_10_5",
                "corners_line_11_5",
                "cards_line_3_5",
                "cards_line_4_5",
                "cards_line_5_5",
                "cards_line_6_5",
            },
        )

    def test_thresholds_decrease_as_line_rises(self):
        # Linee piu' alte -> "Over" sempre piu' raro -> soglia ottimale di
        # Youden sempre piu' bassa (verificato sul training reale, non un
        # valore arbitrario).
        corners_thresholds = [
            LINE_MARKET_SIGNAL_THRESHOLDS[f"corners_line_{label}"].probability_threshold
            for label in ["8_5", "9_5", "10_5", "11_5"]
        ]
        self.assertEqual(corners_thresholds, sorted(corners_thresholds, reverse=True))

        cards_thresholds = [
            LINE_MARKET_SIGNAL_THRESHOLDS[f"cards_line_{label}"].probability_threshold
            for label in ["3_5", "4_5", "5_5", "6_5"]
        ]
        self.assertEqual(cards_thresholds, sorted(cards_thresholds, reverse=True))


class TestEvaluateLineMarketSignal(unittest.TestCase):
    def test_none_for_unsupported_market(self):
        self.assertIsNone(evaluate_line_market_signal(market="under_over_2_5", p_over=0.9))
        self.assertIsNone(evaluate_line_market_signal(market="corners", p_over=0.9))

    def test_none_when_probability_missing(self):
        self.assertIsNone(evaluate_line_market_signal(market="corners_line_8_5", p_over=None))

    def test_signal_true_when_probability_above_threshold(self):
        result = evaluate_line_market_signal(market="corners_line_8_5", p_over=0.70)
        self.assertIsNotNone(result)
        self.assertTrue(result["signal"])
        self.assertEqual(result["policy_version"], LINE_MARKET_SIGNAL_POLICY_VERSION)

    def test_signal_false_when_probability_below_threshold(self):
        result = evaluate_line_market_signal(market="corners_line_8_5", p_over=0.50)
        self.assertFalse(result["signal"])

    def test_signal_can_be_true_below_0_5_on_a_rare_line(self):
        # cards_line_6_5: soglia 0.1686, ben sotto 0.5 - a soglia 0.5 fissa
        # il pick non avrebbe MAI mostrato "Over" per questa linea
        # (verificato: 0 predizioni Over su 646 casi reali, vedi
        # IMPLEMENTATION_LOG.md), il segnale invece si attiva correttamente.
        result = evaluate_line_market_signal(market="cards_line_6_5", p_over=0.20)
        self.assertTrue(result["signal"])
        self.assertLess(result["threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
