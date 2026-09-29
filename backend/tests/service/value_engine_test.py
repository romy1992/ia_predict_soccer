import unittest

from src.oracle.fair_odds.fair_odds_engine import FairOddsOutcome
from src.oracle.value_engine.value_engine import (
    BORDERLINE,
    DEFAULT_POLICY,
    NO_BET,
    PLAY,
    ValueDecision,
    ValueDecisionPolicy,
    compute_expected_value,
    compute_prob_edge,
    evaluate_value,
    evaluate_value_from_fair_odds_outcome,
)


class TestComputeProbEdge(unittest.TestCase):
    def test_numeric_value(self):
        # 0.65 - 0.55 = 0.10
        self.assertAlmostEqual(compute_prob_edge(0.65, 0.55), 0.10, places=9)

    def test_negative_edge_when_model_below_market(self):
        self.assertAlmostEqual(compute_prob_edge(0.40, 0.55), -0.15, places=9)

    def test_none_when_p_model_missing(self):
        self.assertIsNone(compute_prob_edge(None, 0.55))

    def test_none_when_p_market_fair_missing(self):
        self.assertIsNone(compute_prob_edge(0.65, None))


class TestComputeExpectedValue(unittest.TestCase):
    def test_numeric_value(self):
        # 0.65 * 2.0 - 1 = 0.30
        self.assertAlmostEqual(compute_expected_value(0.65, 2.0), 0.30, places=9)

    def test_negative_ev_below_fair_odd(self):
        # 0.40 * 2.0 - 1 = -0.20
        self.assertAlmostEqual(compute_expected_value(0.40, 2.0), -0.20, places=9)

    def test_none_when_odd_missing(self):
        self.assertIsNone(compute_expected_value(0.65, None))

    def test_none_when_p_model_missing(self):
        self.assertIsNone(compute_expected_value(None, 2.0))

    def test_none_when_odd_not_positive(self):
        self.assertIsNone(compute_expected_value(0.65, 0.0))
        self.assertIsNone(compute_expected_value(0.65, -1.5))

    def test_none_when_odd_not_numeric(self):
        self.assertIsNone(compute_expected_value(0.65, "not-a-number"))


class TestEdgeAndEvAreDistinct(unittest.TestCase):
    """Acceptance criteria esplicito: 'Edge ed EV distinti'."""

    def test_prob_edge_and_ev_are_different_metrics_for_same_inputs(self):
        # p_model=0.65, p_market_fair=0.55, odd=2.0 (leggermente sopra la fair odd 1/0.55=1.818)
        decision = evaluate_value(
            market="h2h", outcome="Home", p_model=0.65, p_market_fair=0.55, odd=2.0
        )
        self.assertAlmostEqual(decision.prob_edge, 0.10, places=9)
        self.assertAlmostEqual(decision.ev, 0.30, places=9)
        self.assertNotAlmostEqual(decision.prob_edge, decision.ev, places=6)

    def test_prob_edge_independent_of_odd(self):
        """prob_edge non deve cambiare se cambia SOLO la quota (a parita' di
        probabilita' modello/mercato): e' una metrica di probabilita', non
        di quota — a differenza dell'EV."""
        d1 = evaluate_value(market="h2h", outcome="Home", p_model=0.65, p_market_fair=0.55, odd=1.9)
        d2 = evaluate_value(market="h2h", outcome="Home", p_model=0.65, p_market_fair=0.55, odd=2.5)
        self.assertEqual(d1.prob_edge, d2.prob_edge)
        self.assertNotEqual(d1.ev, d2.ev)


class TestEvaluateValueDecisionThresholds(unittest.TestCase):
    def test_play_when_high_confidence_and_positive_ev(self):
        # p_model=0.70 >= 0.62, ev = 0.70*1.6-1 = 0.12 >= 0.03
        decision = evaluate_value(market="h2h", outcome="Home", p_model=0.70, p_market_fair=0.55, odd=1.6)
        self.assertEqual(decision.decision, PLAY)

    def test_borderline_when_medium_confidence(self):
        # p_model=0.58 (>=0.55, <0.62), ev = 0.58*1.85-1 = 0.073 >= 0.0
        decision = evaluate_value(market="h2h", outcome="Home", p_model=0.58, p_market_fair=0.50, odd=1.85)
        self.assertEqual(decision.decision, BORDERLINE)

    def test_no_bet_when_low_confidence(self):
        decision = evaluate_value(market="h2h", outcome="Home", p_model=0.50, p_market_fair=0.55, odd=1.8)
        self.assertEqual(decision.decision, NO_BET)

    def test_no_bet_when_odd_missing(self):
        decision = evaluate_value(market="h2h", outcome="Home", p_model=0.90, p_market_fair=0.55, odd=None)
        self.assertEqual(decision.decision, NO_BET)
        self.assertEqual(decision.reason, "Quota non disponibile")
        self.assertIsNone(decision.ev)
        # prob_edge resta calcolabile anche senza quota (metrica indipendente dalla quota).
        self.assertIsNotNone(decision.prob_edge)

    def test_no_bet_when_odd_not_positive(self):
        decision = evaluate_value(market="h2h", outcome="Home", p_model=0.90, p_market_fair=0.55, odd=0.0)
        self.assertEqual(decision.decision, NO_BET)

    def test_no_bet_when_p_model_missing(self):
        decision = evaluate_value(market="h2h", outcome="Home", p_model=None, p_market_fair=0.55, odd=2.0)
        self.assertEqual(decision.decision, NO_BET)
        self.assertEqual(decision.reason, "Probabilita' modello non disponibile")

    def test_policy_version_is_reported(self):
        decision = evaluate_value(market="h2h", outcome="Home", p_model=0.5, p_market_fair=0.5, odd=2.0)
        self.assertEqual(decision.policy_version, DEFAULT_POLICY.version)

    def test_custom_policy_changes_decision(self):
        strict_policy = ValueDecisionPolicy(
            version="strict_v1", play_min_probability=0.9, play_min_ev=0.5,
            borderline_min_probability=0.8, borderline_min_ev=0.3,
        )
        decision = evaluate_value(
            market="h2h", outcome="Home", p_model=0.70, p_market_fair=0.55, odd=1.6, policy=strict_policy
        )
        # Con DEFAULT_POLICY questo sarebbe PLAY (vedi test sopra); con una
        # policy piu' severa diventa NO BET: la policy e' davvero usata, non
        # ignorata (verifica che il "versionamento" non sia solo cosmetico).
        self.assertEqual(decision.decision, NO_BET)
        self.assertEqual(decision.policy_version, "strict_v1")


class TestEvaluateValueFromFairOddsOutcome(unittest.TestCase):
    def test_uses_same_outcome_fields_as_fair_odds_outcome(self):
        fair_odds_outcome = FairOddsOutcome(
            market="goal_no_goal", outcome="Yes", odd=1.9, p_market_raw=0.526,
            p_market_fair=0.50, fair_odd=2.0, p_model=0.60, bookmakers=3,
        )
        decision = evaluate_value_from_fair_odds_outcome(fair_odds_outcome)

        self.assertIsInstance(decision, ValueDecision)
        self.assertEqual(decision.market, "goal_no_goal")
        self.assertEqual(decision.outcome, "Yes")
        self.assertAlmostEqual(decision.prob_edge, 0.10, places=9)
        self.assertAlmostEqual(decision.ev, 0.60 * 1.9 - 1, places=9)

    def test_missing_market_data_falls_back_to_no_bet_without_crashing(self):
        fair_odds_outcome = FairOddsOutcome(
            market="corners", outcome="Over 9.5", odd=None, p_market_raw=None,
            p_market_fair=None, fair_odd=None, p_model=0.55,
        )
        decision = evaluate_value_from_fair_odds_outcome(fair_odds_outcome)
        self.assertEqual(decision.decision, NO_BET)
        self.assertIsNone(decision.ev)
        self.assertIsNone(decision.prob_edge)


if __name__ == "__main__":
    unittest.main()
