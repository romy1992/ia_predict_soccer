import unittest

from src.oracle.fair_odds.fair_odds_engine import FairOddsOutcome
from src.oracle.decision_engine.decision_policy import (
    DEFAULT_DECISION_POLICY,
    DEFAULT_THRESHOLDS,
    BORDERLINE,
    NO_BET,
    PLAY,
    Decision,
    DecisionPolicy,
    DecisionThresholds,
    evaluate_decision,
    evaluate_decision_from_fair_odds_outcome,
)


class TestDecisionThresholdsValidation(unittest.TestCase):
    def test_default_thresholds_are_valid(self):
        self.assertEqual(DEFAULT_THRESHOLDS.min_samples, 0)
        self.assertIsNone(DEFAULT_THRESHOLDS.min_odd)
        self.assertIsNone(DEFAULT_THRESHOLDS.max_odd)

    def test_raises_when_min_odd_greater_than_max_odd(self):
        with self.assertRaises(ValueError):
            DecisionThresholds(min_odd=2.0, max_odd=1.5)

    def test_raises_when_min_samples_negative(self):
        with self.assertRaises(ValueError):
            DecisionThresholds(min_samples=-1)


class TestDefaultPolicyMatchesBet02Behaviour(unittest.TestCase):
    """Acceptance criteria implicito (compatibilita'): con i filtri nuovi non
    vincolanti, la decisione deve coincidere con quella gia' validata in
    BET-02 (value_engine_test.py) per gli stessi input."""

    def test_play_when_high_confidence_and_positive_ev(self):
        # Stesso caso di value_engine_test: p_model=0.70, odd=1.6 -> PLAY
        decision = evaluate_decision(market="h2h", outcome="Home", p_model=0.70, p_market_fair=0.55, odd=1.6, samples=3)
        self.assertEqual(decision.decision, PLAY)

    def test_borderline_when_medium_confidence(self):
        decision = evaluate_decision(market="h2h", outcome="Home", p_model=0.58, p_market_fair=0.50, odd=1.85, samples=3)
        self.assertEqual(decision.decision, BORDERLINE)

    def test_no_bet_when_low_confidence(self):
        decision = evaluate_decision(market="h2h", outcome="Home", p_model=0.50, p_market_fair=0.55, odd=1.8, samples=3)
        self.assertEqual(decision.decision, NO_BET)

    def test_no_bet_when_odd_missing(self):
        decision = evaluate_decision(market="h2h", outcome="Home", p_model=0.90, p_market_fair=0.55, odd=None, samples=3)
        self.assertEqual(decision.decision, NO_BET)
        self.assertEqual(decision.reason, "Quota non disponibile")

    def test_no_bet_when_p_model_missing(self):
        decision = evaluate_decision(market="h2h", outcome="Home", p_model=None, p_market_fair=0.55, odd=2.0, samples=3)
        self.assertEqual(decision.decision, NO_BET)
        self.assertEqual(decision.reason, "Probabilita' modello non disponibile")

    def test_policy_version_is_reported(self):
        decision = evaluate_decision(market="h2h", outcome="Home", p_model=0.5, p_market_fair=0.5, odd=2.0, samples=0)
        self.assertEqual(decision.policy_version, DEFAULT_DECISION_POLICY.version)


class TestMinSamplesFilter(unittest.TestCase):
    """Acceptance criteria "Min samples"."""

    def test_no_bet_when_samples_below_threshold(self):
        thresholds = DecisionThresholds(min_samples=3)
        policy = DecisionPolicy(version="test_min_samples", per_market={"h2h": thresholds})
        # Altrimenti sarebbe PLAY (stesso input del test sopra)
        decision = evaluate_decision(
            market="h2h", outcome="Home", p_model=0.70, p_market_fair=0.55, odd=1.6, samples=1, policy=policy
        )
        self.assertEqual(decision.decision, NO_BET)
        self.assertIn("Campione insufficiente", decision.reason)

    def test_play_when_samples_meet_threshold(self):
        thresholds = DecisionThresholds(min_samples=3)
        policy = DecisionPolicy(version="test_min_samples", per_market={"h2h": thresholds})
        decision = evaluate_decision(
            market="h2h", outcome="Home", p_model=0.70, p_market_fair=0.55, odd=1.6, samples=3, policy=policy
        )
        self.assertEqual(decision.decision, PLAY)

    def test_default_min_samples_zero_never_blocks(self):
        # samples=0 col default (min_samples=0) non deve mai bloccare la decisione.
        decision = evaluate_decision(market="h2h", outcome="Home", p_model=0.70, p_market_fair=0.55, odd=1.6, samples=0)
        self.assertEqual(decision.decision, PLAY)


class TestMinMaxOddFilter(unittest.TestCase):
    """Acceptance criteria "Min/max odd opzionali"."""

    def test_no_bet_when_odd_below_min(self):
        thresholds = DecisionThresholds(min_odd=1.5)
        policy = DecisionPolicy(version="test_min_odd", per_market={"h2h": thresholds})
        decision = evaluate_decision(
            market="h2h", outcome="Home", p_model=0.90, p_market_fair=0.55, odd=1.2, samples=5, policy=policy
        )
        self.assertEqual(decision.decision, NO_BET)
        self.assertIn("Quota sotto il minimo", decision.reason)

    def test_no_bet_when_odd_above_max(self):
        thresholds = DecisionThresholds(max_odd=5.0)
        policy = DecisionPolicy(version="test_max_odd", per_market={"h2h": thresholds})
        decision = evaluate_decision(
            market="h2h", outcome="Home", p_model=0.90, p_market_fair=0.10, odd=8.0, samples=5, policy=policy
        )
        self.assertEqual(decision.decision, NO_BET)
        self.assertIn("Quota sopra il massimo", decision.reason)

    def test_play_when_odd_within_range(self):
        thresholds = DecisionThresholds(min_odd=1.2, max_odd=3.0)
        policy = DecisionPolicy(version="test_odd_range", per_market={"h2h": thresholds})
        decision = evaluate_decision(
            market="h2h", outcome="Home", p_model=0.70, p_market_fair=0.55, odd=1.6, samples=5, policy=policy
        )
        self.assertEqual(decision.decision, PLAY)

    def test_default_no_odd_limits_never_blocks(self):
        decision = evaluate_decision(market="h2h", outcome="Home", p_model=0.70, p_market_fair=0.55, odd=50.0, samples=0)
        # Nessun limite di default: una quota alta non blocca la decisione.
        self.assertNotEqual(decision.reason, "Quota sopra il massimo consentito")


class TestMinProbEdgeFilter(unittest.TestCase):
    """Acceptance criteria "Min edge/EV" (prob_edge esplicito, oltre a EV)."""

    def test_no_bet_when_prob_edge_below_threshold_even_if_probability_and_ev_are_high(self):
        # p_model=0.70, p_market_fair=0.65 -> prob_edge=0.05 (basso), ev=0.70*1.6-1=0.12 (alto)
        thresholds = DecisionThresholds(play_min_prob_edge=0.10, borderline_min_prob_edge=0.10)
        policy = DecisionPolicy(version="test_min_edge", per_market={"h2h": thresholds})
        decision = evaluate_decision(
            market="h2h", outcome="Home", p_model=0.70, p_market_fair=0.65, odd=1.6, samples=5, policy=policy
        )
        self.assertEqual(decision.decision, NO_BET)

    def test_play_when_prob_edge_meets_threshold(self):
        thresholds = DecisionThresholds(play_min_prob_edge=0.10)
        policy = DecisionPolicy(version="test_min_edge", per_market={"h2h": thresholds})
        # p_model=0.70, p_market_fair=0.55 -> prob_edge=0.15 (>= 0.10)
        decision = evaluate_decision(
            market="h2h", outcome="Home", p_model=0.70, p_market_fair=0.55, odd=1.6, samples=5, policy=policy
        )
        self.assertEqual(decision.decision, PLAY)


class TestPerMarketOutcomeOverride(unittest.TestCase):
    """Acceptance criteria "Soglie per mercato/outcome"."""

    def test_per_market_override_takes_precedence_over_default(self):
        strict = DecisionThresholds(play_min_probability=0.95, borderline_min_probability=0.95)
        policy = DecisionPolicy(version="test_override", per_market={"h2h": strict})
        decision = evaluate_decision(
            market="h2h", outcome="Home", p_model=0.70, p_market_fair=0.55, odd=1.6, samples=0, policy=policy
        )
        self.assertEqual(decision.decision, NO_BET)

    def test_other_markets_unaffected_by_per_market_override(self):
        strict = DecisionThresholds(play_min_probability=0.95)
        policy = DecisionPolicy(version="test_override", per_market={"h2h": strict})
        # Mercato diverso da "h2h": non tocca l'override, resta il default.
        decision = evaluate_decision(
            market="corners", outcome="Over 9.5", p_model=0.70, p_market_fair=0.55, odd=1.6, samples=0, policy=policy
        )
        self.assertEqual(decision.decision, PLAY)

    def test_per_market_outcome_override_takes_precedence_over_per_market(self):
        market_thresholds = DecisionThresholds(play_min_probability=0.60)
        outcome_thresholds = DecisionThresholds(play_min_probability=0.99, borderline_min_probability=0.99)
        policy = DecisionPolicy(
            version="test_outcome_override",
            per_market={"h2h": market_thresholds},
            per_market_outcome={("h2h", "Away"): outcome_thresholds},
        )
        # "Home" usa l'override di mercato (soglia 0.60) -> PLAY
        decision_home = evaluate_decision(
            market="h2h", outcome="Home", p_model=0.70, p_market_fair=0.55, odd=1.6, samples=0, policy=policy
        )
        self.assertEqual(decision_home.decision, PLAY)
        # "Away" usa l'override specifico dell'outcome (soglia 0.99) -> NO BET
        decision_away = evaluate_decision(
            market="h2h", outcome="Away", p_model=0.70, p_market_fair=0.55, odd=1.6, samples=0, policy=policy
        )
        self.assertEqual(decision_away.decision, NO_BET)

    def test_thresholds_for_resolves_correctly(self):
        market_thresholds = DecisionThresholds(min_samples=2)
        policy = DecisionPolicy(version="v", per_market={"corners": market_thresholds})
        self.assertIs(policy.thresholds_for("corners", "Over 9.5"), market_thresholds)
        self.assertIs(policy.thresholds_for("cards", "Over 4.5"), policy.default)


class TestEvaluateDecisionFromFairOddsOutcome(unittest.TestCase):
    def test_uses_bookmakers_as_samples(self):
        fair_odds_outcome = FairOddsOutcome(
            market="goal_no_goal", outcome="Yes", odd=1.9, p_market_raw=0.526,
            p_market_fair=0.50, fair_odd=2.0, p_model=0.60, bookmakers=5,
        )
        decision = evaluate_decision_from_fair_odds_outcome(fair_odds_outcome)
        self.assertIsInstance(decision, Decision)
        self.assertEqual(decision.samples, 5)
        self.assertAlmostEqual(decision.prob_edge, 0.10, places=9)
        self.assertAlmostEqual(decision.ev, 0.60 * 1.9 - 1, places=9)

    def test_min_samples_blocks_via_bookmakers_count(self):
        fair_odds_outcome = FairOddsOutcome(
            market="h2h", outcome="Home", odd=1.6, p_market_raw=0.6,
            p_market_fair=0.55, fair_odd=1.818, p_model=0.70, bookmakers=1,
        )
        policy = DecisionPolicy(version="v", per_market={"h2h": DecisionThresholds(min_samples=3)})
        decision = evaluate_decision_from_fair_odds_outcome(fair_odds_outcome, policy=policy)
        self.assertEqual(decision.decision, NO_BET)
        self.assertIn("Campione insufficiente", decision.reason)

    def test_missing_market_data_falls_back_to_no_bet_without_crashing(self):
        fair_odds_outcome = FairOddsOutcome(
            market="corners", outcome="Over 9.5", odd=None, p_market_raw=None,
            p_market_fair=None, fair_odd=None, p_model=0.55,
        )
        decision = evaluate_decision_from_fair_odds_outcome(fair_odds_outcome)
        self.assertEqual(decision.decision, NO_BET)
        self.assertIsNone(decision.ev)
        self.assertIsNone(decision.prob_edge)


if __name__ == "__main__":
    unittest.main()



