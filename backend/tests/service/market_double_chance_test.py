import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ml.markets.market_1x2 import Market1x2Expert
from src.ml.markets.market_double_chance import (
    DOUBLE_CHANCE_OUTCOMES,
    DoubleChanceExpert,
    build_double_chance_output,
    derive_double_chance_from_dict,
    derive_double_chance_probabilities,
    fair_odds_from_probabilities,
)


class TestDeriveDoubleChanceProbabilities(unittest.TestCase):
    def test_formulas_match_task_specification(self):
        p_home, p_draw, p_away = 0.50, 0.30, 0.20

        dc = derive_double_chance_probabilities(p_home, p_draw, p_away)

        # P1X = P1 + PX
        self.assertAlmostEqual(dc["Home/Draw"], p_home + p_draw, places=9)
        # P12 = P1 + P2
        self.assertAlmostEqual(dc["Home/Away"], p_home + p_away, places=9)
        # PX2 = PX + P2
        self.assertAlmostEqual(dc["Draw/Away"], p_draw + p_away, places=9)

    def test_three_outcomes_always_available(self):
        dc = derive_double_chance_probabilities(0.4, 0.35, 0.25)
        self.assertEqual(set(dc.keys()), set(DOUBLE_CHANCE_OUTCOMES))
        self.assertEqual(len(dc), 3)

    def test_sum_of_dc_outcomes_is_two_when_1x2_sums_to_one(self):
        dc = derive_double_chance_probabilities(0.45, 0.25, 0.30)
        self.assertAlmostEqual(sum(dc.values()), 2.0, places=9)

    def test_each_dc_outcome_in_unit_interval(self):
        dc = derive_double_chance_probabilities(0.6, 0.1, 0.3)
        for value in dc.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_raises_when_1x2_probabilities_do_not_sum_to_one(self):
        with self.assertRaises(ValueError):
            derive_double_chance_probabilities(0.5, 0.5, 0.5)

    def test_consistent_across_many_random_valid_triples(self):
        rng = np.random.RandomState(0)
        for _ in range(200):
            raw = rng.dirichlet(alpha=[1.0, 1.0, 1.0])
            p_home, p_draw, p_away = float(raw[0]), float(raw[1]), float(raw[2])
            dc = derive_double_chance_probabilities(p_home, p_draw, p_away)
            self.assertAlmostEqual(dc["Home/Draw"], p_home + p_draw, places=9)
            self.assertAlmostEqual(dc["Home/Away"], p_home + p_away, places=9)
            self.assertAlmostEqual(dc["Draw/Away"], p_draw + p_away, places=9)
            self.assertAlmostEqual(sum(dc.values()), 2.0, places=9)


class TestDeriveDoubleChanceFromDict(unittest.TestCase):
    def test_uses_outcome_labels_regardless_of_dict_order(self):
        # Ordine volutamente diverso da OUTCOME_LABELS (HOME, DRAW, AWAY):
        # la derivazione deve leggere per CHIAVE, mai per posizione.
        probabilities = {"AWAY": 0.2, "HOME": 0.5, "DRAW": 0.3}
        dc = derive_double_chance_from_dict(probabilities)
        self.assertAlmostEqual(dc["Home/Draw"], 0.8, places=9)
        self.assertAlmostEqual(dc["Home/Away"], 0.7, places=9)
        self.assertAlmostEqual(dc["Draw/Away"], 0.5, places=9)

    def test_raises_when_key_missing(self):
        with self.assertRaises(ValueError):
            derive_double_chance_from_dict({"HOME": 0.6, "AWAY": 0.4})


class TestFairOddsFromProbabilities(unittest.TestCase):
    def test_fair_odd_is_inverse_of_probability(self):
        odds = fair_odds_from_probabilities({"Home/Draw": 0.8, "Draw/Away": 0.5})
        self.assertAlmostEqual(odds["Home/Draw"], 1.0 / 0.8, places=9)
        self.assertAlmostEqual(odds["Draw/Away"], 2.0, places=9)

    def test_zero_probability_maps_to_none_not_infinity(self):
        odds = fair_odds_from_probabilities({"Home/Away": 0.0})
        self.assertIsNone(odds["Home/Away"])


class TestBuildDoubleChanceOutput(unittest.TestCase):
    def test_output_contains_source_probabilities_and_fair_odds(self):
        output = build_double_chance_output({"HOME": 0.5, "DRAW": 0.3, "AWAY": 0.2})

        self.assertEqual(output["source_1x2"], {"HOME": 0.5, "DRAW": 0.3, "AWAY": 0.2})
        self.assertEqual(set(output["probabilities"].keys()), set(DOUBLE_CHANCE_OUTCOMES))
        self.assertEqual(set(output["fair_odds"].keys()), set(DOUBLE_CHANCE_OUTCOMES))
        self.assertAlmostEqual(output["fair_odds"]["Home/Draw"], 1.0 / 0.8, places=9)


class TestDoubleChanceExpert(unittest.TestCase):
    def _fitted_1x2_expert(self):
        rng = np.random.RandomState(5)
        n = 150
        X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
        y = pd.Series(np.where(X["f1"] > 0.3, "HOME", np.where(X["f1"] < -0.3, "AWAY", "DRAW")))
        model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
        return Market1x2Expert.from_estimator(estimator=model, feature_names=["f1", "f2"]), X

    def test_derive_produces_three_dc_outcomes_per_row_and_matches_formulas(self):
        expert_1x2, X = self._fitted_1x2_expert()
        dc_expert = DoubleChanceExpert(market_1x2_expert=expert_1x2)

        rows = dc_expert.derive(X.iloc[:5])
        self.assertEqual(len(rows), 5)

        for row in rows:
            source = row["source_1x2"]
            probabilities = row["probabilities"]
            self.assertEqual(set(probabilities.keys()), set(DOUBLE_CHANCE_OUTCOMES))
            self.assertAlmostEqual(probabilities["Home/Draw"], source["HOME"] + source["DRAW"], places=6)
            self.assertAlmostEqual(probabilities["Home/Away"], source["HOME"] + source["AWAY"], places=6)
            self.assertAlmostEqual(probabilities["Draw/Away"], source["DRAW"] + source["AWAY"], places=6)
            self.assertAlmostEqual(sum(probabilities.values()), 2.0, places=6)
            self.assertEqual(set(row["fair_odds"].keys()), set(DOUBLE_CHANCE_OUTCOMES))


if __name__ == "__main__":
    unittest.main()
