import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ml.baselines.bookmaker_baseline import compute_market_baseline
from src.ml.ensemble.adapters import (
    from_goal_distribution,
    from_market_odds,
    from_predict_proba_expert,
    from_statistics,
    from_team_strength,
)
from src.ml.ensemble.expert_output import ExpertOutput, combine_expert_outputs
from src.ml.experts.direct.direct_market_expert import DirectMarketExpert
from src.ml.experts.goal_distribution.goal_distribution_expert import GoalDistributionExpert


class _FakeLineExpert:
    """Simula l'interfaccia comune di CornersExpert/CardsExpert (MARKET-05/06)
    senza dipendere dal loro training reale: qui si testa SOLO l'adapter."""

    def __init__(self, line, run_id, stage, market="cards"):
        self.line = line
        self.run_id = run_id
        self.stage = stage
        self.market = market

    def predict_proba_dict(self, X):
        n = len(X)
        over = np.linspace(0.1, 0.9, n)
        return {"over": over, "under": 1.0 - over}


class TestExpertOutputSchema(unittest.TestCase):
    def test_expert_name_required(self):
        with self.assertRaises(ValueError):
            ExpertOutput(expert_name="")

    def test_probability_vector_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            ExpertOutput(expert_name="dummy", probability_vector={"home": 1.5})
        with self.assertRaises(ValueError):
            ExpertOutput(expert_name="dummy", probability_vector={"home": -0.1})

    def test_default_confidence_computed_when_probability_vector_present(self):
        output = ExpertOutput(expert_name="dummy", probability_vector={"yes": 0.9, "no": 0.1})
        self.assertAlmostEqual(output.confidence, 0.8)

    def test_confidence_none_when_probability_vector_empty(self):
        output = ExpertOutput(expert_name="team_strength", probability_vector={})
        self.assertIsNone(output.confidence)

    def test_explicit_confidence_is_preserved(self):
        output = ExpertOutput(expert_name="dummy", probability_vector={"yes": 0.51}, confidence=0.99)
        self.assertEqual(output.confidence, 0.99)

    def test_as_feature_row_prefixes_with_expert_name(self):
        output = ExpertOutput(expert_name="goal_distribution", probability_vector={"over_2_5": 0.6})
        row = output.as_feature_row()
        self.assertIn("goal_distribution__over_2_5", row)
        self.assertIn("goal_distribution__confidence", row)
        self.assertEqual(row["goal_distribution__over_2_5"], 0.6)

    def test_created_at_is_populated(self):
        output = ExpertOutput(expert_name="dummy")
        self.assertTrue(output.created_at)


class TestCombineExpertOutputs(unittest.TestCase):
    def test_combine_merges_rows_from_different_experts(self):
        outputs = [
            ExpertOutput(expert_name="statistics", probability_vector={"home_win": 0.6, "not_home_win": 0.4}),
            ExpertOutput(expert_name="market_odds", probability_vector={"Home": 0.55}),
        ]
        combined = combine_expert_outputs(outputs)
        self.assertIn("statistics__home_win", combined)
        self.assertIn("market_odds__Home", combined)

    def test_combine_raises_on_column_collision(self):
        outputs = [
            ExpertOutput(expert_name="dup", probability_vector={"a": 0.5}),
            ExpertOutput(expert_name="dup", probability_vector={"a": 0.6}),
        ]
        with self.assertRaises(ValueError):
            combine_expert_outputs(outputs)


class TestFromTeamStrength(unittest.TestCase):
    def test_probability_vector_is_empty_and_raw_output_preserved(self):
        rating_row = {
            "prediction_at": "2025-01-01T00:00:00+00:00",
            "rating_version": "abc123",
            "home_team_attack_rating": 1.4,
            "home_team_defense_rating": 1.1,
        }
        output = from_team_strength(rating_row, team_role="home_team")

        self.assertEqual(output.expert_name, "team_strength")
        self.assertEqual(output.probability_vector, {})
        self.assertIsNone(output.confidence)
        self.assertEqual(output.expert_version, "abc123")
        self.assertEqual(output.feature_timestamp, "2025-01-01T00:00:00+00:00")
        self.assertEqual(output.raw_output["home_team_attack_rating"], 1.4)
        self.assertEqual(output.metadata["team_role"], "home_team")


class TestFromGoalDistribution(unittest.TestCase):
    def test_uses_real_expert_output_over_under(self):
        expert = GoalDistributionExpert()
        native_output = expert.build_expert_output(home_lambda=1.6, away_lambda=1.1)

        output = from_goal_distribution(native_output, feature_timestamp="2025-02-01T00:00:00+00:00")

        self.assertEqual(output.expert_name, "goal_distribution")
        self.assertEqual(output.expert_version, GoalDistributionExpert.VERSION)
        self.assertEqual(output.probability_vector, native_output["over_under"])
        self.assertEqual(output.metadata["home_lambda"], native_output["home_lambda"])
        for p in output.probability_vector.values():
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_score_matrix_variant_uses_alternate_key(self):
        expert = GoalDistributionExpert()
        native_output = expert.build_expert_output(home_lambda=1.6, away_lambda=1.1)

        output = from_goal_distribution(native_output, use_score_matrix_probabilities=True)

        self.assertEqual(output.probability_vector, native_output["over_under_from_score_matrix"])


class TestFromStatistics(unittest.TestCase):
    def test_produces_complementary_binary_vector(self):
        output = from_statistics(0.73, outcome="home_win", id_fixture=42)

        self.assertEqual(output.expert_name, "statistics")
        self.assertAlmostEqual(output.probability_vector["home_win"], 0.73)
        self.assertAlmostEqual(output.probability_vector["not_home_win"], 0.27)
        self.assertEqual(output.metadata["id_fixture"], 42)


class TestFromMarketOdds(unittest.TestCase):
    def test_uses_real_compute_market_baseline(self):
        odds_rows = [
            {"outcome": "Home", "avg_odd": 2.0, "bookmakers": 3},
            {"outcome": "Draw", "avg_odd": 3.4, "bookmakers": 3},
            {"outcome": "Away", "avg_odd": 4.0, "bookmakers": 3},
        ]
        fair_probabilities = compute_market_baseline(market="h2h", odds_rows=odds_rows)
        signal = {
            "signal_version": "sig-v1",
            "fixture_id": 999,
            "market": "h2h",
            "as_of": "2025-03-01T12:00:00+00:00",
            "fair_probabilities": fair_probabilities,
            "dispersion": {"Home": {"bookmaker_count": 3}},
            "movement": {},
            "opening_latest_closing": [],
        }

        output = from_market_odds(signal)

        self.assertEqual(output.expert_name, "market_odds")
        self.assertEqual(output.expert_version, "sig-v1")
        self.assertEqual(output.feature_timestamp, "2025-03-01T12:00:00+00:00")
        self.assertIn("Home", output.probability_vector)
        self.assertIn("Draw", output.probability_vector)
        self.assertIn("Away", output.probability_vector)
        total = sum(output.probability_vector.values())
        self.assertAlmostEqual(total, 1.0, places=6)  # h2h e' esclusivo -> normalizzato a 1
        self.assertEqual(output.metadata["fixture_id"], 999)


class TestFromPredictProbaExpert(unittest.TestCase):
    def _fitted_direct_expert(self):
        rng = np.random.RandomState(0)
        X = pd.DataFrame({"f1": rng.normal(size=50), "f2": rng.normal(size=50)})
        y = (X["f1"] > 0).astype(int)
        model = LogisticRegression().fit(X, y)
        expert = DirectMarketExpert.from_estimator(market="h2h", estimator=model, feature_names=["f1", "f2"])
        return expert, X

    def test_direct_market_expert_produces_one_output_per_row(self):
        expert, X = self._fitted_direct_expert()

        outputs = from_predict_proba_expert(expert, X)

        self.assertEqual(len(outputs), len(X))
        expected_proba = expert.predict_proba(X)
        for output, p in zip(outputs, expected_proba):
            self.assertEqual(output.expert_name, "h2h")
            self.assertAlmostEqual(output.probability_vector["home_win"], float(p))
            self.assertAlmostEqual(output.probability_vector["not_home_win"], float(1.0 - p))

    def test_line_expert_with_predict_proba_dict_uses_line_in_name(self):
        expert = _FakeLineExpert(line=4.5, run_id="cards_line_4_5_20260101T000000000000Z", stage="production")
        X = pd.DataFrame({"f1": [0, 1, 2]})

        outputs = from_predict_proba_expert(expert, X)

        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0].expert_name, "cards_line_4_5")
        self.assertEqual(outputs[0].model_run_id, "cards_line_4_5_20260101T000000000000Z")
        self.assertEqual(outputs[0].stage, "production")
        self.assertIn("over", outputs[0].probability_vector)
        self.assertIn("under", outputs[0].probability_vector)
        np.testing.assert_allclose(
            outputs[0].probability_vector["over"] + outputs[0].probability_vector["under"], 1.0
        )

    def test_feature_timestamps_length_mismatch_raises(self):
        expert = _FakeLineExpert(line=4.5, run_id="r1", stage="candidate")
        X = pd.DataFrame({"f1": [0, 1, 2]})
        with self.assertRaises(ValueError):
            from_predict_proba_expert(expert, X, feature_timestamps=["2025-01-01T00:00:00Z"])

    def test_outputs_from_different_experts_are_consumable_by_meta_model(self):
        """Acceptance criteria ORACLE-01: 'Tutti gli esperti consumabili da meta-model'."""
        direct_expert, X = self._fitted_direct_expert()
        line_expert = _FakeLineExpert(line=4.5, run_id="r1", stage="production", market="cards")

        goal_expert = GoalDistributionExpert()
        goal_native = goal_expert.build_expert_output(home_lambda=1.5, away_lambda=1.2)

        row_outputs = [
            from_predict_proba_expert(direct_expert, X.iloc[[0]])[0],
            from_predict_proba_expert(line_expert, X.iloc[[0]])[0],
            from_goal_distribution(goal_native, feature_timestamp="2025-01-01T00:00:00Z"),
            from_statistics(0.6, outcome="home_win"),
        ]

        meta_model_row = combine_expert_outputs(row_outputs)

        self.assertIn("h2h__home_win", meta_model_row)
        self.assertIn("cards_line_4_5__over", meta_model_row)  # market+line -> naming coerente con CardsExpert
        self.assertIn("goal_distribution__over_1_5", meta_model_row)
        self.assertIn("statistics__home_win", meta_model_row)
        self.assertTrue(all(isinstance(v, (int, float)) for v in meta_model_row.values()))


if __name__ == "__main__":
    unittest.main()

