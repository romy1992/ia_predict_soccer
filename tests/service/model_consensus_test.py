import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src.ml.ensemble.expert_output import ExpertOutput
from src.ml.ensemble.model_consensus import (
    ModelConsensusReport,
    _find_meta_model_run,
    build_model_consensus_for_fixture,
    compute_consensus_from_expert_outputs,
)


def _direct_output(probability: float, positive_label: str = "btts_yes") -> ExpertOutput:
    return ExpertOutput(
        expert_name="goal_no_goal",
        probability_vector={positive_label: probability, f"not_{positive_label}": 1.0 - probability},
        model_run_id="goal_no_goal_run_1",
        stage="production",
    )


def _market_odds_output(outcomes: dict[str, float]) -> ExpertOutput:
    return ExpertOutput(expert_name="market_odds", probability_vector=outcomes, feature_timestamp="2026-01-01T00:00:00+00:00")


class _FakeMetaModel:
    """Doppio minimale: espone `predict_proba` come uno stimatore sklearn."""

    def __init__(self, p1: float, raise_error: bool = False):
        self.p1 = p1
        self.raise_error = raise_error

    def predict_proba(self, X):
        if self.raise_error:
            raise RuntimeError("meta-model kaboom")
        return np.array([[1.0 - self.p1, self.p1]])


class _FakeDirectExpert:
    """Doppio minimale di `DirectMarketExpert`: SOLO gli attributi/metodi
    letti da `from_predict_proba_expert` (niente `predict_proba_dict`, a
    differenza di un `mock.Mock()` generico che lo crea automaticamente e
    farebbe prendere il ramo sbagliato dell'adapter)."""

    def __init__(self, p1: float, positive_label: str = "btts_yes"):
        self.p1 = p1
        self.positive_label = positive_label
        self.feature_names: list[str] = ["f1", "f2"]
        self.run_id = "goal_no_goal_run_1"
        self.stage = "production"
        self.market = "goal_no_goal"
        self.line = None
        self.classification_type = "binary"
        self.outcome_semantics = "P(BTTS)"

    def predict_proba(self, X):
        return np.array([self.p1])


class TestComputeConsensusFromExpertOutputs(unittest.TestCase):
    def test_single_comparable_expert_drives_simple_consensus(self):
        outputs = [_direct_output(0.7)]
        report = compute_consensus_from_expert_outputs(
            market="goal_no_goal", fixture_id=1, expert_outputs=outputs, positive_label="btts_yes"
        )

        self.assertIsInstance(report, ModelConsensusReport)
        self.assertEqual(len(report.experts), 1)
        self.assertTrue(report.experts[0]["comparable"])
        self.assertAlmostEqual(report.experts[0]["probability"], 0.7)
        self.assertEqual(report.oracle_final["source"], "simple_consensus_mean")
        self.assertAlmostEqual(report.oracle_final["probability"], 0.7)
        self.assertAlmostEqual(report.consensus["dispersion_std"], 0.0)
        self.assertEqual(report.consensus["agreement_level"], "high")
        self.assertEqual(report.consensus["comparable_expert_count"], 1)

    def test_two_agreeing_experts_have_low_dispersion(self):
        outputs = [_direct_output(0.72), _market_odds_output({"Yes": 0.70, "No": 0.30})]
        report = compute_consensus_from_expert_outputs(
            market="goal_no_goal", fixture_id=1, expert_outputs=outputs, positive_label="btts_yes"
        )

        self.assertEqual(report.consensus["comparable_expert_count"], 2)
        self.assertLess(report.consensus["dispersion_std"], 0.05)
        self.assertEqual(report.consensus["agreement_level"], "high")
        self.assertAlmostEqual(report.oracle_final["probability"], 0.71, places=6)

    def test_two_disagreeing_experts_have_low_agreement(self):
        outputs = [_direct_output(0.9), _market_odds_output({"Yes": 0.20, "No": 0.80})]
        report = compute_consensus_from_expert_outputs(
            market="goal_no_goal", fixture_id=1, expert_outputs=outputs, positive_label="btts_yes"
        )

        self.assertEqual(report.consensus["agreement_level"], "low")
        self.assertGreater(report.consensus["dispersion_std"], 0.15)

    def test_non_comparable_expert_is_reported_but_excluded_from_consensus(self):
        # 'corners' non e' in _POSITIVE_OUTCOME_LABEL_FOR_MARKET: la soglia
        # (linea) e' dinamica, quindi il market_odds non e' comparabile.
        outputs = [
            _direct_output(0.6, positive_label="over_corners_threshold"),
            _market_odds_output({"Over 9.5": 0.55, "Under 9.5": 0.45}),
        ]
        report = compute_consensus_from_expert_outputs(
            market="corners", fixture_id=1, expert_outputs=outputs, positive_label="over_corners_threshold"
        )

        self.assertEqual(len(report.experts), 2)
        market_odds_payload = next(e for e in report.experts if e["expert_name"] == "market_odds")
        self.assertFalse(market_odds_payload["comparable"])
        self.assertIsNone(market_odds_payload["probability"])
        # Solo il direct expert e' comparabile: dispersion 0, oracle_final = 0.6.
        self.assertEqual(report.consensus["comparable_expert_count"], 1)
        self.assertAlmostEqual(report.oracle_final["probability"], 0.6)

    def test_meta_model_is_used_when_provided(self):
        outputs = [_direct_output(0.6), _market_odds_output({"Yes": 0.55, "No": 0.45})]
        meta_model = _FakeMetaModel(p1=0.65)

        report = compute_consensus_from_expert_outputs(
            market="goal_no_goal",
            fixture_id=1,
            expert_outputs=outputs,
            positive_label="btts_yes",
            meta_model=meta_model,
            meta_model_run_id="goal_no_goal_meta_run_1",
        )

        self.assertEqual(report.oracle_final["source"], "meta_model")
        self.assertAlmostEqual(report.oracle_final["probability"], 0.65)
        self.assertEqual(report.oracle_final["model_run_id"], "goal_no_goal_meta_run_1")

    def test_meta_model_failure_falls_back_to_simple_consensus(self):
        outputs = [_direct_output(0.6)]
        meta_model = _FakeMetaModel(p1=0.0, raise_error=True)

        report = compute_consensus_from_expert_outputs(
            market="goal_no_goal",
            fixture_id=1,
            expert_outputs=outputs,
            positive_label="btts_yes",
            meta_model=meta_model,
        )

        self.assertEqual(report.oracle_final["source"], "simple_consensus_mean")
        self.assertTrue(any("meta_model_failed" in w for w in report.warnings))

    def test_no_experts_available_reports_warnings(self):
        report = compute_consensus_from_expert_outputs(market="goal_no_goal", fixture_id=1, expert_outputs=[])

        self.assertEqual(report.experts, [])
        self.assertIsNone(report.oracle_final)
        self.assertIn("no_expert_output_available", report.warnings)
        self.assertIn("oracle_final_not_available", report.warnings)

    def test_expert_payload_structure(self):
        report = compute_consensus_from_expert_outputs(
            market="goal_no_goal", fixture_id=42, expert_outputs=[_direct_output(0.55)], positive_label="btts_yes"
        )
        payload = report.experts[0]
        for key in (
            "expert_name",
            "expert_version",
            "probability_vector",
            "probability",
            "comparable",
            "confidence",
            "model_run_id",
            "stage",
            "feature_timestamp",
        ):
            self.assertIn(key, payload)


class TestFindMetaModelRun(unittest.TestCase):
    def test_selects_only_runs_with_meta_prefix(self):
        registry = mock.Mock()
        registry.tail.return_value = [
            {"run_id": "a", "model_name": "logistic", "created_at": "2026-01-01T00:00:00+00:00"},
            {"run_id": "b", "model_name": "meta_weighted_blend", "created_at": "2026-01-02T00:00:00+00:00"},
        ]
        run = _find_meta_model_run(registry, market="btts")
        self.assertEqual(run["run_id"], "b")

    def test_prefers_production_stage_among_meta_runs(self):
        registry = mock.Mock()
        registry.tail.return_value = [
            {
                "run_id": "old_candidate",
                "model_name": "meta_weighted_blend",
                "created_at": "2026-01-01T00:00:00+00:00",
                "current_stage": "candidate",
            },
            {
                "run_id": "prod",
                "model_name": "oracle_ensemble_learned_stacker_calibrated",
                "created_at": "2025-06-01T00:00:00+00:00",
                "current_stage": "production",
            },
            {
                "run_id": "newer_candidate",
                "model_name": "meta_learned_stacker",
                "created_at": "2026-02-01T00:00:00+00:00",
                "current_stage": "candidate",
            },
        ]
        run = _find_meta_model_run(registry, market="btts")
        self.assertEqual(run["run_id"], "prod")

    def test_returns_none_when_no_meta_run_exists(self):
        registry = mock.Mock()
        registry.tail.return_value = [{"run_id": "a", "model_name": "logistic", "created_at": "2026-01-01T00:00:00+00:00"}]
        self.assertIsNone(_find_meta_model_run(registry, market="btts"))


class TestBuildModelConsensusForFixture(unittest.TestCase):
    def test_raises_on_unsupported_market(self):
        with self.assertRaises(ValueError):
            build_model_consensus_for_fixture(market="not_a_market", fixture_id=1)

    def test_end_to_end_with_direct_and_market_odds(self):
        registry = mock.Mock()
        registry.get_production.return_value = {
            "run_id": "goal_no_goal_run_1",
            "market": "goal_no_goal",
            "model_name": "logistic",
            "model_path": "/tmp/fake.pkl",
            "feature_names": ["f1", "f2"],
            "current_stage": "production",
        }
        registry.tail.return_value = []  # nessun meta-model registrato

        filter_service = mock.Mock()
        filter_service.build_prediction_frame.return_value = pd.DataFrame(
            [{"f1": 1.0, "f2": 2.0, "market": "goal_no_goal", "id_fixture": 10, "season": 2026, "league": 39, "prediction_at": "x"}]
        )

        market_odds_expert = mock.Mock()
        market_odds_expert.build_market_signal.return_value = {
            "signal_version": "v1",
            "fixture_id": 10,
            "market": "goal_no_goal",
            "as_of": "2026-01-01T00:00:00+00:00",
            "fair_probabilities": {"outcomes": [{"outcome": "Yes", "fair_probability": 0.6}, {"outcome": "No", "fair_probability": 0.4}]},
            "dispersion": {},
            "movement": {},
            "opening_latest_closing": [],
        }

        fake_estimator = mock.Mock()
        fake_estimator.predict_proba.return_value = np.array([[0.35, 0.65]])

        with mock.patch(
            "src.ml.ensemble.model_consensus.DirectMarketExpert._from_run"
        ) as from_run_mock:
            from_run_mock.return_value = _FakeDirectExpert(p1=0.65)

            report = build_model_consensus_for_fixture(
                market="goal_no_goal",
                fixture_id=10,
                registry=registry,
                filter_service=filter_service,
                market_odds_expert=market_odds_expert,
            )

        self.assertEqual(report.market, "goal_no_goal")
        self.assertEqual(report.fixture_id, 10)
        self.assertEqual(len(report.experts), 2)
        self.assertIsNotNone(report.oracle_final)
        self.assertEqual(report.oracle_final["source"], "simple_consensus_mean")

    def test_missing_direct_expert_still_produces_market_odds_only_consensus(self):
        registry = mock.Mock()
        registry.get_production.return_value = None
        registry.get_latest.return_value = None
        registry.tail.return_value = []

        filter_service = mock.Mock()
        market_odds_expert = mock.Mock()
        market_odds_expert.build_market_signal.return_value = {
            "signal_version": "v1",
            "fixture_id": 10,
            "market": "goal_no_goal",
            "as_of": "2026-01-01T00:00:00+00:00",
            "fair_probabilities": {"outcomes": [{"outcome": "Yes", "fair_probability": 0.6}, {"outcome": "No", "fair_probability": 0.4}]},
            "dispersion": {},
            "movement": {},
            "opening_latest_closing": [],
        }

        report = build_model_consensus_for_fixture(
            market="goal_no_goal",
            fixture_id=10,
            registry=registry,
            filter_service=filter_service,
            market_odds_expert=market_odds_expert,
        )

        self.assertEqual(len(report.experts), 1)
        self.assertEqual(report.experts[0]["expert_name"], "market_odds")
        # Nessun positive_label noto (niente direct expert): market_odds non
        # e' marcato comparabile per un mercato senza mapping esplicito
        # basato sul positive_label, ma qui 'goal_no_goal' HA un mapping
        # statico -> resta comparabile e guida l'oracle_final.
        self.assertIsNotNone(report.oracle_final)

    def test_no_data_available_returns_empty_but_valid_report(self):
        registry = mock.Mock()
        registry.get_production.return_value = None
        registry.get_latest.return_value = None
        registry.tail.return_value = []

        filter_service = mock.Mock()
        market_odds_expert = mock.Mock()
        market_odds_expert.build_market_signal.side_effect = RuntimeError("no odds table")

        report = build_model_consensus_for_fixture(
            market="goal_no_goal",
            fixture_id=999,
            registry=registry,
            filter_service=filter_service,
            market_odds_expert=market_odds_expert,
        )

        self.assertEqual(report.experts, [])
        self.assertIsNone(report.oracle_final)
        self.assertIn("no_expert_output_available", report.warnings)


if __name__ == "__main__":
    unittest.main()


