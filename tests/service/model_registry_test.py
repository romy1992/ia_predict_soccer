import os
import tempfile
import unittest

from src.service_ia.training.model_registry import ModelRegistry


class TestModelRegistry(unittest.TestCase):
    def test_register_persists_lifecycle_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)

            payload = registry.register(
                model_path=os.path.join(tmp, "model_a.pkl"),
                market="under_over_2_5",
                model_name="logistic",
                metrics={"f1": 0.71},
                feature_names=["f1", "f2"],
                dataset_version="dataset:v1",
                feature_version="features:v1",
                windows={
                    "train": {"from": "2024-01-01", "to": "2025-06-01"},
                    "validation": {"from": "2025-06-02", "to": "2025-08-01"},
                    "test": {"from": "2025-08-02", "to": "2025-09-01"},
                },
                git_sha="abc1234",
                stage="candidate",
            )

            latest = registry.get_latest(market="under_over_2_5")
            self.assertIsNotNone(latest)
            self.assertEqual(latest["run_id"], payload["run_id"])
            self.assertEqual(latest["dataset_version"], "dataset:v1")
            self.assertEqual(latest["feature_version"], "features:v1")
            self.assertEqual(latest["git_sha"], "abc1234")
            self.assertEqual(latest["current_stage"], "candidate")
            self.assertEqual(len(latest["promotion_history"]), 0)

    def test_latest_is_not_implicitly_production(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)

            payload_1 = registry.register(
                model_path=os.path.join(tmp, "model_a.pkl"),
                market="under_over_2_5",
                model_name="logistic",
                metrics={"f1": 0.71},
                feature_names=["f1", "f2"],
            )
            payload_2 = registry.register(
                model_path=os.path.join(tmp, "model_b.pkl"),
                market="under_over_2_5",
                model_name="stacking",
                metrics={"f1": 0.75},
                feature_names=["f1", "f2", "f3"],
            )

            self.assertIsNone(registry.get_production(market="under_over_2_5"))

            promoted = registry.promote(
                run_id=payload_1["run_id"],
                to_stage="production",
                reason="manual promote",
                actor="test",
            )
            self.assertIsNotNone(promoted)
            self.assertEqual(promoted["current_stage"], "production")

            latest = registry.get_latest(market="under_over_2_5")
            production = registry.get_production(market="under_over_2_5")
            self.assertIsNotNone(latest)
            self.assertIsNotNone(production)
            self.assertEqual(latest["market"], "under_over_2_5")
            self.assertEqual(latest["run_id"], payload_2["run_id"])
            self.assertEqual(production["run_id"], payload_1["run_id"])
            self.assertNotEqual(latest["run_id"], production["run_id"])

            tail = registry.tail(limit=10, market="under_over_2_5")
            self.assertEqual(len(tail), 2)

            markets = registry.list_markets()
            self.assertEqual(markets, ["under_over_2_5"])

    def test_promoting_new_production_demotes_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)

            payload_1 = registry.register(
                model_path=os.path.join(tmp, "model_a.pkl"),
                market="h2h",
                model_name="logistic",
            )
            payload_2 = registry.register(
                model_path=os.path.join(tmp, "model_b.pkl"),
                market="h2h",
                model_name="stacking",
            )

            registry.promote(run_id=payload_1["run_id"], to_stage="production", reason="first production", actor="test")
            registry.promote(run_id=payload_2["run_id"], to_stage="production", reason="better model", actor="test")

            production = registry.get_production(market="h2h")
            run_1 = registry.get_run(payload_1["run_id"])
            run_2 = registry.get_run(payload_2["run_id"])

            self.assertIsNotNone(production)
            self.assertIsNotNone(run_1)
            self.assertIsNotNone(run_2)
            self.assertEqual(production["run_id"], payload_2["run_id"])
            self.assertEqual(run_1["current_stage"], "champion")
            self.assertEqual(run_2["current_stage"], "production")
            self.assertGreaterEqual(len(run_1["promotion_history"]), 2)
            self.assertGreaterEqual(len(run_2["promotion_history"]), 1)

    def test_lookup_production_per_market(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)

            under_prod = registry.register(
                model_path=os.path.join(tmp, "under_prod.pkl"),
                market="under_over_2_5",
                model_name="logistic",
            )
            under_latest = registry.register(
                model_path=os.path.join(tmp, "under_latest.pkl"),
                market="under_over_2_5",
                model_name="stacking",
            )
            h2h_prod = registry.register(
                model_path=os.path.join(tmp, "h2h_prod.pkl"),
                market="h2h",
                model_name="xgboost",
            )

            registry.promote(run_id=under_prod["run_id"], to_stage="production", reason="uo prod", actor="test")
            registry.promote(run_id=h2h_prod["run_id"], to_stage="production", reason="h2h prod", actor="test")

            production_under = registry.get_production(market="under_over_2_5")
            production_h2h = registry.get_production(market="h2h")
            latest_under = registry.get_latest(market="under_over_2_5")

            self.assertIsNotNone(production_under)
            self.assertIsNotNone(production_h2h)
            self.assertIsNotNone(latest_under)
            self.assertEqual(production_under["run_id"], under_prod["run_id"])
            self.assertEqual(production_h2h["run_id"], h2h_prod["run_id"])
            self.assertEqual(latest_under["run_id"], under_latest["run_id"])
            self.assertNotEqual(latest_under["run_id"], production_under["run_id"])
            self.assertIsNone(registry.get_production(market="1x2"))

    def test_lifecycle_reaches_retired_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)

            payload = registry.register(
                model_path=os.path.join(tmp, "model_retire.pkl"),
                market="under_over_2_5",
                model_name="logistic",
            )

            registry.promote(run_id=payload["run_id"], to_stage="champion", reason="validated", actor="test")
            registry.promote(run_id=payload["run_id"], to_stage="production", reason="go live", actor="test")
            registry.promote(run_id=payload["run_id"], to_stage="retired", reason="sunset", actor="test")

            run_state = registry.get_run(payload["run_id"])
            production = registry.get_production(market="under_over_2_5")

            self.assertIsNotNone(run_state)
            self.assertEqual(run_state["current_stage"], "retired")
            self.assertGreaterEqual(len(run_state["promotion_history"]), 3)
            self.assertIsNone(production)


if __name__ == "__main__":
    unittest.main()


