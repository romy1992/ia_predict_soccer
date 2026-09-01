import os
import tempfile
import unittest

from src.service_ia.training.model_registry import ModelRegistry


class TestModelRegistry(unittest.TestCase):
    def test_register_and_get_latest(self):
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

            latest = registry.get_latest(market="under_over_2_5")
            self.assertIsNotNone(latest)
            self.assertEqual(latest["market"], "under_over_2_5")
            self.assertIn(latest["run_id"], [payload_1["run_id"], payload_2["run_id"]])

            tail = registry.tail(limit=10, market="under_over_2_5")
            self.assertEqual(len(tail), 2)

            markets = registry.list_markets()
            self.assertEqual(markets, ["under_over_2_5"])


if __name__ == "__main__":
    unittest.main()

