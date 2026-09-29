import os
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from src.api import main as api_main
from src.service_ia.training.model_registry import ModelRegistry


class TestModelLegendEndpoint(unittest.TestCase):
    def test_reflects_real_production_state_per_market(self):
        # Stesso principio di test di `markets_endpoint_test.py`: un
        # registry temporaneo, isolato, con UN mercato promosso e uno no -
        # `active_in_production` deve rispecchiare esattamente quello, mai
        # un'assunzione statica (bug reale trovato 2026-09-19: si era
        # documentato "nessun modello corners mai promosso" senza mai
        # verificarlo sul registry).
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            promosso = registry.register(
                model_path=os.path.join(tmp, "best_models", "cards", "cards_line_3_5", "champion.pkl"),
                market="cards_line_3_5",
                model_name="stacking",
            )
            registry.promote(run_id=promosso["run_id"], to_stage="production", actor="test")
            registry.register(
                model_path=os.path.join(tmp, "best_models", "corners", "corners_line_8_5", "champion.pkl"),
                market="corners_line_8_5",
                model_name="calibrated_random_forest",
            )  # mai promosso: resta "candidate"

            with mock.patch.object(api_main, "ModelRegistry", lambda: registry):
                client = TestClient(api_main.app)
                response = client.get("/models/legend")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        by_market = {entry["market"]: entry for entry in payload["entries"]}

        self.assertTrue(by_market["cards_line_3_5"]["active_in_production"])
        self.assertFalse(by_market["corners_line_8_5"]["active_in_production"])

    def test_covers_over_signal_line_market_signal_and_generic_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            with mock.patch.object(api_main, "ModelRegistry", lambda: registry):
                client = TestClient(api_main.app)
                response = client.get("/models/legend")

        payload = response.json()
        by_market = {entry["market"]: entry for entry in payload["entries"]}

        # over_signal: Under/Over gol, direzione sempre "over"
        self.assertEqual(by_market["under_over_2_5"]["policy_family"], "over_signal")
        self.assertEqual(by_market["under_over_2_5"]["direction"], "over")

        # line_market_signal: corners resta "over" (storico), cards e' "under"
        self.assertEqual(by_market["corners_line_8_5"]["direction"], "over")
        self.assertEqual(by_market["cards_line_3_5"]["direction"], "under")

        # bucket generico per i mercati senza soglia dedicata
        generic = next(e for e in payload["entries"] if e["policy_family"] == "generic_decision_policy")
        self.assertIsNone(generic["direction"])

        self.assertIn("over_signal", payload["policy_versions"])
        self.assertIn("line_market_signal", payload["policy_versions"])
        self.assertIn("generic_decision_policy", payload["policy_versions"])


if __name__ == "__main__":
    unittest.main()
