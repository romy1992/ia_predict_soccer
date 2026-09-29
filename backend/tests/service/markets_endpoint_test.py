import os
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from src.api import main as api_main
from src.service_ia.training.model_registry import ModelRegistry


class TestMarketsEndpoint(unittest.TestCase):
    def test_excludes_markets_whose_production_is_archived(self):
        # Riproduce lo scenario reale (2026-09-16): under_over_4_5 non e'
        # mai stato rifatto con la procedura nuova, il suo modello in
        # produzione e' rimasto in archivio/ - non deve piu' comparire tra
        # i mercati selezionabili (filtro "Mercato" della Dashboard,
        # selettore "Crea previsione manuale").
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            archiviato = registry.register(
                model_path=os.path.join(tmp, "best_models", "archivio", "under_over_4_5_champion.pkl"),
                market="under_over_4_5",
                model_name="logistic",
            )
            attivo = registry.register(
                model_path=os.path.join(tmp, "best_models", "h2h", "champion.pkl"),
                market="h2h",
                model_name="logistic",
            )
            registry.promote(run_id=archiviato["run_id"], to_stage="production", actor="test")
            registry.promote(run_id=attivo["run_id"], to_stage="production", actor="test")

            with mock.patch.object(api_main, "ModelRegistry", lambda: registry):
                client = TestClient(api_main.app)
                response = client.get("/markets")

            self.assertEqual(response.status_code, 200)
            values = response.json()["markets"]
            self.assertIn("h2h", values)
            self.assertNotIn("under_over_4_5", values)


if __name__ == "__main__":
    unittest.main()
