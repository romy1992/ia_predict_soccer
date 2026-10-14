"""Il retrain automatico riaddestra solo i mercati con un modello in produzione."""

import os
import tempfile
import unittest
from unittest import mock

from src.service_ia.training import train_multi_market as tmm
from src.service_ia.training.model_registry import ModelRegistry


class RetrainProductionOnlyTest(unittest.TestCase):
    def _registry_with(self, tmp, production=(), candidate_only=()):
        registry = ModelRegistry(registry_dir=tmp)
        for m in production:
            run = registry.register(model_path=os.path.join(tmp, f"{m}.pkl"), market=m, model_name="logistic")
            registry.promote(run_id=run["run_id"], to_stage="production", actor="test")
        for m in candidate_only:
            registry.register(model_path=os.path.join(tmp, f"{m}.pkl"), market=m, model_name="logistic")
        return registry

    def test_default_markets_are_filtered_by_production(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry_with(tmp, production=("h2h", "dc"), candidate_only=("cards",))
            with mock.patch("src.service_ia.training.model_registry.ModelRegistry", return_value=registry):
                self.assertEqual(tmm.markets_in_production(tmm.DEFAULT_MARKETS), ["h2h", "dc"])

    def test_train_all_markets_trains_only_production_markets_unless_told(self):
        trained = []

        def fake_train(market, **_):
            trained.append(market)
            return mock.Mock(market=market, rows=1, status="ok", champion="x", best_cv_f1=0.5,
                             selected_features=[], details={})

        with mock.patch.object(tmm, "markets_in_production", return_value=["h2h", "dc"]), \
                mock.patch.object(tmm, "train_market", side_effect=fake_train), \
                mock.patch.object(tmm.bucket_store, "put_json"):
            tmm.train_all_markets()
            self.assertEqual(trained, ["h2h", "dc"])
            trained.clear()
            tmm.train_all_markets(markets=["cards"])  # esplicito: si rispetta la richiesta
            self.assertEqual(trained, ["cards"])


if __name__ == "__main__":
    unittest.main()
