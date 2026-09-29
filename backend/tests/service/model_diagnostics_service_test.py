import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ml.evaluation.model_diagnostics_service import (
    MarketDiagnostics,
    evaluate_market_diagnostics,
    evaluate_markets_diagnostics,
    list_diagnosable_markets,
)
from src.service_ia.training.market_service.filter_market_service import FilterMarketService


def _synthetic_binary_dataset(n: int = 150, seed: int = 7) -> pd.DataFrame:
    """Dataset sintetico con la STESSA shape di `FilterMarketService.build_dataset`
    (meta columns + `y` + feature numeriche) - `feature_signal` e' costruita
    per essere davvero predittiva di `y`, cosi' il modello risultante ha
    un'AUC ben sopra 0.5 (evita un test fragile su un caso limite/degenere)."""
    rng = np.random.RandomState(seed)
    base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        signal = rng.normal()
        noise = rng.normal(scale=0.3)
        y = int(signal + noise > 0)
        rows.append(
            {
                "id_fixture": 1000 + i,
                "season": 2025,
                "league": 39,
                "market": "h2h",
                "prediction_at": (base_date + timedelta(days=i)).isoformat(),
                "y": y,
                "feature_signal": signal,
                "feature_noise": float(rng.normal()),
            }
        )
    return pd.DataFrame(rows)


class _FakeRegistry:
    def __init__(self, production=None, latest=None, markets=None):
        self._production = production or {}
        self._latest = latest or {}
        self._markets = markets or []

    def get_production(self, market):
        return self._production.get(market)

    def get_latest(self, market):
        return self._latest.get(market)

    def list_markets(self):
        return list(self._markets)


class TestListDiagnosableMarkets(unittest.TestCase):
    def test_excludes_1x2_and_specialized_line_markets(self):
        registry = _FakeRegistry(markets=["h2h", "1x2", "corners_line_9_5", "goal_no_goal", "cards_line_4_5"])
        self.assertEqual(list_diagnosable_markets(registry=registry), ["goal_no_goal", "h2h"])

    def test_empty_when_nothing_registered(self):
        registry = _FakeRegistry(markets=[])
        self.assertEqual(list_diagnosable_markets(registry=registry), [])


class TestEvaluateMarketDiagnosticsStatuses(unittest.TestCase):
    def test_no_model_when_nothing_registered(self):
        registry = _FakeRegistry()
        result = evaluate_market_diagnostics(market="h2h", registry=registry)
        self.assertEqual(result.status, "no_model")
        self.assertIsNone(result.accuracy)

    def test_no_model_when_model_path_does_not_exist_on_disk(self):
        registry = _FakeRegistry(production={"h2h": {"model_path": "/nonexistent/path.pkl"}})
        result = evaluate_market_diagnostics(market="h2h", registry=registry)
        self.assertEqual(result.status, "no_model")

    def test_insufficient_data_when_dataset_is_empty(self):
        registry = _FakeRegistry(production={"h2h": {"model_path": __file__}})  # file esistente, path irrilevante qui
        with mock.patch.object(FilterMarketService, "build_dataset", return_value=pd.DataFrame()):
            result = evaluate_market_diagnostics(market="h2h", registry=registry)
        self.assertEqual(result.status, "insufficient_data")

    def test_insufficient_data_when_too_few_rows_for_temporal_cv(self):
        registry = _FakeRegistry(production={"h2h": {"model_path": __file__}})
        tiny_df = _synthetic_binary_dataset(n=5)
        with mock.patch.object(FilterMarketService, "build_dataset", return_value=tiny_df):
            result = evaluate_market_diagnostics(market="h2h", registry=registry)
        self.assertEqual(result.status, "insufficient_data")


class TestEvaluateMarketDiagnosticsEndToEnd(unittest.TestCase):
    def setUp(self):
        self._tmp_model_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        self._tmp_model_file.close()
        self.model_path = self._tmp_model_file.name
        joblib.dump(LogisticRegression(max_iter=1000), self.model_path)

    def test_ok_status_with_full_report(self):
        registry = _FakeRegistry(
            production={"h2h": {"model_path": self.model_path, "current_stage": "production"}}
        )
        df = _synthetic_binary_dataset(n=150)

        with mock.patch.object(FilterMarketService, "build_dataset", return_value=df):
            result = evaluate_market_diagnostics(market="h2h", registry=registry)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.market, "h2h")
        self.assertEqual(result.champion, "LogisticRegression")
        self.assertEqual(result.stage, "production")
        self.assertIsInstance(result.n_oof, int)
        self.assertGreater(result.n_oof, 0)
        self.assertIsNotNone(result.accuracy)
        self.assertIsNotNone(result.auc)
        self.assertGreaterEqual(result.auc, 0.0)
        self.assertLessEqual(result.auc, 1.0)

        cm = result.cm
        self.assertEqual(cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"], result.n_oof)

        for key in ("precision", "recall", "f1", "support"):
            self.assertIn(key, result.class0)
            self.assertIn(key, result.class1)
        for key in ("precision", "recall", "f1"):
            self.assertIn(key, result.weighted)

        self.assertGreater(len(result.roc_fpr), 0)
        self.assertEqual(len(result.roc_fpr), len(result.roc_tpr))

    def test_falls_back_to_latest_when_no_production(self):
        registry = _FakeRegistry(latest={"h2h": {"model_path": self.model_path, "current_stage": "candidate"}})
        df = _synthetic_binary_dataset(n=150)

        with mock.patch.object(FilterMarketService, "build_dataset", return_value=df):
            result = evaluate_market_diagnostics(market="h2h", registry=registry)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.stage, "candidate")

    def test_to_dict_returns_plain_dict(self):
        registry = _FakeRegistry(production={"h2h": {"model_path": self.model_path}})
        df = _synthetic_binary_dataset(n=150)

        with mock.patch.object(FilterMarketService, "build_dataset", return_value=df):
            result = evaluate_market_diagnostics(market="h2h", registry=registry)

        payload = result.to_dict()
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["market"], "h2h")
        self.assertEqual(payload["status"], "ok")


class TestEvaluateMarketsDiagnosticsBatch(unittest.TestCase):
    def setUp(self):
        self._tmp_model_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        self._tmp_model_file.close()
        self.model_path = self._tmp_model_file.name
        joblib.dump(LogisticRegression(max_iter=1000), self.model_path)

    def test_isolates_per_market_errors_without_failing_the_whole_batch(self):
        registry = _FakeRegistry()

        def _fake_eval(market, seasons=None, registry=None):
            if market == "goal_no_goal":
                raise RuntimeError("boom")
            return MarketDiagnostics(market=market, status="ok", n_oof=100, accuracy=0.7, auc=0.65)

        with mock.patch(
            "src.ml.evaluation.model_diagnostics_service.evaluate_market_diagnostics", side_effect=_fake_eval
        ):
            results = evaluate_markets_diagnostics(markets=["h2h", "goal_no_goal"], registry=registry)

        by_market = {r.market: r for r in results}
        self.assertEqual(by_market["h2h"].status, "ok")
        self.assertTrue(by_market["goal_no_goal"].status.startswith("error:"))

    def test_defaults_to_all_diagnosable_markets_when_none_given(self):
        registry = _FakeRegistry(
            production={"h2h": {"model_path": self.model_path}},
            markets=["h2h", "1x2"],
        )
        df = _synthetic_binary_dataset(n=150)

        with mock.patch.object(FilterMarketService, "build_dataset", return_value=df):
            results = evaluate_markets_diagnostics(registry=registry)

        markets_evaluated = [r.market for r in results]
        self.assertEqual(markets_evaluated, ["h2h"])  # "1x2" escluso da list_diagnosable_markets


if __name__ == "__main__":
    unittest.main()
