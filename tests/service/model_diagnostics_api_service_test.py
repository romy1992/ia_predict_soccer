import unittest
from unittest import mock

from src.api.model_diagnostics_service import ModelDiagnosticsService
from src.ml.evaluation.model_diagnostics_service import MarketDiagnostics


class TestModelDiagnosticsService(unittest.TestCase):
    def setUp(self):
        ModelDiagnosticsService.clear_cache()
        self.addCleanup(ModelDiagnosticsService.clear_cache)

    def test_defaults_to_all_diagnosable_markets_when_none_given(self):
        service = ModelDiagnosticsService()
        with mock.patch(
            "src.api.model_diagnostics_service.list_diagnosable_markets", return_value=["goal_no_goal", "h2h"]
        ), mock.patch(
            "src.api.model_diagnostics_service.evaluate_markets_diagnostics",
            return_value=[MarketDiagnostics(market="h2h", status="ok")],
        ) as mock_eval:
            payload = service.get_diagnostics()

        mock_eval.assert_called_once()
        self.assertEqual(mock_eval.call_args.kwargs["markets"], ["goal_no_goal", "h2h"])
        self.assertIn("generated_at", payload)
        self.assertEqual(payload["markets"], [{"market": "h2h", "status": "ok", "n_oof": None, "champion": None, "stage": None, "accuracy": None, "auc": None, "cm": None, "class0": None, "class1": None, "weighted": None, "roc_fpr": [], "roc_tpr": []}])

    def test_uses_explicit_markets_when_given(self):
        service = ModelDiagnosticsService()
        with mock.patch(
            "src.api.model_diagnostics_service.evaluate_markets_diagnostics",
            return_value=[MarketDiagnostics(market="corners", status="ok")],
        ) as mock_eval:
            service.get_diagnostics(markets=["corners"])

        self.assertEqual(mock_eval.call_args.kwargs["markets"], ["corners"])

    def test_second_call_within_ttl_is_served_from_cache(self):
        service = ModelDiagnosticsService()
        with mock.patch(
            "src.api.model_diagnostics_service.evaluate_markets_diagnostics",
            return_value=[MarketDiagnostics(market="h2h", status="ok")],
        ) as mock_eval:
            service.get_diagnostics(markets=["h2h"])
            service.get_diagnostics(markets=["h2h"])

        self.assertEqual(mock_eval.call_count, 1)

    def test_force_refresh_bypasses_cache(self):
        service = ModelDiagnosticsService()
        with mock.patch(
            "src.api.model_diagnostics_service.evaluate_markets_diagnostics",
            return_value=[MarketDiagnostics(market="h2h", status="ok")],
        ) as mock_eval:
            service.get_diagnostics(markets=["h2h"])
            service.get_diagnostics(markets=["h2h"], force_refresh=True)

        self.assertEqual(mock_eval.call_count, 2)

    def test_different_market_sets_use_different_cache_entries(self):
        service = ModelDiagnosticsService()
        with mock.patch(
            "src.api.model_diagnostics_service.evaluate_markets_diagnostics",
            return_value=[MarketDiagnostics(market="h2h", status="ok")],
        ) as mock_eval:
            service.get_diagnostics(markets=["h2h"])
            service.get_diagnostics(markets=["goal_no_goal"])

        self.assertEqual(mock_eval.call_count, 2)

    def test_market_order_does_not_create_separate_cache_entries(self):
        service = ModelDiagnosticsService()
        with mock.patch(
            "src.api.model_diagnostics_service.evaluate_markets_diagnostics",
            return_value=[MarketDiagnostics(market="h2h", status="ok")],
        ) as mock_eval:
            service.get_diagnostics(markets=["h2h", "goal_no_goal"])
            service.get_diagnostics(markets=["goal_no_goal", "h2h"])

        self.assertEqual(mock_eval.call_count, 1)


if __name__ == "__main__":
    unittest.main()
