"""Test per OPS-03 (Monitoring performance modello e data drift) — parte
PURA (`src/ml/monitoring/monitoring_policy.py`), nessun DB coinvolto."""

from __future__ import annotations

import unittest

from src.ml.monitoring.monitoring_policy import (
    CRITICAL,
    INFO,
    WARNING,
    MonitoringThresholds,
    compute_calibration_drift,
    compute_feature_coverage,
    evaluate_alerts,
)

PERMISSIVE_THRESHOLDS = MonitoringThresholds(
    min_prediction_volume_recent=None,
    max_calibration_ece_drift=None,
    min_roi_rolling=None,
    min_feature_coverage_ratio=None,
    max_recent_failed_jobs=None,
)


class TestComputeCalibrationDrift(unittest.TestCase):
    def test_none_when_recent_missing(self):
        self.assertIsNone(compute_calibration_drift(None, {"ece": 0.1, "sample_size": 50}))

    def test_none_when_baseline_missing(self):
        self.assertIsNone(compute_calibration_drift({"ece": 0.1, "sample_size": 50}, None))

    def test_positive_delta_means_worse_calibration(self):
        recent = {"ece": 0.30, "brier": 0.28, "log_loss": 0.80, "sample_size": 40}
        baseline = {"ece": 0.10, "brier": 0.20, "log_loss": 0.60, "sample_size": 60}
        drift = compute_calibration_drift(recent, baseline)

        self.assertAlmostEqual(drift["ece_drift"], 0.20)
        self.assertAlmostEqual(drift["brier_drift"], 0.08)
        self.assertAlmostEqual(drift["log_loss_drift"], 0.20)
        self.assertEqual(drift["recent_sample_size"], 40)
        self.assertEqual(drift["baseline_sample_size"], 60)

    def test_negative_delta_means_improvement(self):
        recent = {"ece": 0.05, "sample_size": 40}
        baseline = {"ece": 0.20, "sample_size": 40}
        drift = compute_calibration_drift(recent, baseline)

        self.assertAlmostEqual(drift["ece_drift"], -0.15)

    def test_missing_metric_key_is_none_not_zero(self):
        drift = compute_calibration_drift({"ece": 0.1, "sample_size": 10}, {"sample_size": 10})
        self.assertIsNone(drift["baseline_ece"])
        self.assertIsNone(drift["ece_drift"])


class TestComputeFeatureCoverage(unittest.TestCase):
    def test_full_coverage(self):
        coverage = compute_feature_coverage(
            expected_features=["a", "b"],
            column_presence_counts={"a": 10, "b": 10},
            sampled_frames=10,
            requested_fixtures=10,
        )
        self.assertEqual(coverage["overall_column_presence_ratio"], 1.0)
        self.assertEqual(coverage["fixture_coverage_ratio"], 1.0)
        self.assertEqual(coverage["fully_missing_features"], [])

    def test_partial_and_fully_missing_feature(self):
        coverage = compute_feature_coverage(
            expected_features=["a", "b", "c"],
            column_presence_counts={"a": 10, "b": 5},  # 'c' assente dal dict -> 0
            sampled_frames=10,
            requested_fixtures=12,
        )
        self.assertAlmostEqual(coverage["fixture_coverage_ratio"], 10 / 12)
        per_feature = {row["feature"]: row for row in coverage["per_feature"]}
        self.assertEqual(per_feature["a"]["column_presence_ratio"], 1.0)
        self.assertEqual(per_feature["b"]["column_presence_ratio"], 0.5)
        self.assertEqual(per_feature["c"]["column_presence_ratio"], 0.0)
        self.assertEqual(coverage["fully_missing_features"], ["c"])
        # overall = media dei ratio disponibili (1.0 + 0.5 + 0.0) / 3
        self.assertAlmostEqual(coverage["overall_column_presence_ratio"], 0.5)

    def test_zero_sampled_frames_yields_none_ratios_not_zero(self):
        """Nessun frame costruito con successo: mai un falso 0% di
        coverage, il ratio resta esplicitamente None (dato insufficiente,
        non 'tutto mancante')."""
        coverage = compute_feature_coverage(
            expected_features=["a"], column_presence_counts={}, sampled_frames=0, requested_fixtures=5
        )
        self.assertIsNone(coverage["overall_column_presence_ratio"])
        self.assertEqual(coverage["per_feature"][0]["column_presence_ratio"], None)
        self.assertEqual(coverage["fully_missing_features"], [])  # sampled=0 -> non dichiarabile "missing"

    def test_zero_requested_fixtures_yields_none_fixture_ratio(self):
        coverage = compute_feature_coverage(
            expected_features=["a"], column_presence_counts={}, sampled_frames=0, requested_fixtures=0
        )
        self.assertIsNone(coverage["fixture_coverage_ratio"])


class TestEvaluateAlerts(unittest.TestCase):
    def test_no_alerts_when_everything_healthy(self):
        alerts = evaluate_alerts(
            prediction_volume_recent=50,
            calibration_drift={"ece_drift": 0.01, "recent_sample_size": 100, "baseline_sample_size": 100},
            roi_rolling={"roi": 0.05, "bets": 100},
            feature_coverage={"overall_column_presence_ratio": 0.99, "fully_missing_features": []},
            recent_failed_jobs=0,
        )
        self.assertEqual(alerts, [])

    def test_low_prediction_volume_alert(self):
        alerts = evaluate_alerts(prediction_volume_recent=0, thresholds=MonitoringThresholds(min_prediction_volume_recent=1))
        codes = [a.code for a in alerts]
        self.assertIn("low_prediction_volume", codes)
        alert = next(a for a in alerts if a.code == "low_prediction_volume")
        self.assertEqual(alert.severity, WARNING)

    def test_prediction_volume_check_disabled_with_none_threshold(self):
        alerts = evaluate_alerts(
            prediction_volume_recent=0, thresholds=MonitoringThresholds(min_prediction_volume_recent=None)
        )
        self.assertEqual([a.code for a in alerts if a.code == "low_prediction_volume"], [])

    def test_calibration_drift_alert_only_with_enough_samples(self):
        thresholds = MonitoringThresholds(max_calibration_ece_drift=0.10, min_samples_for_calibration=20)

        # Campione insufficiente: nessun alert anche se il drift supera la soglia.
        alerts_small_sample = evaluate_alerts(
            prediction_volume_recent=10,
            calibration_drift={"ece_drift": 0.50, "recent_sample_size": 5, "baseline_sample_size": 5},
            thresholds=thresholds,
        )
        self.assertEqual([a.code for a in alerts_small_sample if a.code == "calibration_drift"], [])

        # Campione sufficiente e drift oltre soglia: alert CRITICAL.
        alerts_enough_sample = evaluate_alerts(
            prediction_volume_recent=10,
            calibration_drift={"ece_drift": 0.50, "recent_sample_size": 30, "baseline_sample_size": 30},
            thresholds=thresholds,
        )
        alert = next(a for a in alerts_enough_sample if a.code == "calibration_drift")
        self.assertEqual(alert.severity, CRITICAL)
        self.assertAlmostEqual(alert.observed, 0.50)

    def test_calibration_improvement_never_alerts(self):
        """Un delta NEGATIVO (miglioramento) non deve mai scatenare
        l'alert, indipendentemente dalla soglia."""
        alerts = evaluate_alerts(
            prediction_volume_recent=10,
            calibration_drift={"ece_drift": -0.50, "recent_sample_size": 30, "baseline_sample_size": 30},
            thresholds=MonitoringThresholds(max_calibration_ece_drift=0.10, min_samples_for_calibration=20),
        )
        self.assertEqual([a.code for a in alerts if a.code == "calibration_drift"], [])

    def test_roi_rolling_alert_is_info_and_disabled_by_default(self):
        # Default: min_roi_rolling=None -> mai un alert, anche con ROI molto negativo.
        alerts_default = evaluate_alerts(
            prediction_volume_recent=10, roi_rolling={"roi": -0.9, "bets": 100}, thresholds=MonitoringThresholds()
        )
        self.assertEqual([a.code for a in alerts_default if a.code == "roi_rolling_below_threshold"], [])

        # Soglia attivata esplicitamente: alert INFO (mai bloccante).
        thresholds = MonitoringThresholds(min_roi_rolling=0.0, min_bets_for_roi_alert=10)
        alerts_enabled = evaluate_alerts(
            prediction_volume_recent=10, roi_rolling={"roi": -0.1, "bets": 50}, thresholds=thresholds
        )
        alert = next(a for a in alerts_enabled if a.code == "roi_rolling_below_threshold")
        self.assertEqual(alert.severity, INFO)

    def test_roi_alert_skipped_when_too_few_bets(self):
        thresholds = MonitoringThresholds(min_roi_rolling=0.0, min_bets_for_roi_alert=50)
        alerts = evaluate_alerts(prediction_volume_recent=10, roi_rolling={"roi": -0.5, "bets": 3}, thresholds=thresholds)
        self.assertEqual([a.code for a in alerts if a.code == "roi_rolling_below_threshold"], [])

    def test_low_feature_coverage_alert(self):
        alerts = evaluate_alerts(
            prediction_volume_recent=10,
            feature_coverage={"overall_column_presence_ratio": 0.5, "fully_missing_features": []},
            thresholds=MonitoringThresholds(min_feature_coverage_ratio=0.9),
        )
        alert = next(a for a in alerts if a.code == "low_feature_coverage")
        self.assertEqual(alert.severity, WARNING)

    def test_fully_missing_features_alert(self):
        alerts = evaluate_alerts(
            prediction_volume_recent=10,
            feature_coverage={"overall_column_presence_ratio": 1.0, "fully_missing_features": ["feat_x", "feat_y"]},
            thresholds=MonitoringThresholds(min_feature_coverage_ratio=0.9),
        )
        alert = next(a for a in alerts if a.code == "feature_fully_missing")
        self.assertEqual(alert.severity, CRITICAL)
        self.assertEqual(alert.observed, 2.0)

    def test_recent_failed_jobs_alert(self):
        alerts = evaluate_alerts(
            prediction_volume_recent=10, recent_failed_jobs=3, thresholds=MonitoringThresholds(max_recent_failed_jobs=0)
        )
        alert = next(a for a in alerts if a.code == "recent_failed_jobs")
        self.assertEqual(alert.severity, CRITICAL)
        self.assertEqual(alert.observed, 3.0)

    def test_all_checks_disabled_yields_no_alerts_regardless_of_signals(self):
        """Tutte le soglie a `None`/permissive: nessun alert, qualunque sia
        il segnale osservato (disattivazione ESPLICITA, mai un controllo
        residuo nascosto) - `feature_fully_missing` e' annidato sotto
        `min_feature_coverage_ratio`, quindi e' disattivato insieme al
        resto del controllo feature coverage."""
        alerts = evaluate_alerts(
            prediction_volume_recent=0,
            calibration_drift={"ece_drift": 0.9, "recent_sample_size": 1000, "baseline_sample_size": 1000},
            roi_rolling={"roi": -0.99, "bets": 1000},
            feature_coverage={"overall_column_presence_ratio": 0.0, "fully_missing_features": ["x"]},
            recent_failed_jobs=999,
            thresholds=PERMISSIVE_THRESHOLDS,
        )
        self.assertEqual(alerts, [])


if __name__ == "__main__":
    unittest.main()


