import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ml.markets.corners.corners_market import (
    DEFAULT_LINES,
    CornersBenchmarkReport,
    CornersExpert,
    CornersLineTrainResult,
    _feature_columns_for,
    _line_independent_oof,
    _line_specific_odds_features,
    build_corners_frame_from_records,
    build_corners_prediction_row,
    compute_monotonicity_report,
    label_corners_over,
    run_corners_benchmark,
    run_corners_benchmark_from_db,
    train_corners_all_lines,
    train_corners_line,
)
from src.ml.validation.temporal_split import expanding_window_splits


def _make_match(fixture_id: int, date: datetime, home_rating: float, away_rating: float, rng: np.random.RandomState) -> dict:
    home_corner_lambda = max(1.0, 5.5 + 1.5 * home_rating)
    away_corner_lambda = max(1.0, 4.5 + 1.5 * away_rating)
    home_corners = int(rng.poisson(home_corner_lambda))
    away_corners = int(rng.poisson(away_corner_lambda))

    return {
        "id_fixture": fixture_id,
        "season": 2025,
        "status": "FT",
        "date_match": date.isoformat(),
        "current_league": 39,
        "id_team_home": 100,
        "id_team_away": 200,
        "mean_statistics": [
            {"id_team": 100, "Corner Kicks": home_corner_lambda, "Shots on Goal": 5.0},
            {"id_team": 200, "Corner Kicks": away_corner_lambda, "Shots on Goal": 4.0},
        ],
        "statistics": [
            {"statistics_team_id": 100, "score_ft": 1, "corners": home_corners},
            {"statistics_team_id": 200, "score_ft": 1, "corners": away_corners},
        ],
        # Quote REALISTICHE multi-linea (2026-09-12): il provider mette
        # TUTTE le linee quotate in un unico bucket piatto 'corners' (a
        # differenza degli Under/Over gol, gia' separati per soglia) - qui
        # si simula lo stesso formato reale confermato in
        # filter_market_service_test.py ('Over 2.5_bookA'), con piu' linee
        # e quote diverse per linea (necessario per esercitare
        # `_line_specific_odds_features`/`_feature_columns_for`, che PRIMA
        # avrebbero pooled indiscriminatamente linee diverse insieme).
        "odds": [
            {
                "corners": {
                    "Over 8.5_bookA": 1.55,
                    "Under 8.5_bookA": 2.35,
                    "Over 8.5_bookB": 1.60,
                    "Under 8.5_bookB": 2.30,
                    "Over 9.5_bookA": 1.95,
                    "Under 9.5_bookA": 1.85,
                    "Over 9.5_bookB": 1.90,
                    "Under 9.5_bookB": 1.90,
                    "Over 10.5_bookA": 2.50,
                    "Under 10.5_bookA": 1.55,
                    "Over 10.5_bookB": 2.45,
                    "Under 10.5_bookB": 1.58,
                    "Over 11.5_bookA": 3.40,
                    "Under 11.5_bookA": 1.28,
                    "Over 11.5_bookB": 3.30,
                    "Under 11.5_bookB": 1.30,
                }
            }
        ],
    }


def _synthetic_matches(n: int = 260, seed: int = 21) -> list[dict]:
    rng = np.random.RandomState(seed)
    base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    matches = []
    for i in range(n):
        home_rating = float(rng.normal(scale=0.9))
        away_rating = float(rng.normal(scale=0.9))
        matches.append(_make_match(1000 + i, base_date + timedelta(days=i), home_rating, away_rating, rng))
    return matches


class TestLabelCornersOver(unittest.TestCase):
    def test_scalar_input_returns_scalar(self):
        self.assertEqual(label_corners_over(10, line=9.5), 1)
        self.assertEqual(label_corners_over(9, line=9.5), 0)

    def test_array_input_returns_array(self):
        result = label_corners_over(np.array([8, 9, 10, 11, 12]), line=9.5)
        np.testing.assert_array_equal(result, [0, 0, 1, 1, 1])

    def test_line_is_a_free_parameter_not_hardcoded(self):
        totals = np.array([8, 9, 10, 11, 12])
        low_line = label_corners_over(totals, line=8.5)
        high_line = label_corners_over(totals, line=11.5)
        # linea piu' bassa -> piu' Over (1) rispetto a linea piu' alta.
        self.assertGreaterEqual(int(low_line.sum()), int(high_line.sum()))


class TestBuildCornersFrameFromRecords(unittest.TestCase):
    def test_target_matches_real_total_corners_for_every_line(self):
        matches = _synthetic_matches(n=60)
        frame = build_corners_frame_from_records(matches)

        self.assertFalse(frame.empty)
        for _, row in frame.iterrows():
            match = next(m for m in matches if m["id_fixture"] == row["id_fixture"])
            stats = {s["statistics_team_id"]: s["corners"] for s in match["statistics"]}
            total = stats[100] + stats[200]
            self.assertEqual(int(row["total_corners"]), total)
            for line in DEFAULT_LINES:
                label = f"y_line_{str(line).replace('.', '_')}"
                self.assertEqual(int(row[label]), int(total > line))

    def test_dedicated_corner_features_present_and_consistent(self):
        matches = _synthetic_matches(n=40)
        frame = build_corners_frame_from_records(matches)

        for col in [
            "corner_mean_home_dedicated",
            "corner_mean_away_dedicated",
            "corner_mean_total_dedicated",
            "corner_mean_diff_dedicated",
        ]:
            self.assertIn(col, frame.columns)

        np.testing.assert_allclose(
            frame["corner_mean_total_dedicated"], frame["corner_mean_home_dedicated"] + frame["corner_mean_away_dedicated"]
        )
        np.testing.assert_allclose(
            frame["corner_mean_diff_dedicated"], frame["corner_mean_home_dedicated"] - frame["corner_mean_away_dedicated"]
        )

    def test_configurable_lines_produce_different_target_columns(self):
        matches = _synthetic_matches(n=40)
        custom_lines = (7.5, 13.5)
        frame = build_corners_frame_from_records(matches, lines=custom_lines)
        self.assertIn("y_line_7_5", frame.columns)
        self.assertIn("y_line_13_5", frame.columns)
        self.assertNotIn("y_line_9_5", frame.columns)

    def test_rows_ordered_by_prediction_at(self):
        matches = _synthetic_matches(n=40)
        frame = build_corners_frame_from_records(matches)
        ordered_dates = pd.to_datetime(frame["prediction_at"], utc=True)
        self.assertTrue((ordered_dates.diff().dropna() >= pd.Timedelta(0)).all())

    def test_line_specific_odds_columns_present_and_differ_per_line(self):
        # 2026-09-12: le quote 'corners' arrivano mischiate su piu' linee
        # nello stesso bucket - il frame deve esporre colonne SEPARATE per
        # ciascuna linea configurata, con valori diversi (non un pool unico
        # ripetuto identico su tutte le linee).
        matches = _synthetic_matches(n=40)
        frame = build_corners_frame_from_records(matches)

        for line in DEFAULT_LINES:
            label = str(line).replace(".", "_")
            for key in ("odds_mean", "odds_min", "odds_max"):
                self.assertIn(f"{key}_line_{label}", frame.columns)

        row = frame.iloc[0]
        self.assertNotAlmostEqual(row["odds_mean_line_8_5"], row["odds_mean_line_11_5"])


class TestBuildCornersPredictionRow(unittest.TestCase):
    """2026-09-13: builder feature per UNA fixture live/futura (serving),
    non solo in batch per il training - riusa la stessa logica pura di
    `build_corners_frame_from_records` (nessuna feature dedicata Corners
    dipende dallo storico, a differenza dell'arbitro per Cards)."""

    def test_matches_batch_builder_for_same_single_match(self):
        matches = _synthetic_matches(n=5)
        match = matches[0]

        row = build_corners_prediction_row(match)
        frame = build_corners_frame_from_records([match])

        self.assertIsNotNone(row)
        self.assertFalse(frame.empty)
        frame_row = frame.iloc[0]
        for col in ["corner_mean_home_dedicated", "corner_mean_away_dedicated", "odds_mean_line_8_5", "odds_mean"]:
            self.assertAlmostEqual(row[col], frame_row[col])

    def test_no_target_columns_present(self):
        matches = _synthetic_matches(n=5)
        row = build_corners_prediction_row(matches[0])
        self.assertNotIn("total_corners", row)
        for line in DEFAULT_LINES:
            self.assertNotIn(f"y_line_{str(line).replace('.', '_')}", row)

    def test_missing_odds_returns_none(self):
        match = {"odds": [], "mean_statistics": [], "id_fixture": 1, "season": 2025, "current_league": 39, "date_match": "2026-01-01T00:00:00+00:00"}
        self.assertIsNone(build_corners_prediction_row(match))


class TestLineSpecificOddsFeatures(unittest.TestCase):
    def test_pools_only_the_requested_line(self):
        match = {
            "odds": [
                {
                    "corners": {
                        "Over 8.5_bookA": 1.55,
                        "Under 8.5_bookA": 2.35,
                        "Over 11.5_bookA": 3.40,
                        "Under 11.5_bookA": 1.28,
                    }
                }
            ]
        }
        features = _line_specific_odds_features(match, odds_market="corners", lines=(8.5, 11.5))

        self.assertAlmostEqual(features["odds_mean_line_8_5"], (1.55 + 2.35) / 2)
        self.assertAlmostEqual(features["odds_mean_line_11_5"], (3.40 + 1.28) / 2)
        self.assertEqual(features["odds_count_line_8_5"], 2.0)
        self.assertEqual(features["odds_count_line_11_5"], 2.0)

    def test_missing_odds_returns_empty_dict(self):
        self.assertEqual(_line_specific_odds_features({"odds": []}, odds_market="corners", lines=(8.5,)), {})
        self.assertEqual(_line_specific_odds_features({}, odds_market="corners", lines=(8.5,)), {})


class TestFeatureColumnsForLineScoping(unittest.TestCase):
    def test_active_line_odds_kept_other_lines_and_pooled_excluded(self):
        matches = _synthetic_matches(n=40)
        frame = build_corners_frame_from_records(matches)

        columns = _feature_columns_for(frame, DEFAULT_LINES, active_line=8.5, use_line_specific_odds=True)

        self.assertIn("odds_mean_line_8_5", columns)
        self.assertNotIn("odds_mean_line_11_5", columns)
        # Le vecchie quote "pooled" (mischiano tutte le linee) restano
        # escluse quando si usano quelle per-linea.
        self.assertNotIn("odds_mean", columns)
        # Feature generiche (mean_statistics/dedicate) restano sempre incluse.
        self.assertIn("corner_mean_total_dedicated", columns)

    def test_legacy_pooled_mode_excludes_all_line_specific_columns(self):
        matches = _synthetic_matches(n=40)
        frame = build_corners_frame_from_records(matches)

        columns = _feature_columns_for(frame, DEFAULT_LINES, active_line=8.5, use_line_specific_odds=False)

        self.assertIn("odds_mean", columns)
        self.assertNotIn("odds_mean_line_8_5", columns)
        self.assertNotIn("odds_mean_line_11_5", columns)

    def test_no_column_of_another_line_ever_leaks_into_the_active_line(self):
        """Regressione (2026-09-13): le feature quote per esito sono emesse
        da DUE percorsi con suffissi diversi - '..._over_9_5_line_9_5' dal
        builder per linea e '..._over_9_5' dal bucket intero. La seconda
        forma sfuggiva all'esclusione, quindi il modello della linea 8.5
        riceveva anche le quote di 9.5/10.5/11.5: esattamente il leak che le
        quote per-linea erano nate per eliminare."""
        matches = _synthetic_matches(n=40)
        frame = build_corners_frame_from_records(matches)

        for active in DEFAULT_LINES:
            columns = _feature_columns_for(frame, DEFAULT_LINES, active_line=active, use_line_specific_odds=True)
            other_labels = [
                str(line).replace(".", "_") for line in DEFAULT_LINES if line != active
            ]
            leaked = [
                col
                for col in columns
                if col.startswith(("odds_", "implied_prob_", "prob_norm_", "overround"))
                and any(col.endswith(f"_{label}") for label in other_labels)
            ]
            self.assertEqual(leaked, [], f"linea attiva {active}: colonne di altre linee {leaked}")

    def test_per_outcome_odds_split_over_and_under_within_the_active_line(self):
        """Filtrare per linea non basta: dentro la linea restano i due lati
        ('over 8.5' e 'under 8.5' contengono entrambi "8.5")."""
        matches = _synthetic_matches(n=40)
        frame = build_corners_frame_from_records(matches)

        columns = _feature_columns_for(frame, DEFAULT_LINES, active_line=8.5, use_line_specific_odds=True)
        over = [c for c in columns if c.startswith("odds_mean_over_8_5")]
        under = [c for c in columns if c.startswith("odds_mean_under_8_5")]

        self.assertTrue(over, "manca la media quote del lato Over")
        self.assertTrue(under, "manca la media quote del lato Under")
        row = frame.iloc[0]
        self.assertNotAlmostEqual(row[over[0]], row[under[0]])


class TestComputeMonotonicityReport(unittest.TestCase):
    """2026-09-12: "le linee di corners seguono lo stesso principio di
    under/over gol? Over X implica Over di una linea piu' bassa" - stesso
    principio gia' in produzione per i gol (MARKET-04,
    `enforce_monotonic_over_probabilities`), qui applicato a Corners con un
    RandomForest fisso INDIPENDENTE dal champion per linea (diagnostico,
    veloce - non richiama mai `_select_champion_via_model_search`)."""

    def test_reports_violations_and_stays_fast_without_grid_search(self):
        matches = _synthetic_matches(n=260)
        frame = build_corners_frame_from_records(matches)
        cv_splits = expanding_window_splits(
            frame=frame, time_col="prediction_at", n_splits=5,
            min_train_size=max(30, int(len(frame) * 0.45)), min_valid_size=max(10, int(len(frame) * 0.1)),
        )

        report = compute_monotonicity_report(frame, lines=DEFAULT_LINES, cv_splits=cv_splits)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["lines"], list(DEFAULT_LINES))
        self.assertGreater(report["n_rows_evaluated"], 0)
        self.assertGreaterEqual(report["violations_before_projection"], 0)
        self.assertGreaterEqual(report["violations_pct"], 0.0)
        self.assertLessEqual(report["violations_pct"], 1.0)

        # "quindi le metriche dopo aver applicato questo fix quali sono?"
        # (2026-09-12): confronto prima/dopo per ogni linea, stesse chiavi
        # di compute_probability_metrics + accuracy/f1/selection_score.
        metrics_by_line = report["metrics_by_line"]
        for line in DEFAULT_LINES:
            label = f"line_{str(float(line)).replace('.', '_')}"
            self.assertIn(label, metrics_by_line)
            for phase in ("before_projection", "after_projection"):
                entry = metrics_by_line[label][phase]
                self.assertIn("log_loss", entry)
                self.assertIn("selection_score", entry)
                self.assertIn("accuracy", entry)
        # La linea piu' bassa e' l'ancora della proiezione cumulativa:
        # non puo' cambiare (nessuna linea sotto con cui fare il minimo).
        lowest_label = f"line_{str(float(min(DEFAULT_LINES))).replace('.', '_')}"
        self.assertEqual(
            metrics_by_line[lowest_label]["before_projection"]["log_loss"],
            metrics_by_line[lowest_label]["after_projection"]["log_loss"],
        )

    def test_single_line_is_not_applicable(self):
        matches = _synthetic_matches(n=60)
        frame = build_corners_frame_from_records(matches, lines=(9.5,))
        report = compute_monotonicity_report(frame, lines=(9.5,), cv_splits=[([0], [1])])
        self.assertEqual(report["status"], "not_applicable")

    def test_line_independent_oof_returns_frame_aligned_arrays_with_nan_outside_folds(self):
        matches = _synthetic_matches(n=260)
        frame = build_corners_frame_from_records(matches)
        cv_splits = expanding_window_splits(
            frame=frame, time_col="prediction_at", n_splits=5,
            min_train_size=max(30, int(len(frame) * 0.45)), min_valid_size=max(10, int(len(frame) * 0.1)),
        )

        oof = _line_independent_oof(frame, DEFAULT_LINES, cv_splits)

        for line in DEFAULT_LINES:
            label = f"line_{str(float(line)).replace('.', '_')}"
            self.assertIn(label, oof)
            self.assertEqual(len(oof[label]), len(frame))
            valid = ~np.isnan(oof[label])
            self.assertTrue(valid.any())
            self.assertTrue(np.all((oof[label][valid] >= 0.0) & (oof[label][valid] <= 1.0)))


class TestTrainCornersLine(unittest.TestCase):
    def test_produces_calibration_result_with_pre_post_metrics(self):
        matches = _synthetic_matches(n=260)
        frame = build_corners_frame_from_records(matches)
        from src.ml.validation.temporal_split import expanding_window_splits

        cv_splits = expanding_window_splits(frame=frame, time_col="prediction_at", n_splits=4, min_train_size=120, min_valid_size=30)

        result = train_corners_line(frame=frame, line=9.5, cv_splits=cv_splits)

        self.assertIsInstance(result, CornersLineTrainResult)
        self.assertEqual(result.line, 9.5)
        self.assertIn("log_loss", result.calibration.pre_metrics)
        self.assertIn("log_loss", result.calibration.post_metrics)
        self.assertTrue(hasattr(result.calibration.calibrator, "predict_proba"))

    def test_raises_for_line_not_in_dataset(self):
        matches = _synthetic_matches(n=60)
        frame = build_corners_frame_from_records(matches, lines=(9.5,))
        with self.assertRaises(ValueError):
            train_corners_line(frame=frame, line=20.5, cv_splits=[([0], [1])])


class TestTrainCornersAllLines(unittest.TestCase):
    def test_report_has_metrics_per_line(self):
        matches = _synthetic_matches(n=260)
        frame = build_corners_frame_from_records(matches)

        report = train_corners_all_lines(frame)

        self.assertIsInstance(report, CornersBenchmarkReport)
        self.assertGreater(len(report.results), 0)
        summary = report.metrics_summary()
        for label, entry in summary.items():
            self.assertIn("pre_metrics", entry)
            self.assertIn("post_metrics", entry)
            self.assertIn("line", entry)
            # "Tutte le metriche possibili" (2026-09-12): confusion matrix,
            # precision/recall/F1, ROC/PR e soglia ottimale, per pre e post
            # calibrazione, sugli stessi array OOF gia' calcolati.
            for report_key in ("classification_report_pre", "classification_report_post"):
                self.assertIn(report_key, entry)
                classification = entry[report_key]
                self.assertIn(classification["status"], {"ok", "single_class"})
                if classification["status"] == "ok":
                    self.assertIn("confusion_matrix", classification)
                    self.assertIn("roc", classification)
                    self.assertIn("pr_curve", classification)
                    self.assertIn("optimal_threshold", classification)

    def test_raises_when_dataset_too_small_for_temporal_cv(self):
        matches = _synthetic_matches(n=10)
        frame = build_corners_frame_from_records(matches)
        with self.assertRaises(ValueError):
            train_corners_all_lines(frame)


class TestCornersExpert(unittest.TestCase):
    def _fitted_expert(self, line=9.5):
        matches = _synthetic_matches(n=150)
        frame = build_corners_frame_from_records(matches)
        feature_cols = [c for c in frame.columns if not c.startswith("y_line_") and c not in {"id_fixture", "season", "league", "market", "prediction_at", "total_corners"}]
        X = frame[feature_cols]
        y = frame[f"y_line_{str(line).replace('.', '_')}"].astype(int)
        model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
        expert = CornersExpert.from_estimator(line=line, estimator=model, feature_names=feature_cols)
        return expert, X

    def test_predict_proba_dict_sums_to_one(self):
        expert, X = self._fitted_expert()
        result = expert.predict_proba_dict(X)
        np.testing.assert_allclose(result["over"] + result["under"], np.ones(len(X)))

    def test_predict_matches_threshold_on_predict_proba(self):
        expert, X = self._fitted_expert()
        proba = expert.predict_proba(X)
        prediction = expert.predict(X)
        np.testing.assert_array_equal(prediction, (proba >= 0.5).astype(int))

    def test_load_production_raises_lookup_error_when_missing(self):
        registry = mock.Mock()
        registry.get_production.return_value = None
        with self.assertRaises(LookupError):
            CornersExpert.load_production(line=9.5, registry=registry)
        registry.get_production.assert_called_once_with(market="corners_line_9_5")

    def test_load_latest_raises_lookup_error_when_missing(self):
        registry = mock.Mock()
        registry.get_latest.return_value = None
        with self.assertRaises(LookupError):
            CornersExpert.load_latest(line=10.5, registry=registry)
        registry.get_latest.assert_called_once_with(market="corners_line_10_5")


class TestRunCornersBenchmark(unittest.TestCase):
    def test_end_to_end_benchmark_without_saving(self):
        matches = _synthetic_matches(n=260)
        result = run_corners_benchmark(matches=matches, save_model=False)

        self.assertEqual(result.market, "corners")
        self.assertEqual(result.status, "benchmarked")
        self.assertGreater(result.rows, 0)
        self.assertGreater(len(result.lines_trained), 0)
        self.assertEqual(result.details["runs"], {})  # save_model=False -> nessuna registrazione

    def test_empty_matches_are_skipped_without_raising(self):
        result = run_corners_benchmark(matches=[], save_model=False)
        self.assertEqual(result.status, "skipped_no_data")
        self.assertEqual(result.rows, 0)

    def test_uses_provided_frame_directly_ignoring_matches(self):
        # 2026-09-12: `frame=` permette di eseguire il training in un
        # ambiente senza accesso DB (CSV esportato altrove - vedi
        # train_corners_cards_from_export.py). Qui verifichiamo che, quando
        # fornito, il frame venga usato DIRETTAMENTE - `matches` (qui
        # deliberatamente diverso/vuoto) viene ignorato, non ricostruito.
        matches = _synthetic_matches(n=260)
        frame = build_corners_frame_from_records(matches)

        result = run_corners_benchmark(matches=[], frame=frame, save_model=False)

        self.assertEqual(result.status, "benchmarked")
        self.assertEqual(result.rows, len(frame))
        self.assertGreater(len(result.lines_trained), 0)


class TestRunCornersBenchmarkFromDb(unittest.TestCase):
    def test_empty_db_result_is_skipped_without_raising(self):
        with mock.patch("src.ml.markets.corners.corners_market.MatchRepository") as repo_cls:
            repo_cls.return_value.search_filter.return_value = []
            result = run_corners_benchmark_from_db(save_model=False)

        self.assertEqual(result.status, "skipped_no_data")


if __name__ == "__main__":
    unittest.main()
