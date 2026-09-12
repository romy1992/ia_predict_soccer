import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ml.markets.cards.cards_market import (
    DEFAULT_LINES,
    DEFAULT_REFEREE_PRIOR_CARDS,
    REFEREE_SHRINKAGE_K,
    CardsBenchmarkReport,
    CardsExpert,
    CardsLineTrainResult,
    _shrink_toward_baseline,
    build_cards_frame_from_records,
    build_referee_features_dataset,
    label_cards_over,
    run_cards_benchmark,
    run_cards_benchmark_from_db,
    train_cards_all_lines,
    train_cards_line,
)


def _make_match(
    fixture_id: int,
    date: datetime,
    referee: str,
    home_cards_yellow: int,
    away_cards_yellow: int,
    home_cards_red: int = 0,
    away_cards_red: int = 0,
    league: int = 39,
) -> dict:
    return {
        "id_fixture": fixture_id,
        "season": 2025,
        "status": "FT",
        "date_match": date.isoformat(),
        "current_league": league,
        "referee": referee,
        "id_team_home": 100,
        "id_team_away": 200,
        "mean_statistics": [
            {"id_team": 100, "Yellow Cards": 2.0, "Fouls": 12.0},
            {"id_team": 200, "Yellow Cards": 1.5, "Fouls": 10.0},
        ],
        "statistics": [
            {"statistics_team_id": 100, "score_ft": 1, "yellow_cards": home_cards_yellow, "red_cards": home_cards_red},
            {"statistics_team_id": 200, "score_ft": 1, "yellow_cards": away_cards_yellow, "red_cards": away_cards_red},
        ],
        "odds": [{"cards": {"over_bookA": 1.9, "under_bookA": 1.9, "over_bookB": 1.95, "under_bookB": 1.85}}],
    }


def _synthetic_matches(n: int = 240, seed: int = 31) -> list[dict]:
    rng = np.random.RandomState(seed)
    base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    referees = ["Rossi", "Bianchi", "Verdi"]
    matches = []
    for i in range(n):
        referee = referees[i % len(referees)]
        home_yellow = int(rng.poisson(2.2))
        away_yellow = int(rng.poisson(2.0))
        home_red = int(rng.binomial(1, 0.05))
        away_red = int(rng.binomial(1, 0.05))
        matches.append(
            _make_match(1000 + i, base_date + timedelta(days=i), referee, home_yellow, away_yellow, home_red, away_red)
        )
    return matches


class TestLabelCardsOver(unittest.TestCase):
    def test_scalar_input_returns_scalar(self):
        self.assertEqual(label_cards_over(5, line=4.5), 1)
        self.assertEqual(label_cards_over(4, line=4.5), 0)

    def test_array_input_returns_array(self):
        result = label_cards_over(np.array([3, 4, 5, 6, 7]), line=4.5)
        np.testing.assert_array_equal(result, [0, 0, 1, 1, 1])

    def test_line_is_a_free_parameter_not_hardcoded(self):
        totals = np.array([3, 4, 5, 6, 7])
        low_line = label_cards_over(totals, line=3.5)
        high_line = label_cards_over(totals, line=6.5)
        self.assertGreaterEqual(int(low_line.sum()), int(high_line.sum()))


class TestBuildRefereeFeaturesDataset(unittest.TestCase):
    def test_no_leakage_prior_uses_only_past_matches_of_same_referee(self):
        # Un solo arbitro/una sola lega in questo scenario: la media di lega
        # point-in-time coincide sempre con la media grezza dell'arbitro,
        # quindi lo shrinkage verso il baseline e' un no-op numerico (stesso
        # valore con o senza raffinamento) - test invariato.
        base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        matches = [
            _make_match(1, base_date, "Rossi", home_cards_yellow=2, away_cards_yellow=1),  # totale 3
            _make_match(2, base_date + timedelta(days=1), "Rossi", home_cards_yellow=4, away_cards_yellow=2),  # totale 6
            _make_match(3, base_date + timedelta(days=2), "Rossi", home_cards_yellow=1, away_cards_yellow=1),  # totale 2
        ]
        frame = build_referee_features_dataset(matches)

        row0 = frame[frame["id_fixture"] == 1].iloc[0]
        self.assertEqual(int(row0["referee_has_history"]), 0)
        self.assertAlmostEqual(row0["referee_avg_cards_prior"], DEFAULT_REFEREE_PRIOR_CARDS)
        self.assertAlmostEqual(row0["referee_severity_index_prior"], 1.0)
        self.assertEqual(int(row0["referee_matches_officiated_prior"]), 0)

        row1 = frame[frame["id_fixture"] == 2].iloc[0]
        self.assertEqual(int(row1["referee_has_history"]), 1)
        self.assertAlmostEqual(row1["referee_avg_cards_prior"], 3.0)
        self.assertAlmostEqual(row1["referee_severity_index_prior"], 1.0)
        self.assertEqual(int(row1["referee_matches_officiated_prior"]), 1)

        row2 = frame[frame["id_fixture"] == 3].iloc[0]
        self.assertAlmostEqual(row2["referee_avg_cards_prior"], (3.0 + 6.0) / 2.0)
        self.assertEqual(int(row2["referee_matches_officiated_prior"]), 2)

    def test_different_referees_have_independent_histories(self):
        # Bianchi (media di lega piu' bassa, 0 cartellini) tira verso il
        # basso il baseline di lega usato per lo shrinkage di Rossi alla sua
        # 2a partita - stesso principio "meno storico = piu' peso al
        # baseline" gia' verificato in isolamento da
        # `TestShrinkTowardBaseline` sotto.
        base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        matches = [
            _make_match(1, base_date, "Rossi", home_cards_yellow=5, away_cards_yellow=5),  # totale 10
            _make_match(2, base_date + timedelta(days=1), "Bianchi", home_cards_yellow=0, away_cards_yellow=0),  # totale 0
            _make_match(3, base_date + timedelta(days=2), "Rossi", home_cards_yellow=1, away_cards_yellow=1),
        ]
        frame = build_referee_features_dataset(matches)

        row_bianchi = frame[frame["id_fixture"] == 2].iloc[0]
        self.assertEqual(int(row_bianchi["referee_has_history"]), 0)  # Bianchi non ha storico (Rossi non lo influenza)

        row_rossi_second = frame[frame["id_fixture"] == 3].iloc[0]
        self.assertEqual(int(row_rossi_second["referee_matches_officiated_prior"]), 1)  # solo la partita 1 di Rossi
        # Baseline di lega al momento della 3a partita: media di (10, 0) = 5.0.
        # Shrink(raw=10.0, n=1, baseline=5.0, k=REFEREE_SHRINKAGE_K).
        expected = _shrink_toward_baseline(raw_average=10.0, matches_officiated=1, baseline=5.0, k=REFEREE_SHRINKAGE_K)
        self.assertAlmostEqual(row_rossi_second["referee_avg_cards_prior"], expected)
        self.assertLess(row_rossi_second["referee_avg_cards_prior"], 10.0)  # tirato verso il basso dal baseline
        self.assertAlmostEqual(row_rossi_second["referee_severity_index_prior"], expected / 5.0)

    def test_missing_referee_always_uses_default_prior_never_league_baseline(self):
        # Un arbitro sconosciuto resta SEMPRE al prior neutro fisso, anche
        # quando la lega nel frattempo ha gia' accumulato storico (qui la
        # prima partita porta la media di lega a 10.0): nessuna identita'
        # su cui basare una stima "furba".
        base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        matches = [
            _make_match(1, base_date, "", home_cards_yellow=5, away_cards_yellow=5),
            _make_match(2, base_date + timedelta(days=1), "", home_cards_yellow=1, away_cards_yellow=1),
        ]
        frame = build_referee_features_dataset(matches)
        for _, row in frame.iterrows():
            self.assertEqual(int(row["referee_has_history"]), 0)
            self.assertAlmostEqual(row["referee_avg_cards_prior"], DEFAULT_REFEREE_PRIOR_CARDS)
            self.assertAlmostEqual(row["referee_severity_index_prior"], 1.0)

    def test_new_referee_first_match_uses_league_baseline_when_available(self):
        # Un arbitro MAI visto prima (ma con nome noto) alla sua primissima
        # partita eredita il baseline di lega gia' accumulato da ALTRI
        # arbitri - a differenza del caso "arbitro sconosciuto" sopra.
        base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        matches = [
            _make_match(1, base_date, "Rossi", home_cards_yellow=5, away_cards_yellow=5),  # totale 10
            _make_match(2, base_date + timedelta(days=1), "Bianchi", home_cards_yellow=0, away_cards_yellow=0),
        ]
        frame = build_referee_features_dataset(matches)
        row_bianchi = frame[frame["id_fixture"] == 2].iloc[0]
        self.assertEqual(int(row_bianchi["referee_has_history"]), 0)
        self.assertAlmostEqual(row_bianchi["referee_avg_cards_prior"], 10.0)  # media di lega al momento (solo Rossi)
        self.assertAlmostEqual(row_bianchi["referee_severity_index_prior"], 1.0)

    def test_severity_index_reflects_league_normalization_across_leagues(self):
        # Due arbitri DIVERSI (per non contaminare lo storico globale di uno
        # stesso arbitro tra leghe), ciascuno con cartellini stabili nella
        # propria lega: il prior ASSOLUTO diverge enormemente (8.0 vs 1.0),
        # ma l'indice di severita' - la stessa metrica "sei nella media
        # della TUA lega?" - li giudica ENTRAMBI perfettamente nella media.
        base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        matches = [
            # Lega 39 "dura": Neri, sempre 8 cartellini totali.
            _make_match(1, base_date, "Neri", home_cards_yellow=4, away_cards_yellow=4, league=39),  # totale 8
            _make_match(2, base_date + timedelta(days=1), "Neri", home_cards_yellow=4, away_cards_yellow=4, league=39),
            _make_match(3, base_date + timedelta(days=2), "Neri", home_cards_yellow=4, away_cards_yellow=4, league=39),
            # Lega 78 "morbida": Verdi, sempre 1 cartellino totale.
            _make_match(4, base_date, "Verdi", home_cards_yellow=1, away_cards_yellow=0, league=78),  # totale 1
            _make_match(5, base_date + timedelta(days=1), "Verdi", home_cards_yellow=1, away_cards_yellow=0, league=78),
            _make_match(6, base_date + timedelta(days=2), "Verdi", home_cards_yellow=1, away_cards_yellow=0, league=78),
        ]
        frame = build_referee_features_dataset(matches)

        # 3a partita di ciascun arbitro: raw average == baseline di lega in
        # ENTRAMBI i casi (stesso valore ripetuto, nessuna contaminazione
        # incrociata tra le due leghe) -> lo shrinkage e' un no-op.
        row_league39 = frame[frame["id_fixture"] == 3].iloc[0]
        row_league78 = frame[frame["id_fixture"] == 6].iloc[0]

        self.assertAlmostEqual(row_league39["referee_avg_cards_prior"], 8.0)
        self.assertAlmostEqual(row_league78["referee_avg_cards_prior"], 1.0)
        # Il VALORE ASSOLUTO del prior differisce enormemente...
        self.assertNotAlmostEqual(row_league39["referee_avg_cards_prior"], row_league78["referee_avg_cards_prior"])
        # ...ma l'indice di severita' normalizzato dice "nella media" per
        # ENTRAMBI - e' esattamente cio' che la normalizzazione per lega
        # deve garantire (senza, un modello guarderebbe solo l'8.0 assoluto
        # di Neri e lo giudicherebbe "severo" confrontandolo con arbitri di
        # altre leghe, un confronto non significativo).
        self.assertAlmostEqual(row_league39["referee_severity_index_prior"], 1.0)
        self.assertAlmostEqual(row_league78["referee_severity_index_prior"], 1.0)

    def test_empty_matches_returns_empty_frame_with_expected_columns(self):
        frame = build_referee_features_dataset([])
        self.assertTrue(frame.empty)
        for col in [
            "id_fixture",
            "referee_avg_cards_prior",
            "referee_severity_index_prior",
            "referee_matches_officiated_prior",
            "referee_has_history",
        ]:
            self.assertIn(col, frame.columns)


class TestShrinkTowardBaseline(unittest.TestCase):
    def test_no_history_edge_case_is_not_called_by_the_dataset_builder(self):
        # Documenta il contratto: con matches_officiated=0 la formula
        # degenera al baseline puro (0*raw + k*baseline)/(0+k) = baseline -
        # `build_referee_features_dataset` non chiama comunque mai questa
        # funzione per n=0 (usa direttamente il baseline), ma il
        # comportamento resta coerente se richiamata cosi'.
        result = _shrink_toward_baseline(raw_average=99.0, matches_officiated=0, baseline=4.0, k=REFEREE_SHRINKAGE_K)
        self.assertAlmostEqual(result, 4.0)

    def test_more_history_weighs_raw_average_more_than_baseline(self):
        few = _shrink_toward_baseline(raw_average=10.0, matches_officiated=1, baseline=4.0, k=REFEREE_SHRINKAGE_K)
        many = _shrink_toward_baseline(raw_average=10.0, matches_officiated=100, baseline=4.0, k=REFEREE_SHRINKAGE_K)
        self.assertGreater(many, few)
        self.assertLess(many, 10.0)
        self.assertAlmostEqual(many, 10.0, delta=1.0)  # con 100 partite il baseline conta quasi zero

    def test_raw_equal_to_baseline_is_a_no_op(self):
        result = _shrink_toward_baseline(raw_average=5.0, matches_officiated=3, baseline=5.0, k=REFEREE_SHRINKAGE_K)
        self.assertAlmostEqual(result, 5.0)


class TestBuildCardsFrameFromRecords(unittest.TestCase):
    def test_target_matches_real_total_cards_for_every_line(self):
        matches = _synthetic_matches(n=60)
        frame = build_cards_frame_from_records(matches)

        self.assertFalse(frame.empty)
        for _, row in frame.iterrows():
            match = next(m for m in matches if m["id_fixture"] == row["id_fixture"])
            stats = {s["statistics_team_id"]: s["yellow_cards"] + s["red_cards"] for s in match["statistics"]}
            total = stats[100] + stats[200]
            self.assertEqual(int(row["total_cards"]), total)
            for line in DEFAULT_LINES:
                label = f"y_line_{str(line).replace('.', '_')}"
                self.assertEqual(int(row[label]), int(total > line))

    def test_referee_features_present_and_within_expected_range(self):
        matches = _synthetic_matches(n=60)
        frame = build_cards_frame_from_records(matches)

        for col in [
            "referee_avg_cards_prior",
            "referee_severity_index_prior",
            "referee_matches_officiated_prior",
            "referee_has_history",
        ]:
            self.assertIn(col, frame.columns)
        self.assertTrue((frame["referee_avg_cards_prior"] >= 0.0).all())
        self.assertTrue((frame["referee_severity_index_prior"] >= 0.0).all())
        self.assertTrue(frame["referee_has_history"].isin([0, 1]).all())

    def test_configurable_lines_produce_different_target_columns(self):
        matches = _synthetic_matches(n=40)
        custom_lines = (2.5, 8.5)
        frame = build_cards_frame_from_records(matches, lines=custom_lines)
        self.assertIn("y_line_2_5", frame.columns)
        self.assertIn("y_line_8_5", frame.columns)
        self.assertNotIn("y_line_4_5", frame.columns)

    def test_rows_ordered_by_prediction_at(self):
        matches = _synthetic_matches(n=40)
        frame = build_cards_frame_from_records(matches)
        ordered_dates = pd.to_datetime(frame["prediction_at"], utc=True)
        self.assertTrue((ordered_dates.diff().dropna() >= pd.Timedelta(0)).all())


class TestTrainCardsLine(unittest.TestCase):
    def test_produces_calibration_result_with_pre_post_metrics(self):
        matches = _synthetic_matches(n=240)
        frame = build_cards_frame_from_records(matches)
        from src.ml.validation.temporal_split import expanding_window_splits

        cv_splits = expanding_window_splits(frame=frame, time_col="prediction_at", n_splits=4, min_train_size=110, min_valid_size=25)

        result = train_cards_line(frame=frame, line=4.5, cv_splits=cv_splits)

        self.assertIsInstance(result, CardsLineTrainResult)
        self.assertEqual(result.line, 4.5)
        self.assertIn("log_loss", result.calibration.pre_metrics)
        self.assertIn("log_loss", result.calibration.post_metrics)
        self.assertTrue(hasattr(result.calibration.calibrator, "predict_proba"))
        self.assertIn("referee_avg_cards_prior", result.feature_names)

    def test_raises_for_line_not_in_dataset(self):
        matches = _synthetic_matches(n=60)
        frame = build_cards_frame_from_records(matches, lines=(4.5,))
        with self.assertRaises(ValueError):
            train_cards_line(frame=frame, line=99.5, cv_splits=[([0], [1])])


class TestTrainCardsAllLines(unittest.TestCase):
    def test_report_has_metrics_per_line(self):
        matches = _synthetic_matches(n=240)
        frame = build_cards_frame_from_records(matches)

        report = train_cards_all_lines(frame)

        self.assertIsInstance(report, CardsBenchmarkReport)
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
        frame = build_cards_frame_from_records(matches)
        with self.assertRaises(ValueError):
            train_cards_all_lines(frame)


class TestCardsExpert(unittest.TestCase):
    def _fitted_expert(self, line=4.5):
        matches = _synthetic_matches(n=150)
        frame = build_cards_frame_from_records(matches)
        feature_cols = [
            c
            for c in frame.columns
            if not c.startswith("y_line_") and c not in {"id_fixture", "season", "league", "market", "prediction_at", "total_cards"}
        ]
        X = frame[feature_cols]
        y = frame[f"y_line_{str(line).replace('.', '_')}"].astype(int)
        model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
        expert = CardsExpert.from_estimator(line=line, estimator=model, feature_names=feature_cols)
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
            CardsExpert.load_production(line=4.5, registry=registry)
        registry.get_production.assert_called_once_with(market="cards_line_4_5")

    def test_load_latest_raises_lookup_error_when_missing(self):
        registry = mock.Mock()
        registry.get_latest.return_value = None
        with self.assertRaises(LookupError):
            CardsExpert.load_latest(line=5.5, registry=registry)
        registry.get_latest.assert_called_once_with(market="cards_line_5_5")


class TestRunCardsBenchmark(unittest.TestCase):
    def test_end_to_end_benchmark_without_saving(self):
        matches = _synthetic_matches(n=240)
        result = run_cards_benchmark(matches=matches, save_model=False)

        self.assertEqual(result.market, "cards")
        self.assertEqual(result.status, "benchmarked")
        self.assertGreater(result.rows, 0)
        self.assertGreater(len(result.lines_trained), 0)
        self.assertEqual(result.details["runs"], {})  # save_model=False -> nessuna registrazione

    def test_empty_matches_are_skipped_without_raising(self):
        result = run_cards_benchmark(matches=[], save_model=False)
        self.assertEqual(result.status, "skipped_no_data")
        self.assertEqual(result.rows, 0)

    def test_uses_provided_frame_directly_ignoring_matches(self):
        # 2026-09-12: `frame=` permette di eseguire il training in un
        # ambiente senza accesso DB (CSV esportato altrove - vedi
        # train_corners_cards_from_export.py). Qui verifichiamo che, quando
        # fornito, il frame venga usato DIRETTAMENTE - `matches` (qui
        # deliberatamente diverso/vuoto) viene ignorato, non ricostruito.
        matches = _synthetic_matches(n=240)
        frame = build_cards_frame_from_records(matches)

        result = run_cards_benchmark(matches=[], frame=frame, save_model=False)

        self.assertEqual(result.status, "benchmarked")
        self.assertEqual(result.rows, len(frame))
        self.assertGreater(len(result.lines_trained), 0)


class TestRunCardsBenchmarkFromDb(unittest.TestCase):
    def test_empty_db_result_is_skipped_without_raising(self):
        with mock.patch("src.ml.markets.cards.cards_market.MatchRepository") as repo_cls:
            repo_cls.return_value.search_filter.return_value = []
            result = run_cards_benchmark_from_db(save_model=False)

        self.assertEqual(result.status, "skipped_no_data")


if __name__ == "__main__":
    unittest.main()
