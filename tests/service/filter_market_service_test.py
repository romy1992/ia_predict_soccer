import unittest
import uuid
from unittest import mock

import numpy as np

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.repository.base.crud_repository as crud_repository_module
from src.service_ia.model.match import Base, Match, Odds
from src.service_ia.training.market_service.filter_market_service import FilterMarketService


class TestBuildPredictionFrames(unittest.TestCase):
    """Fix performance (cambio giorno lento in Dashboard): il vecchio
    `build_prediction_frame(market, fixture_id)` rifaceva la query Match da
    zero per OGNI mercato richiesto (fino a 9 SUPPORTED_MARKETS), anche se
    fixture/odds/statistics sono identici tra un mercato e l'altro. Questi
    test esercitano un vero engine SQLite in-memory (stesso pattern di
    `dashboard_service_test.py`/`crud_repository_test.py`) per verificare
    DAVVERO il numero di query eseguite, non solo il valore di ritorno."""

    def _make_session_factory(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def _seed_match(self, session_factory, fixture_id: int):
        with session_factory() as session:
            match = Match(
                id_match_fk=str(uuid.uuid4()),
                id_fixture=fixture_id,
                date_match="2026-09-01T18:00:00+00:00",
                status="NS",
                season=2026,
                current_league=135,
            )
            match.odds = [
                Odds(
                    id_odds_fk=str(uuid.uuid4()),
                    h2h={"Home_bookA": "1.80", "Draw_bookA": "3.40", "Away_bookA": "4.20"},
                    under_over_2_5={"Over 2.5_bookA": "1.95", "Under 2.5_bookA": "1.85"},
                    # "cards"/"corners"/... restano NULL: nessuna quota per
                    # quei mercati su questa fixture (comportamento realistico).
                )
            ]
            session.add(match)
            session.commit()

    def test_build_prediction_frames_runs_single_query_for_multiple_markets(self):
        session_factory = self._make_session_factory()
        self._seed_match(session_factory, fixture_id=9001)

        original_session_local = crud_repository_module.SessionLocal
        crud_repository_module.SessionLocal = session_factory
        try:
            service = FilterMarketService()
            call_count = {"n": 0}
            original_filter_by = service.match_repo.filter_by

            def counting_filter_by(**kwargs):
                call_count["n"] += 1
                return original_filter_by(**kwargs)

            service.match_repo.filter_by = counting_filter_by

            frames = service.build_prediction_frames(
                fixture_id=9001, markets=["h2h", "under_over_2_5", "cards"]
            )
        finally:
            crud_repository_module.SessionLocal = original_session_local

        # UNA sola query condivisa tra i 3 mercati richiesti (prima: 3 query,
        # una per mercato - qui la vera regressione di performance).
        self.assertEqual(call_count["n"], 1)
        self.assertIn("h2h", frames)
        self.assertIn("under_over_2_5", frames)
        # "cards" non ha quote popolate su questa fixture -> nessuna riga.
        self.assertNotIn("cards", frames)
        self.assertEqual(frames["h2h"].iloc[0]["id_fixture"], 9001)

    def test_build_prediction_frame_wrapper_still_works_for_single_market(self):
        """Retro-compatibilita': `main.py`/`monitoring_service.py`/
        `model_consensus.py` chiamano ancora `build_prediction_frame` con UN
        solo mercato - deve continuare a funzionare identico a prima."""
        session_factory = self._make_session_factory()
        self._seed_match(session_factory, fixture_id=9002)

        original_session_local = crud_repository_module.SessionLocal
        crud_repository_module.SessionLocal = session_factory
        try:
            service = FilterMarketService()
            frame = service.build_prediction_frame(market="h2h", fixture_id=9002)
        finally:
            crud_repository_module.SessionLocal = original_session_local

        self.assertIsNotNone(frame)
        self.assertEqual(frame.iloc[0]["id_fixture"], 9002)

    def test_build_prediction_frame_unknown_fixture_returns_none(self):
        session_factory = self._make_session_factory()

        original_session_local = crud_repository_module.SessionLocal
        crud_repository_module.SessionLocal = session_factory
        try:
            service = FilterMarketService()
            frame = service.build_prediction_frame(market="h2h", fixture_id=424242)
        finally:
            crud_repository_module.SessionLocal = original_session_local

        self.assertIsNone(frame)

    def test_build_prediction_frame_invalid_market_raises(self):
        service = FilterMarketService.__new__(FilterMarketService)
        with self.assertRaises(ValueError):
            service.build_prediction_frame(market="not_a_market", fixture_id=1)

    def test_build_prediction_frames_from_match_accepts_orm_object_without_query(self):
        """Il percorso usato da `DashboardService._predict_fixture` quando
        riceve un `Match` GIA' caricato in batch altrove (query fatta
        altrove, es. `_fetch_matches`): zero interazioni col repository."""
        session_factory = self._make_session_factory()
        self._seed_match(session_factory, fixture_id=9003)

        with session_factory() as session:
            match = session.query(Match).filter_by(id_fixture=9003).first()

            service = FilterMarketService.__new__(FilterMarketService)  # niente match_repo/DB
            frames = service.build_prediction_frames_from_match(match, markets=["h2h", "under_over_2_5"])

        self.assertIn("h2h", frames)
        self.assertIn("under_over_2_5", frames)
        self.assertEqual(frames["h2h"].iloc[0]["id_fixture"], 9003)

    def test_build_prediction_frames_from_match_accepts_plain_dict(self):
        service = FilterMarketService.__new__(FilterMarketService)
        match_dict = {
            "id_fixture": 42,
            "season": 2026,
            "current_league": 135,
            "date_match": "2026-09-01T18:00:00+00:00",
            "odds": [{"h2h": {"Home_bookA": "1.80", "Draw_bookA": "3.40", "Away_bookA": "4.20"}}],
        }

        frames = service.build_prediction_frames_from_match(match_dict, markets=["h2h", "dc"])

        self.assertIn("h2h", frames)
        self.assertNotIn("dc", frames)

    def test_build_prediction_frames_empty_markets_returns_empty_dict(self):
        service = FilterMarketService.__new__(FilterMarketService)
        self.assertEqual(service.build_prediction_frames_from_match({"odds": []}, markets=[]), {})


class TestBuildPredictionFramesLineMarkets(unittest.TestCase):
    """2026-09-13: Corners/Cards a linea configurabile (`LINE_MARKETS`) -
    wiring nel percorso di predizione condiviso con gli altri mercati
    (`build_prediction_frames_from_match`, usato da `PredictionSnapshotService`)."""

    def _match(self):
        return {
            "id_fixture": 999,
            "season": 2026,
            "current_league": 39,
            "date_match": "2026-09-13T18:00:00+00:00",
            "referee": "Rossi",
            "mean_statistics": [
                {"id_team": 100, "Corner Kicks": 5.5, "Yellow Cards": 2.0},
                {"id_team": 200, "Corner Kicks": 4.5, "Yellow Cards": 1.5},
            ],
            "id_team_home": 100,
            "id_team_away": 200,
            "odds": [
                {
                    "corners": {"Over 8.5_bookA": 1.90, "Under 8.5_bookA": 1.95, "Over 9.5_bookA": 2.10, "Under 9.5_bookA": 1.75},
                    "cards": {"Over 3.5_bookA": 1.80, "Under 3.5_bookA": 2.00},
                }
            ],
        }

    def test_corners_line_market_produces_line_specific_columns(self):
        service = FilterMarketService.__new__(FilterMarketService)
        frames = service.build_prediction_frames_from_match(self._match(), markets=["corners_line_8_5"])

        self.assertIn("corners_line_8_5", frames)
        row = frames["corners_line_8_5"].iloc[0]
        self.assertIn("odds_mean_line_8_5", row.index)
        self.assertIn("corner_mean_home_dedicated", row.index)
        # La riga grezza contiene le quote di TUTTE le linee (stesso
        # comportamento del training, vedi build_corners_frame_from_records):
        # e' il caricamento del modello (feature_names dal registry, in
        # PredictionSnapshotService) a restringere alla sola linea attiva,
        # non questo builder.
        self.assertIn("odds_mean_line_9_5", row.index)

    def test_cards_line_market_includes_referee_features_via_cached_index(self):
        with mock.patch("src.ml.markets.cards.cards_market.MatchRepository") as repo_cls, mock.patch(
            "src.ml.markets.cards.cards_market.convert_orm_match_to_dict", side_effect=lambda x: x
        ):
            repo_cls.return_value.search_filter.return_value = []
            service = FilterMarketService.__new__(FilterMarketService)
            frames = service.build_prediction_frames_from_match(self._match(), markets=["cards_line_3_5"])

        self.assertIn("cards_line_3_5", frames)
        row = frames["cards_line_3_5"].iloc[0]
        self.assertIn("odds_mean_line_3_5", row.index)
        for col in [
            "referee_avg_cards_prior",
            "referee_severity_index_prior",
            "referee_matches_officiated_prior",
            "referee_has_history",
        ]:
            self.assertIn(col, row.index)

    def test_line_markets_included_in_normalized_dashboard_market_request(self):
        from src.api.dashboard_service import DashboardService

        normalized = DashboardService._normalize_market_request(["cards_line_3_5", "corners_line_8_5", "bogus_market"])
        self.assertIn("cards_line_3_5", normalized)
        self.assertIn("corners_line_8_5", normalized)
        self.assertNotIn("bogus_market", normalized)

    def test_unknown_market_prefix_is_ignored(self):
        service = FilterMarketService.__new__(FilterMarketService)
        frames = service.build_prediction_frames_from_match(self._match(), markets=["corners_line_99_5"])
        self.assertEqual(frames, {})


class TestExtractLineFromOddsKey(unittest.TestCase):
    """2026-09-12: Corners/Cards ('Corners Over Under'/'Cards Over/Under')
    mettono TUTTE le linee quotate in un unico bucket piatto (a differenza
    degli Under/Over gol, gia' separati per soglia dall'ingestion) - qui si
    estrae la linea dalla chiave (formato reale confermato da
    `TestBuildPredictionFrames` sopra: 'Over 2.5_bookA'/'Under X.Y_book')
    per poter poi filtrare le quote pertinenti a UNA sola linea."""

    def test_extracts_line_from_realistic_key_format(self):
        self.assertEqual(FilterMarketService._extract_line_from_odds_key("Over 8.5_bet365"), 8.5)
        self.assertEqual(FilterMarketService._extract_line_from_odds_key("Under 9.5_pinnacle"), 9.5)
        self.assertEqual(FilterMarketService._extract_line_from_odds_key("over 10.5_bookA"), 10.5)

    def test_returns_none_when_no_number_present(self):
        self.assertIsNone(FilterMarketService._extract_line_from_odds_key("Home_bookA"))

    def test_bookmaker_name_with_digits_does_not_confuse_extraction(self):
        # La linea precede sempre il nome bookmaker nella chiave prodotta da
        # `map_odds()` (f'{alternate_value}_{name_book}') - il primo numero
        # trovato deve essere sempre quello della linea, mai una cifra nel
        # nome del bookmaker (es. "888sport", "1xBet").
        self.assertEqual(FilterMarketService._extract_line_from_odds_key("Over 8.5_888sport"), 8.5)
        self.assertEqual(FilterMarketService._extract_line_from_odds_key("Under 3.5_1xBet"), 3.5)


class TestExtractLineSpecificOddsFeatures(unittest.TestCase):
    def test_pools_only_matching_line_both_sides_all_bookmakers(self):
        market_odds = {
            "Over 8.5_bookA": "1.90",
            "Under 8.5_bookA": "1.95",
            "Over 8.5_bookB": "1.85",
            "Under 8.5_bookB": "2.00",
            # Altre linee, DEVONO essere escluse dal pool per la linea 8.5:
            "Over 9.5_bookA": "2.50",
            "Under 9.5_bookA": "1.55",
            "Over 11.5_bookA": "6.75",
            "Under 11.5_bookA": "1.06",
        }
        features = FilterMarketService._extract_line_specific_odds_features(market_odds, line=8.5)

        self.assertEqual(features["odds_count"], 4.0)
        self.assertAlmostEqual(features["odds_min"], 1.85)
        self.assertAlmostEqual(features["odds_max"], 2.00)
        # Con la vecchia estrazione "pooled" (tutte le linee insieme)
        # odds_max sarebbe stato 6.75 (la linea 11.5) - qui deve restare
        # circoscritto alla sola linea 8.5.
        self.assertLess(features["odds_max"], 6.75)

    def test_no_matching_line_returns_empty_like_no_odds(self):
        market_odds = {"Over 9.5_bookA": "2.50", "Under 9.5_bookA": "1.55"}
        features = FilterMarketService._extract_line_specific_odds_features(market_odds, line=8.5)
        self.assertEqual(features, {})

    def test_different_lines_produce_different_pools_from_same_raw_odds(self):
        # Stesso identico dizionario grezzo multi-linea: filtrando per 8.5 vs
        # per 11.5 si devono ottenere pool completamente diversi - e' questo
        # l'intero punto della funzione (prima, un'unica pool mischiava tutto).
        market_odds = {
            "Over 8.5_bookA": "1.30",
            "Under 8.5_bookA": "3.20",
            "Over 11.5_bookA": "6.75",
            "Under 11.5_bookA": "1.06",
        }
        low_line = FilterMarketService._extract_line_specific_odds_features(market_odds, line=8.5)
        high_line = FilterMarketService._extract_line_specific_odds_features(market_odds, line=11.5)

        self.assertAlmostEqual(low_line["odds_mean"], (1.30 + 3.20) / 2)
        self.assertAlmostEqual(high_line["odds_mean"], (6.75 + 1.06) / 2)
        self.assertNotAlmostEqual(low_line["odds_mean"], high_line["odds_mean"])


class TestMissingStatsArePreservedForAnalysis(unittest.TestCase):
    """2026-09-13: `_safe_float` azzera i mancanti ALLA FONTE, quindi dopo
    l'estrazione un dato assente e uno zero reale sono indistinguibili. Su
    `expected_goals` (assente sul 46% delle partite) significa dare al modello
    "squadra che non tira mai in porta" invece di "non lo sappiamo". Il
    default resta l'azzeramento (i modelli registrati sono addestrati cosi'),
    ma serve un percorso che preservi i NaN per poter fare EDA."""

    MATCH = {
        "id_team_home": 10,
        "id_team_away": 20,
        "mean_statistics": [
            {"id_team": 10, "expected_goals": None, "shots_on_goal": 4.2},
            {"id_team": 20, "expected_goals": 1.3, "shots_on_goal": 5.1},
        ],
    }

    def test_default_still_zeroes_missing_values(self):
        features = FilterMarketService._extract_mean_features(self.MATCH)
        self.assertEqual(features["expected_goals_home_stat"], 0.0)
        self.assertEqual(features["expected_goals_diff_stat"], -1.3)

    def test_keep_missing_preserves_nan_and_propagates_to_the_difference(self):
        features = FilterMarketService._extract_mean_features(self.MATCH, keep_missing=True)
        self.assertTrue(np.isnan(features["expected_goals_home_stat"]))
        # La differenza con un termine ignoto e' a sua volta ignota: azzerarla
        # inventerebbe un vantaggio/svantaggio mai osservato.
        self.assertTrue(np.isnan(features["expected_goals_diff_stat"]))
        # I valori realmente presenti non vengono toccati.
        self.assertAlmostEqual(features["expected_goals_away_stat"], 1.3)
        self.assertAlmostEqual(features["shots_on_goal_home_stat"], 4.2)

    def test_a_real_zero_stays_zero_in_both_modes(self):
        match = {
            "id_team_home": 10,
            "id_team_away": 20,
            "mean_statistics": [
                {"id_team": 10, "red_cards": 0},
                {"id_team": 20, "red_cards": 0},
            ],
        }
        for keep in (False, True):
            features = FilterMarketService._extract_mean_features(match, keep_missing=keep)
            self.assertEqual(features["red_cards_home_stat"], 0.0, f"keep_missing={keep}")


class TestPerOutcomeOddsFeatures(unittest.TestCase):
    """Feature quote SEPARATE PER ESITO (2026-09-13): la media legacy e'
    calcolata su tutto il bucket del mercato, che contiene TUTTI gli esiti -
    quindi non corrisponde alla quota di nessuna scommessa reale. Qui si
    verifica che ogni esito abbia la propria media fra bookmaker."""

    def test_over_and_under_get_separate_means(self):
        market_odds = {
            "over 2.5_Bet365": "1.80",
            "over 2.5_Pinnacle": "1.82",
            "under 2.5_Bet365": "2.05",
            "under 2.5_Pinnacle": "2.01",
        }
        features = FilterMarketService._extract_per_outcome_odds_features(market_odds)

        self.assertAlmostEqual(features["odds_mean_over_2_5"], (1.80 + 1.82) / 2)
        self.assertAlmostEqual(features["odds_mean_under_2_5"], (2.05 + 2.01) / 2)
        self.assertEqual(features["odds_count_over_2_5"], 2.0)
        self.assertEqual(features["odds_count_under_2_5"], 2.0)
        # La media legacy mescola i due lati e cade in mezzo: non e' la
        # quota di nessuno dei due esiti.
        legacy = FilterMarketService._extract_market_odds_features(market_odds)
        self.assertGreater(legacy["odds_mean"], features["odds_mean_over_2_5"])
        self.assertLess(legacy["odds_mean"], features["odds_mean_under_2_5"])

    def test_goal_no_goal_keys_with_trailing_underscore_are_parsed(self):
        """`map_odds()` produce 'goal__book'/'no_goal__book' (l'esito ha gia'
        un underscore finale): lo split deve cadere sull'ULTIMO underscore,
        altrimenti 'no_goal' finirebbe spezzato."""
        market_odds = {
            "goal__Bet365": "1.72",
            "goal__Pinnacle": "1.75",
            "no_goal__Bet365": "2.10",
            "no_goal__Pinnacle": "2.05",
        }
        features = FilterMarketService._extract_per_outcome_odds_features(market_odds)

        self.assertAlmostEqual(features["odds_mean_goal"], (1.72 + 1.75) / 2)
        self.assertAlmostEqual(features["odds_mean_no_goal"], (2.10 + 2.05) / 2)

    def test_three_outcome_market_keeps_outcomes_apart(self):
        market_odds = {
            "home_Bet365": "1.95",
            "draw_Bet365": "3.60",
            "away_Bet365": "4.20",
        }
        features = FilterMarketService._extract_per_outcome_odds_features(market_odds)

        self.assertAlmostEqual(features["odds_mean_home"], 1.95)
        self.assertAlmostEqual(features["odds_mean_draw"], 3.60)
        self.assertAlmostEqual(features["odds_mean_away"], 4.20)

    def test_normalized_probabilities_sum_to_one_and_overround_is_the_margin(self):
        market_odds = {"home_B": "2.00", "draw_B": "4.00", "away_B": "4.00"}
        features = FilterMarketService._extract_per_outcome_odds_features(market_odds)

        # 1/2 + 1/4 + 1/4 = 1.0 esatto: mercato senza margine.
        self.assertAlmostEqual(features["overround"], 1.0)
        total = sum(v for k, v in features.items() if k.startswith("prob_norm_"))
        self.assertAlmostEqual(total, 1.0)

        # Con margine: le probabilita' grezze sommano > 1, le normalizzate no.
        with_margin = FilterMarketService._extract_per_outcome_odds_features(
            {"home_B": "1.90", "draw_B": "3.70", "away_B": "3.80"}
        )
        self.assertGreater(with_margin["overround"], 1.0)
        self.assertAlmostEqual(
            sum(v for k, v in with_margin.items() if k.startswith("prob_norm_")), 1.0
        )

    def test_line_specific_extraction_also_splits_the_two_sides(self):
        """Filtrare per linea non bastava: dentro la linea 8.5 restavano
        insieme 'over 8.5' e 'under 8.5' (entrambe contengono "8.5")."""
        market_odds = {
            "over 8.5_bookA": "1.50",
            "under 8.5_bookA": "2.49",
            "over 11.5_bookA": "6.75",
        }
        features = FilterMarketService._extract_line_specific_odds_features(market_odds, line=8.5)

        self.assertAlmostEqual(features["odds_mean_over_8_5"], 1.50)
        self.assertAlmostEqual(features["odds_mean_under_8_5"], 2.49)
        self.assertNotIn("odds_mean_over_11_5", features)

    def test_empty_or_invalid_odds_return_empty_like_legacy(self):
        self.assertEqual(FilterMarketService._extract_per_outcome_odds_features({}), {})
        self.assertEqual(
            FilterMarketService._extract_per_outcome_odds_features({"over 2.5_B": "0"}), {}
        )

    def test_legacy_features_are_still_emitted_for_already_promoted_models(self):
        """I modelli gia' promossi elencano le colonne legacy nei loro
        `feature_names`: se sparissero, il serving le riempirebbe con 0.0
        azzerando in silenzio tutta l'informazione quote."""
        market_odds = {"over 2.5_B": "1.80", "under 2.5_B": "2.05"}
        legacy = FilterMarketService._extract_market_odds_features(market_odds)
        for name in ("odds_mean", "odds_std", "odds_min", "odds_max", "odds_count", "odds_slot_1"):
            self.assertIn(name, legacy)
        self.assertTrue(FilterMarketService.LEGACY_ODDS_FEATURES.issuperset(legacy.keys()))


if __name__ == "__main__":
    unittest.main()

