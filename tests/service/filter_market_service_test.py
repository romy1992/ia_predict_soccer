import unittest
import uuid

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


if __name__ == "__main__":
    unittest.main()

