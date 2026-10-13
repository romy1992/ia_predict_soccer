"""Test del report di copertura marcatori: SQLite in-memory, nessun DB reale."""

from __future__ import annotations

import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.data.player.coverage_report import build_report, render_report
from src.data.player.player_lineup_models import Base, PlayerLineup
from src.data.player.player_models import PlayerMatchEvent
from src.service_ia.model.match import Match

FINISHED = ["FT"]


def _match(fixture_id, league, season, status="FT", home=0, away=0):
    return Match(id_fixture=fixture_id, league_match=league, season=season, status=status,
                 score_home=home, score_away=away)


def _lineup(fixture_id, skip_midfielders=0):
    """22 titolari (2 squadre: 1 G, 4 D, 4 M, 2 F) + 1 panchinaro per squadra."""
    rows, pid = [], fixture_id * 1000
    for team in (1, 2):
        plan = ["G"] + ["D"] * 4 + ["M"] * (4 - (skip_midfielders if team == 1 else 0)) + ["F"] * 2
        for pos in plan:
            pid += 1
            rows.append(PlayerLineup(fixture_id=fixture_id, player_id=pid, player_name=f"P{pid}", team_id=team,
                                     position=pos, is_starter=True))
        pid += 1
        rows.append(PlayerLineup(fixture_id=fixture_id, player_id=pid, player_name=f"P{pid}", team_id=team,
                                 position="F", is_starter=False))
    return rows


def _event(fixture_id, n, event_type="Goal", detail="Normal Goal", player_id=None, comments=None):
    return PlayerMatchEvent(id_event=f"{fixture_id}-{n}", fixture_id=fixture_id, event_type=event_type,
                            event_detail=detail, player_id=player_id, comments=comments)


class CoverageReportTest(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine, tables=[Match.__table__, PlayerLineup.__table__,
                                                 PlayerMatchEvent.__table__])
        self.session = sessionmaker(bind=engine)()
        s = self.session
        # F1 (Serie A 2025): completa e coerente. Gol normale dell'attaccante 1021 + autogol, punteggio 1-1.
        s.add(_match(1, 135, 2025, home=1, away=1))
        s.add_all(_lineup(1))
        s.add_all([
            _event(1, 1, player_id=1011),                                   # 1011 = primo F della squadra 1
            _event(1, 2, detail="Own Goal", player_id=1002),                # autogol: non e' un "marcatore"
            _event(1, 3, detail="Missed Penalty", player_id=1011),          # non entra nel punteggio
            _event(1, 4, detail="Penalty", comments="Penalty Shootout", player_id=1012),  # lotteria: esclusa
            _event(1, 5, event_type="subst", detail="Substitution 1", player_id=1003),
        ])
        # F2: formazione completa ma NESSUN evento catturato.
        s.add(_match(2, 135, 2025, home=0, away=0))
        s.add_all(_lineup(2))
        # F3: eventi + formazione incompleta (20 titolari), gol di un giocatore fuori formazione,
        # e punteggio 3 contro 1 solo gol registrato.
        s.add(_match(3, 135, 2025, home=2, away=1))
        s.add_all(_lineup(3, skip_midfielders=2))
        s.add(_event(3, 1, player_id=999))
        # Fuori scope: fixture non conclusa e lega non richiesta.
        s.add(_match(4, 135, 2025, status="NS"))
        s.add(_match(5, 78, 2025))
        s.commit()
        self.report = build_report(s, [135], [2025], FINISHED)

    def tearDown(self):
        self.session.close()

    def test_scope_only_finished_fixtures_of_requested_leagues(self):
        row = self.report.by_competition.iloc[0]
        self.assertEqual((row["lega"], row["stagione"], row["partite"]), ("Serie A", 2025, 3))

    def test_events_and_lineups_coverage(self):
        row = self.report.by_competition.iloc[0]
        self.assertEqual((row["con_eventi"], row["con_formazioni"]), (2, 3))
        self.assertAlmostEqual(row["events_pct"], 100 * 2 / 3)
        self.assertAlmostEqual(row["lineups_pct"], 100.0)
        self.assertAlmostEqual(row["full_lineup_pct"], 100 * 2 / 3)  # F3 ha 20 titolari

    def test_goal_matching_ignores_own_goals_missed_penalties_and_shootouts(self):
        row = self.report.by_competition.iloc[0]
        # Marcatori veri: 1011 in F1 (in formazione) e 999 in F3 (fuori formazione).
        self.assertEqual(row["gol_non_autogol"], 2)
        self.assertAlmostEqual(row["goal_in_lineup_pct"], 50.0)

    def test_score_consistency_counts_own_goals_but_not_shootouts(self):
        row = self.report.by_competition.iloc[0]
        # F1: 2 gol nel punteggio (normale + autogol) == 1+1. F3: 1 gol contro 3.
        self.assertAlmostEqual(row["score_consistent_pct"], 50.0)

    def test_base_rates_only_use_fixtures_with_events_and_lineups(self):
        rates = self.report.base_rates.set_index("ruolo_codice")
        # F1 (22 titolari) + F3 (20 titolari); F2 senza eventi e' esclusa.
        self.assertEqual(rates.loc["-", "titolari"], 42)
        self.assertEqual(rates.loc["-", "segnato"], 1)
        self.assertEqual(rates.loc["F", "segnato"], 1)
        self.assertEqual(rates.loc["G", "segnato"], 0)

    def test_anomalies_and_warnings_point_to_the_bad_fixture(self):
        self.assertEqual(self.report.anomalies["fixture_id"].tolist(), [3])
        problem = self.report.anomalies.iloc[0]["problema"]
        self.assertIn("titolari=20", problem)
        self.assertIn("punteggio=3", problem)
        warns = " ".join(self.report.warnings())
        self.assertIn("events_pct", warns)
        self.assertIn("goal_in_lineup_pct", warns)

    def test_render_mentions_attention_and_rates(self):
        text = render_report(self.report)
        self.assertIn("ATTENZIONE", text)
        self.assertIn("Attaccante", text)
        self.assertIn("Serie A", text)

    def test_empty_scope_is_reported_not_crashed(self):
        report = build_report(self.session, [999], [2025], FINISHED)
        self.assertTrue(report.by_competition.empty)
        self.assertIn("Nessuna fixture conclusa", render_report(report))


if __name__ == "__main__":
    unittest.main()
