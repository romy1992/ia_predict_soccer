import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from src.data.quality_report_service import DataQualityService
from src.service_ia.model.match import Match, Odds, Statistics


class FakeMatchRepo:
    def __init__(self, matches):
        self._matches = matches

    def search_all(self):
        return self._matches


class FakeSnapshotRepo:
    def __init__(self, rows):
        self._rows = rows

    def list_all(self):
        return self._rows


def _build_matches():
    match_ok = Match(
        id_match_fk="m1",
        id_fixture=1001,
        id_team_home=10,
        id_team_away=20,
        name_home="A",
        name_away="B",
        date_match="2026-09-01T18:00:00+00:00",
        current_league=135,
        season=2026,
        status="FT",
        settlement_status="complete",
    )
    match_ok.statistics = [
        Statistics(statistics_team_id=10, score_ft=2),
        Statistics(statistics_team_id=20, score_ft=1),
    ]
    match_ok.odds = [Odds(h2h={"home_Book": "1.80"}, under_over_2_5={"over 2.5_Book": "1.95"})]

    match_bad = Match(
        id_match_fk="m2",
        id_fixture=1002,
        id_team_home=11,
        id_team_away=21,
        name_home="C",
        name_away="D",
        date_match="bad-date",
        current_league=None,
        season=None,
        status="FT",
        settlement_status=None,
    )
    match_bad.statistics = []
    match_bad.odds = []

    return [match_ok, match_bad]


class TestDataQualityService(unittest.TestCase):
    @patch("src.data.quality_report_service._safe_orphan_count", return_value=0)
    def test_build_report_machine_readable(self, _mock_orphans):
        rows = [
            SimpleNamespace(
                fixture_id=1001,
                market="h2h",
                captured_at=datetime(2026, 9, 1, 17, 0, tzinfo=timezone.utc),
            ),
            SimpleNamespace(
                fixture_id=1001,
                market="h2h",
                captured_at=datetime(2026, 9, 1, 19, 0, tzinfo=timezone.utc),
            ),
        ]
        service = DataQualityService(
            match_repo=FakeMatchRepo(_build_matches()),
            snapshot_repo=FakeSnapshotRepo(rows),
        )

        report = service.build_report(top_n=5)

        self.assertIn("coverage", report)
        self.assertIn("anomalies", report)
        self.assertIn("distribution", report)
        self.assertIn("temporal_checks", report)
        self.assertIn("settlement", report)

        self.assertEqual(report["source"]["fixtures_total"], 2)
        self.assertEqual(report["anomalies"]["duplicate_fixture_count"], 0)
        self.assertEqual(report["temporal_checks"]["invalid_match_datetime_count"], 1)
        self.assertEqual(report["temporal_checks"]["snapshot_after_kickoff_count"], 1)
        self.assertEqual(report["settlement"]["complete"], 1)

    @patch("src.data.quality_report_service._safe_orphan_count", return_value=0)
    def test_build_report_with_filters(self, _mock_orphans):
        rows = [
            SimpleNamespace(
                fixture_id=1001,
                market="h2h",
                captured_at=datetime(2026, 9, 1, 17, 0, tzinfo=timezone.utc),
            ),
            SimpleNamespace(
                fixture_id=1002,
                market="h2h",
                captured_at=datetime(2026, 9, 1, 17, 0, tzinfo=timezone.utc),
            ),
        ]
        service = DataQualityService(
            match_repo=FakeMatchRepo(_build_matches()),
            snapshot_repo=FakeSnapshotRepo(rows),
        )

        report = service.build_report(top_n=5, seasons=[2026], leagues=[135])

        self.assertEqual(report["source"]["fixtures_total"], 1)
        self.assertEqual(report["source"]["filters"]["seasons"], [2026])
        self.assertEqual(report["source"]["filters"]["leagues"], [135])


def _mean_stats(id_home: int, id_away: int) -> list[dict]:
    return [
        {"id_team": id_home, "Shots on Goal": 5},
        {"id_team": id_away, "Shots on Goal": 3},
    ]


def _build_under_over_matches() -> list[Match]:
    # match_all: FT, odds per TUTTE e 4 le soglie, total_goals=3 (over 1.5/2.5, under 3.5/4.5).
    match_all = Match(
        id_match_fk="u1", id_fixture=2001, id_team_home=10, id_team_away=20,
        date_match="2026-01-01T18:00:00+00:00", current_league=135, season=2026, status="FT",
    )
    match_all.statistics = [Statistics(statistics_team_id=10, score_ft=2), Statistics(statistics_team_id=20, score_ft=1)]
    match_all.mean_statistics = _mean_stats(10, 20)
    match_all.odds = [Odds(
        under_over_1_5={"over_bookA": "1.30"}, under_over_2_5={"over_bookA": "1.90"},
        under_over_3_5={"over_bookA": "3.00"}, under_over_4_5={"over_bookA": "5.00"},
    )]

    # match_only_2_5: FT, odds SOLO per 2.5, total_goals=1 (under su tutte le soglie).
    match_only_2_5 = Match(
        id_match_fk="u2", id_fixture=2002, id_team_home=11, id_team_away=21,
        date_match="2026-01-02T18:00:00+00:00", current_league=135, season=2026, status="FT",
    )
    match_only_2_5.statistics = [Statistics(statistics_team_id=11, score_ft=1), Statistics(statistics_team_id=21, score_ft=0)]
    match_only_2_5.mean_statistics = _mean_stats(11, 21)
    match_only_2_5.odds = [Odds(under_over_2_5={"over_bookA": "1.90"})]

    # match_not_ft: NS, non deve contare in fixtures_ft_total.
    match_not_ft = Match(
        id_match_fk="u3", id_fixture=2003, id_team_home=12, id_team_away=22,
        date_match="2026-01-03T18:00:00+00:00", current_league=135, season=2026, status="NS",
    )
    match_not_ft.statistics = []
    match_not_ft.mean_statistics = None
    match_not_ft.odds = [Odds(under_over_1_5={"over_bookA": "1.30"})]

    # match_no_odds: FT ma senza NESSUNA quota -> escluso da tutte le soglie.
    match_no_odds = Match(
        id_match_fk="u4", id_fixture=2004, id_team_home=13, id_team_away=23,
        date_match="2026-01-04T18:00:00+00:00", current_league=135, season=2026, status="FT",
    )
    match_no_odds.statistics = [Statistics(statistics_team_id=13, score_ft=1), Statistics(statistics_team_id=23, score_ft=1)]
    match_no_odds.mean_statistics = _mean_stats(13, 23)
    match_no_odds.odds = []

    return [match_all, match_only_2_5, match_not_ft, match_no_odds]


class TestBuildUnderOverThresholdReport(unittest.TestCase):
    def test_reports_odds_coverage_and_usable_rows_per_threshold(self):
        service = DataQualityService(
            match_repo=FakeMatchRepo(_build_under_over_matches()),
            snapshot_repo=FakeSnapshotRepo([]),
        )

        report = service.build_under_over_threshold_report(min_train_rows=1, min_valid_rows=1, n_splits=2)

        self.assertEqual(report["fixtures_ft_total"], 3)  # match_not_ft escluso

        over_1_5 = report["per_threshold"]["under_over_1_5"]
        self.assertEqual(over_1_5["fixtures_with_odds_for_market"], 1)  # solo match_all
        self.assertEqual(over_1_5["usable_rows_for_training"], 1)
        self.assertEqual(over_1_5["class_balance"], {"under_0": 0, "over_1": 1})
        self.assertEqual(over_1_5["positive_class_ratio_over"], 1.0)

        over_2_5 = report["per_threshold"]["under_over_2_5"]
        self.assertEqual(over_2_5["fixtures_with_odds_for_market"], 2)  # match_all + match_only_2_5
        self.assertEqual(over_2_5["usable_rows_for_training"], 2)
        self.assertEqual(over_2_5["class_balance"], {"under_0": 1, "over_1": 1})

        over_4_5 = report["per_threshold"]["under_over_4_5"]
        self.assertEqual(over_4_5["fixtures_with_odds_for_market"], 1)
        self.assertEqual(over_4_5["class_balance"], {"under_0": 1, "over_1": 0})

        # Solo match_all ha ODDS per TUTTE e 4 le soglie contemporaneamente.
        self.assertEqual(report["cross_threshold"]["fixtures_with_all_four_thresholds_odds_available"], 1)

    def test_accepts_preloaded_matches_without_extra_query(self):
        """`matches` pre-caricato deve essere riusato COSI' COM'E' (nessuna
        chiamata a `match_repo.search_all`, verificato forzando un repo che
        solleverebbe se interrogato)."""

        class ExplodingMatchRepo:
            def search_all(self):
                raise AssertionError("search_all non deve essere chiamato quando 'matches' e' gia' fornito")

        service = DataQualityService(match_repo=ExplodingMatchRepo(), snapshot_repo=FakeSnapshotRepo([]))
        report = service.build_under_over_threshold_report(matches=_build_under_over_matches())
        self.assertEqual(report["fixtures_ft_total"], 3)


if __name__ == "__main__":
    unittest.main()

