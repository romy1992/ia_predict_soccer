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


if __name__ == "__main__":
    unittest.main()

