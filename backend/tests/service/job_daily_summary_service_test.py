import unittest
from datetime import datetime, timedelta, timezone

from src.jobs.job_daily_summary_service import JobDailySummaryService


class FakeJobHistory:
    def __init__(self, rows):
        self._rows = rows

    def tail(self, limit=5000, job_type=None, status=None):
        rows = self._rows
        if job_type:
            rows = [row for row in rows if row.get("job_type") == job_type]
        if status:
            rows = [row for row in rows if row.get("status") == status]
        return rows


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _report(inserted=0, updated=0, skipped=0, failed=0, fixtures_seen=0):
    return {
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
        "failed": failed,
        "fixtures_seen": fixtures_seen,
        "params": {"leagues": [135, 39]},
        "errors": [],
    }


class TestJobDailySummaryService(unittest.TestCase):
    def test_aggregates_flat_report_shape(self):
        """job_type 'import'/'future_sync': summary['report'] = {...}."""
        now = datetime.now(timezone.utc)
        rows = [
            {
                "job_type": "import",
                "finished_at": _iso(now),
                "summary": {"report": _report(inserted=5, failed=1, fixtures_seen=6)},
            }
        ]
        service = JobDailySummaryService(history=FakeJobHistory(rows))
        report = service.build_report(days=30)

        self.assertEqual(len(report["days"]), 1)
        day = report["days"][0]
        self.assertEqual(day["runs"], 1)
        self.assertEqual(day["inserted"], 5)
        self.assertEqual(day["failed"], 1)
        self.assertEqual(day["fixtures_seen"], 6)
        self.assertEqual(day["locked_skips"], 0)

    def test_aggregates_differently_named_report_key(self):
        """job_type 'settlement': summary['import_report'] = {...} (stesse
        chiavi del report ma nome diverso - l'estrazione e' generica, non
        deve dipendere dal nome della chiave)."""
        now = datetime.now(timezone.utc)
        rows = [
            {
                "job_type": "settlement",
                "finished_at": _iso(now),
                "summary": {"from_date": "2026-10-05", "import_report": _report(inserted=3)},
            }
        ]
        service = JobDailySummaryService(history=FakeJobHistory(rows))
        report = service.build_report(days=30)

        self.assertEqual(report["days"][0]["inserted"], 3)

    def test_sums_multiple_nested_reports_in_same_run(self):
        """job_type 'daily_refresh': piu' sotto-fasi nello stesso run
        (es. played_matches + future_sync) - entrambe contribuiscono."""
        now = datetime.now(timezone.utc)
        rows = [
            {
                "job_type": "daily_refresh",
                "finished_at": _iso(now),
                "summary": {
                    "played_matches": _report(inserted=2, updated=1),
                    "future_sync": _report(inserted=4),
                },
            }
        ]
        service = JobDailySummaryService(history=FakeJobHistory(rows))
        report = service.build_report(days=30)

        day = report["days"][0]
        self.assertEqual(day["inserted"], 6)
        self.assertEqual(day["updated"], 1)

    def test_skipped_locked_counted_separately_not_as_zero_report(self):
        """Un run saltato per lock (un altro processo aveva gia' la stessa
        tabella) non deve contare come '0 fixture prese' insieme ai run
        reali - e' un'informazione diversa, tracciata a parte."""
        now = datetime.now(timezone.utc)
        rows = [
            {
                "job_type": "today_update",
                "finished_at": _iso(now),
                "summary": {"skipped_locked": True, "motivo": "lock attivo"},
            }
        ]
        service = JobDailySummaryService(history=FakeJobHistory(rows))
        report = service.build_report(days=30)

        day = report["days"][0]
        self.assertEqual(day["runs"], 1)
        self.assertEqual(day["locked_skips"], 1)
        self.assertEqual(day["inserted"], 0)

    def test_groups_by_day_and_sorts_descending(self):
        today = datetime.now(timezone.utc)
        yesterday = today - timedelta(days=1)
        rows = [
            {"job_type": "import", "finished_at": _iso(yesterday), "summary": {"report": _report(inserted=1)}},
            {"job_type": "import", "finished_at": _iso(today), "summary": {"report": _report(inserted=2)}},
        ]
        service = JobDailySummaryService(history=FakeJobHistory(rows))
        report = service.build_report(days=30)

        self.assertEqual(len(report["days"]), 2)
        self.assertEqual(report["days"][0]["date"], today.date().isoformat())
        self.assertEqual(report["days"][1]["date"], yesterday.date().isoformat())

    def test_excludes_runs_older_than_window(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=40)
        rows = [
            {"job_type": "import", "finished_at": _iso(old), "summary": {"report": _report(inserted=9)}},
            {"job_type": "import", "finished_at": _iso(now), "summary": {"report": _report(inserted=1)}},
        ]
        service = JobDailySummaryService(history=FakeJobHistory(rows))
        report = service.build_report(days=30)

        self.assertEqual(len(report["days"]), 1)
        self.assertEqual(report["days"][0]["inserted"], 1)

    def test_falls_back_to_timestamp_when_finished_at_missing(self):
        now = datetime.now(timezone.utc)
        rows = [{"job_type": "import", "timestamp": _iso(now), "summary": {"report": _report(inserted=1)}}]
        service = JobDailySummaryService(history=FakeJobHistory(rows))
        report = service.build_report(days=30)

        self.assertEqual(len(report["days"]), 1)

    def test_by_job_type_breakdown(self):
        now = datetime.now(timezone.utc)
        rows = [
            {"job_type": "import", "finished_at": _iso(now), "summary": {"report": _report(inserted=1)}},
            {"job_type": "settlement", "finished_at": _iso(now), "summary": {"import_report": _report(inserted=2)}},
        ]
        service = JobDailySummaryService(history=FakeJobHistory(rows))
        report = service.build_report(days=30)

        by_type = report["days"][0]["by_job_type"]
        self.assertEqual(by_type["import"]["inserted"], 1)
        self.assertEqual(by_type["settlement"]["inserted"], 2)

    def test_job_type_filter_delegates_to_history(self):
        now = datetime.now(timezone.utc)
        rows = [
            {"job_type": "import", "finished_at": _iso(now), "summary": {"report": _report(inserted=1)}},
            {"job_type": "settlement", "finished_at": _iso(now), "summary": {"import_report": _report(inserted=2)}},
        ]
        service = JobDailySummaryService(history=FakeJobHistory(rows))
        report = service.build_report(days=30, job_type="import")

        self.assertEqual(report["days"][0]["inserted"], 1)

    def test_invalid_days_raises(self):
        service = JobDailySummaryService(history=FakeJobHistory([]))
        with self.assertRaises(ValueError):
            service.build_report(days=0)

    def test_no_rows_returns_empty_days(self):
        service = JobDailySummaryService(history=FakeJobHistory([]))
        report = service.build_report(days=30)
        self.assertEqual(report["days"], [])


if __name__ == "__main__":
    unittest.main()
