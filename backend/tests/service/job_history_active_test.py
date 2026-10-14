import unittest
from unittest import mock

from src.jobs import job_history as jh


class FakeBucket:
    def __init__(self):
        self.data = {}

    def get_json(self, key):
        return self.data.get(key)

    def put_json(self, key, value):
        self.data[key] = value

    def list_json(self, prefix):
        return [v for k, v in self.data.items() if k.startswith(prefix)]


class ActiveRunsTest(unittest.TestCase):
    def setUp(self):
        self.bucket = FakeBucket()
        patcher = mock.patch.object(jh, "bucket_store", self.bucket)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.history = jh.JobHistory()

    def test_running_job_is_listed_until_it_finishes(self):
        row = self.history.create_job(job_type="scorer_predictions", status="running", params={})
        self.history.set_progress(row["job_id"], 40, "Formazioni")
        active = self.history.active_runs()
        self.assertEqual(active["scorer_predictions"]["summary"]["progress_pct"], 40)
        self.assertEqual(active["scorer_predictions"]["summary"]["phase"], "Formazioni")

        self.history.mark_success(row["job_id"], {"predicted": 3})
        self.assertEqual(self.history.active_runs(), {})

    def test_failed_job_is_not_active(self):
        row = self.history.create_job(job_type="settlement", status="running", params={})
        self.history.mark_failed(row["job_id"], {"message": "boom"})
        self.assertEqual(self.history.active_runs(), {})

    def test_queued_then_running_keeps_one_entry(self):
        row = self.history.queue_job(job_type="prediction_snapshot_refresh")
        self.assertIn("prediction_snapshot_refresh", self.history.active_runs())
        self.history.mark_running(row["job_id"])
        self.assertEqual(len(self.history.active_runs()), 1)

    def test_stale_running_rows_are_ignored(self):
        row = self.history.create_job(job_type="settlement", status="running", params={})
        stored = self.bucket.data[self.history._job_key(row["job_id"])]
        stored["timestamp"] = "2020-01-01T00:00:00+00:00"
        self.assertEqual(self.history.active_runs(), {})

    def test_job_list_exposes_active_run(self):
        from src.jobs import job_settings

        row = self.history.create_job(job_type="scorer_predictions", status="running", params={})
        rows = job_settings.list_job_definitions()
        scorer = next(r for r in rows if r["job_id"] == "scorer_predictions")
        self.assertEqual(scorer["active_run"]["job_id"], row["job_id"])
        other = next(r for r in rows if r["job_id"] == "scorer_training")
        self.assertIsNone(other["active_run"])
