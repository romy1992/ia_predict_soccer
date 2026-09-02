import tempfile
import unittest

from src.jobs.job_history import JobHistory


class TestJobHistory(unittest.TestCase):
    def test_queue_running_success_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = JobHistory(path=f"{tmp}/jobs.jsonl")

            queued = history.queue_job("import", params={"fixture_date": "2026-09-02"})
            job_id = queued["job_id"]

            running = history.mark_running(job_id=job_id, params={"fixture_date": "2026-09-02"})
            success = history.mark_success(job_id=job_id, summary={"inserted": 10, "updated": 3})

            self.assertEqual(queued["status"], "queued")
            self.assertEqual(running["status"], "running")
            self.assertEqual(success["status"], "success")
            self.assertIsNotNone(success["started_at"])
            self.assertIsNotNone(success["finished_at"])

            rows = history.tail(limit=20, job_type="import", status="success")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["summary"]["inserted"], 10)

    def test_mark_failed_sets_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = JobHistory(path=f"{tmp}/jobs.jsonl")
            row = history.create_job(job_type="retrain", status="running", params={"market": "h2h"})
            job_id = row["job_id"]

            failed = history.mark_failed(job_id=job_id, error={"message": "boom"})
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["error"]["message"], "boom")
            self.assertIsNotNone(failed["finished_at"])


if __name__ == "__main__":
    unittest.main()

