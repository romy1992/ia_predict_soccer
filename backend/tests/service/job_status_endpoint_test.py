import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from src.jobs.job_history import JobHistory
from src.api import main as api_main


class TestJobStatusEndpoint(unittest.TestCase):
    def test_get_job_returns_progress_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = JobHistory(path=f"{tmp}/jobs.jsonl")
            row = history.queue_job("prediction_snapshot_refresh", params={"target_date": "2026-09-16"})
            history.mark_running(row["job_id"])
            history.update_job(
                row["job_id"],
                summary={"percent": 40.0, "fixtures_done": 4, "fixtures_total": 10, "target_date": "2026-09-16"},
            )

            with mock.patch.object(api_main, "JobHistory", lambda: history):
                client = TestClient(api_main.app)
                missing = client.get("/jobs/does-not-exist")
                self.assertEqual(missing.status_code, 404)
                ok = client.get(f"/jobs/{row['job_id']}")
                self.assertEqual(ok.status_code, 200)
                payload = ok.json()
                self.assertEqual(payload["status"], "running")
                self.assertEqual(payload["summary"]["percent"], 40.0)
                self.assertEqual(payload["summary"]["fixtures_done"], 4)


if __name__ == "__main__":
    unittest.main()
