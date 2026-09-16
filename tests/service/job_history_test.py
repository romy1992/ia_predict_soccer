import tempfile
import threading
import unittest

from src.jobs.job_history import JobHistory


class TestJobHistory(unittest.TestCase):
    def test_concurrent_writers_do_not_lose_rows(self):
        # Riproduce lo scenario di produzione: il container `api` (job
        # manuale con progress bar, molti `update_job` ravvicinati) e il
        # container `scheduler` (stesso file `jobs_history.jsonl`, stesso
        # volume) scrivono in concorrenza. Senza lock un read-modify-write
        # perde le righe scritte dall'altro thread nel frattempo, facendo
        # fallire un `update_job` successivo con "Job id non trovato".
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/jobs.jsonl"
            job_ids = [f"job-{i}" for i in range(20)]
            errors = []

            def worker(job_id):
                try:
                    history = JobHistory(path=path)
                    history.create_job(job_type="prediction_snapshot_refresh", status="queued", job_id=job_id)
                    for step in range(10):
                        history.update_job(job_id=job_id, summary={"progress": step})
                    history.mark_success(job_id=job_id, summary={"progress": 10})
                except Exception as exc:  # pragma: no cover - il test fallisce comunque sotto
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(job_id,)) for job_id in job_ids]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [])
            rows = JobHistory(path=path).tail(limit=100)
            self.assertEqual({row["job_id"] for row in rows}, set(job_ids))
            for row in rows:
                self.assertEqual(row["status"], "success")
                self.assertEqual(row["summary"]["progress"], 10)

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
