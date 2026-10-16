import json
import unittest

from src.service_ia.training.prediction_logger import PredictionLogger
from src.storage import bucket_store


class PredictionLoggerTailTest(unittest.TestCase):
    def test_tail_of_a_large_log_keeps_only_the_newest_rows(self):
        logger = PredictionLogger(key="best_models/predictions_log_tail.jsonl")
        lines = []
        for index in range(20):
            lines.append(
                json.dumps(
                    {
                        "timestamp": f"2026-01-{index + 1:02d}T00:00:00+00:00",
                        "fixture_id": index,
                        "market": "h2h",
                        "prediction": 1,
                        "probability": 0.5,
                        "model_run_id": "run",
                    }
                )
            )
        raw = ("\n".join(lines) + "\n").encode("utf-8")
        bucket_store.put_bytes(logger.key, raw)
        line_bytes = len((lines[-1] + "\n").encode("utf-8"))
        logger._TAIL_BYTES = line_bytes * 3 + 5

        rows = logger.tail(limit=10)

        self.assertEqual(rows[-1]["fixture_id"], 19)
        self.assertNotIn(0, [row["fixture_id"] for row in rows])
        self.assertGreaterEqual(len(rows), 2)

    def test_missing_log_is_empty(self):
        logger = PredictionLogger(key="best_models/missing_predictions_log.jsonl")
        self.assertEqual(logger.tail(), [])
