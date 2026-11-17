"""Primitive di resilienza: backoff con jitter e circuit breaker (thread-safe)."""
import threading
import unittest

from src.service_ia.utility.resilience import (
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
    CircuitBreaker,
    backoff_delay,
)


class TestBackoff(unittest.TestCase):
    def test_limiti_del_jitter_e_tetto(self):
        for attempt in (1, 2, 3, 10):
            step = min(30.0, 1.5 * 2 ** (attempt - 1))
            lo = backoff_delay(attempt, base_seconds=1.5, max_seconds=30.0, rng=lambda: 0.0)
            hi = backoff_delay(attempt, base_seconds=1.5, max_seconds=30.0, rng=lambda: 1.0)
            self.assertAlmostEqual(lo, step / 2)
            self.assertAlmostEqual(hi, step)

    def test_input_non_validi_danno_zero(self):
        self.assertEqual(backoff_delay(0, base_seconds=1, max_seconds=5), 0.0)
        self.assertEqual(backoff_delay(1, base_seconds=0, max_seconds=5), 0.0)


class TestBreakerConcorrenza(unittest.TestCase):
    def test_half_open_ammette_una_sola_sonda_con_thread_concorrenti(self):
        t = [0.0]
        b = CircuitBreaker(failure_threshold=1, recovery_seconds=10, clock=lambda: t[0])
        self.assertTrue(b.record_failure())
        self.assertEqual(b.state, STATE_OPEN)
        t[0] = 11.0
        ammessi, barrier = [], threading.Barrier(16)

        def worker():
            barrier.wait()
            ammessi.append(b.allow_request())

        threads = [threading.Thread(target=worker) for _ in range(16)]
        [x.start() for x in threads]
        [x.join() for x in threads]
        self.assertEqual(sum(ammessi), 1)
        self.assertEqual(b.state, STATE_HALF_OPEN)
        b.record_success()
        self.assertEqual(b.state, STATE_CLOSED)

    def test_fallimenti_concorrenti_aprono_una_volta_sola(self):
        b = CircuitBreaker(failure_threshold=5, recovery_seconds=60)
        aperture, barrier = [], threading.Barrier(20)

        def worker():
            barrier.wait()
            aperture.append(b.record_failure())

        threads = [threading.Thread(target=worker) for _ in range(20)]
        [x.start() for x in threads]
        [x.join() for x in threads]
        self.assertEqual(sum(aperture), 1)
        self.assertEqual(b.state, STATE_OPEN)


if __name__ == "__main__":
    unittest.main()
