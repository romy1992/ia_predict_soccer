import os
import tempfile
import unittest
from unittest import mock

from src.oracle.betslip import market_isolation_policy


class _IsolatedMarketsPath(unittest.TestCase):
    """Isola `isolated_markets.json` in una directory temporanea per ogni
    test, cosi' nessun test tocca mai il file reale in `best_models/` del
    progetto (stesso principio di `job_settings_test.py`)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        path = os.path.join(self._tmp.name, "isolated_markets.json")
        patcher = mock.patch.object(market_isolation_policy, "_isolated_markets_path", return_value=path)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestGetIsolatedMarkets(_IsolatedMarketsPath):
    def test_returns_default_when_no_file(self):
        self.assertEqual(market_isolation_policy.get_isolated_markets(), frozenset({"cards"}))

    def test_returns_default_on_corrupt_file(self):
        path = market_isolation_policy._isolated_markets_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        self.assertEqual(market_isolation_policy.get_isolated_markets(), frozenset({"cards"}))

    def test_returns_default_when_file_is_not_a_list(self):
        path = market_isolation_policy._isolated_markets_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"cards": true}')
        self.assertEqual(market_isolation_policy.get_isolated_markets(), frozenset({"cards"}))

    def test_normalizes_case_and_whitespace(self):
        market_isolation_policy.set_isolated_markets([" Cards ", "CORNERS", ""])
        self.assertEqual(market_isolation_policy.get_isolated_markets(), frozenset({"cards", "corners"}))


class TestSetIsolatedMarkets(_IsolatedMarketsPath):
    def test_update_then_read_back(self):
        result = market_isolation_policy.set_isolated_markets(["cards", "corners"])
        self.assertEqual(result, frozenset({"cards", "corners"}))
        self.assertEqual(market_isolation_policy.get_isolated_markets(), frozenset({"cards", "corners"}))

    def test_replaces_entirely_not_merges(self):
        market_isolation_policy.set_isolated_markets(["cards", "corners"])
        market_isolation_policy.set_isolated_markets(["h2h"])
        self.assertEqual(market_isolation_policy.get_isolated_markets(), frozenset({"h2h"}))

    def test_empty_list_clears_isolation(self):
        market_isolation_policy.set_isolated_markets(["cards"])
        market_isolation_policy.set_isolated_markets([])
        self.assertEqual(market_isolation_policy.get_isolated_markets(), frozenset())

    def test_persists_atomically_no_tmp_file_left(self):
        market_isolation_policy.set_isolated_markets(["cards"])
        path = market_isolation_policy._isolated_markets_path()
        self.assertTrue(os.path.exists(path))
        self.assertFalse(os.path.exists(f"{path}.tmp"))


if __name__ == "__main__":
    unittest.main()
