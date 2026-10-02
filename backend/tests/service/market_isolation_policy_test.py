import unittest

from src.oracle.betslip import market_isolation_policy
from src.storage import bucket_store

_DEFAULT_CARDS_LINES = frozenset({"cards_line_3_5", "cards_line_4_5", "cards_line_5_5", "cards_line_6_5"})


class _IsolatedMarketsPath(unittest.TestCase):
    """Isolamento garantito dalla fixture autouse di `conftest.py` (bucket
    fake nuovo ad ogni test): nessun mock di path necessario."""


class TestGetIsolatedMarkets(_IsolatedMarketsPath):
    def test_returns_default_when_no_file(self):
        self.assertEqual(market_isolation_policy.get_isolated_markets(), _DEFAULT_CARDS_LINES)

    def test_returns_default_on_corrupt_file(self):
        bucket_store.put_bytes(market_isolation_policy._ISOLATED_MARKETS_KEY, b"{not valid json")
        self.assertEqual(market_isolation_policy.get_isolated_markets(), _DEFAULT_CARDS_LINES)

    def test_returns_default_when_file_is_not_a_list(self):
        bucket_store.put_json(market_isolation_policy._ISOLATED_MARKETS_KEY, {"cards": True})
        self.assertEqual(market_isolation_policy.get_isolated_markets(), _DEFAULT_CARDS_LINES)

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

    def test_persists_on_the_bucket(self):
        market_isolation_policy.set_isolated_markets(["cards"])
        self.assertTrue(bucket_store.exists(market_isolation_policy._ISOLATED_MARKETS_KEY))


if __name__ == "__main__":
    unittest.main()
