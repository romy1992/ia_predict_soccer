import unittest

import numpy as np

from src.ml.markets.h2h_dc_coherence import (
    count_home_but_not_1x,
    enforce_p_1x_at_least_p_home,
    force_dc_1x_when_home_win,
)


class TestH2hDcCoherence(unittest.TestCase):
    def test_home_win_forces_double_chance_1x(self):
        h2h = np.array([1, 1, 0, 0])
        dc = np.array([0, 1, 0, 1])
        forced = force_dc_1x_when_home_win(h2h, dc)
        np.testing.assert_array_equal(forced, [1, 1, 0, 1])
        self.assertEqual(count_home_but_not_1x(h2h, dc), 1)
        self.assertEqual(count_home_but_not_1x(h2h, forced), 0)

    def test_probability_floor_is_p_home(self):
        p_home = np.array([0.7, 0.2, 0.55])
        p_1x = np.array([0.4, 0.8, 0.55])
        out = enforce_p_1x_at_least_p_home(p_home, p_1x)
        np.testing.assert_allclose(out, [0.7, 0.8, 0.55])
