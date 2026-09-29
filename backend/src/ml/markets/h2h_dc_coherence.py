"""Regola di coerenza h2h -> dc: vittoria casa implica 1X (e 12).

Il mercato `dc` binario e' P(1X) vs P(trasferta). Home win ⊂ 1X, quindi
se h2h predice casa la decisione dc non puo' essere "non 1X".
`p_dc` deve rispettare P(1X) >= P(casa).
"""
from __future__ import annotations

import numpy as np


def force_dc_1x_when_home_win(h2h_pred: np.ndarray, dc_pred: np.ndarray) -> np.ndarray:
    """Se h2h dice casa (1), dc diventa 1X (1). 12 non e' il target binario."""
    h2h_pred = np.asarray(h2h_pred, dtype=int)
    forced = np.asarray(dc_pred, dtype=int).copy()
    forced[h2h_pred == 1] = 1
    return forced


def enforce_p_1x_at_least_p_home(p_home: np.ndarray, p_1x: np.ndarray) -> np.ndarray:
    """P(1X) = P(casa)+P(pareggio) >= P(casa). Clip in [0, 1]."""
    return np.clip(np.maximum(np.asarray(p_1x, dtype=float), np.asarray(p_home, dtype=float)), 0.0, 1.0)


def count_home_but_not_1x(h2h_pred: np.ndarray, dc_pred: np.ndarray) -> int:
    h2h_pred = np.asarray(h2h_pred, dtype=int)
    dc_pred = np.asarray(dc_pred, dtype=int)
    return int(((h2h_pred == 1) & (dc_pred == 0)).sum())
