"""Segnale a soglia OTTIMALE per Corners/Cards a linea configurabile
(MARKET-05/06, 2026-09-13) - stesso principio di `over_signal_policy.py`
per i gol, necessario perche' a soglia 0.5 diverse linee sono DEGENERI
(eventi rari: es. cards 6.5 non prediceva mai "Over" a soglia 0.5, corners
11.5 quasi mai - verificato sul training reale del 2026-09-12).

Distinto DELIBERATAMENTE dal pick/probabilita' esposti in `predictions`
(soglia fissa 0.5, MAI toccata da questo modulo, stesso principio di
`over_signal_policy.py`): qui la soglia e' quella di Youden (statistica
J = TPR - FPR massimizzata, calcolata da `classification_report.py`
durante il training reale), diversa dalla soglia orientata alla PRECISIONE
usata da `over_signal_policy.py` per i gol - scelta consistente con come
questo progetto valuta gia' Corners/Cards ("tutte le metriche possibili",
soglia ottimale sempre riportata nel report di classificazione).

Il segnale e' quindi INDIPENDENTE dal pick a 0.5: puo' attivarsi (P(Over)
>= soglia_linea) anche quando il pick mostrato resta "Under", e viceversa.

Valori dal training reale (`best_models/corners_cards_from_export_summary.json`,
`classification_report_post.optimal_threshold`, 2026-09-12 - vedi
IMPLEMENTATION_LOG.md) - una ritaratura futura (nuovo training) richiede una
nuova versione qui, MAI un edit silenzioso di questi valori (stesso
principio di `DecisionPolicy`/BET-04 e di `over_signal_policy.py`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

LINE_MARKET_SIGNAL_POLICY_VERSION = "line_market_signal_policy_v1"


@dataclass(frozen=True)
class LineMarketSignalThreshold:
    market: str
    probability_threshold: float
    expected_accuracy: float
    expected_precision_over: float
    expected_recall_over: float
    n_oof: int


# fmt: off
LINE_MARKET_SIGNAL_THRESHOLDS: dict[str, LineMarketSignalThreshold] = {
    "corners_line_8_5": LineMarketSignalThreshold("corners_line_8_5", 0.6140, 0.5277, 0.6185, 0.5538, n_oof=8804),
    "corners_line_9_5": LineMarketSignalThreshold("corners_line_9_5", 0.4906, 0.5298, 0.5182, 0.4923, n_oof=8804),
    "corners_line_10_5": LineMarketSignalThreshold("corners_line_10_5", 0.3438, 0.4600, 0.3894, 0.7947, n_oof=8804),
    "corners_line_11_5": LineMarketSignalThreshold("corners_line_11_5", 0.2786, 0.5209, 0.2774, 0.5034, n_oof=8804),
    "cards_line_3_5": LineMarketSignalThreshold("cards_line_3_5", 0.6282, 0.6074, 0.6784, 0.6362, n_oof=8355),
    "cards_line_4_5": LineMarketSignalThreshold("cards_line_4_5", 0.4827, 0.6024, 0.5217, 0.5199, n_oof=8355),
    "cards_line_5_5": LineMarketSignalThreshold("cards_line_5_5", 0.2744, 0.5157, 0.3124, 0.7434, n_oof=8355),
    "cards_line_6_5": LineMarketSignalThreshold("cards_line_6_5", 0.1686, 0.5308, 0.2119, 0.7477, n_oof=8355),
}
# fmt: on


def evaluate_line_market_signal(market: str, p_over: Optional[float]) -> Optional[dict]:
    """`None` se il mercato non ha una soglia ottima calcolata, o se
    `p_over` non e' disponibile (mai un segnale inventato). Altrimenti un
    dict con `signal` (bool) + i valori di riferimento della soglia, cosi'
    il chiamante puo' sempre mostrare "a quale accuracy/precision/recall
    attesi corrisponde" invece di un booleano nudo."""
    spec = LINE_MARKET_SIGNAL_THRESHOLDS.get(market)
    if spec is None or p_over is None:
        return None
    return {
        "signal": bool(float(p_over) >= spec.probability_threshold),
        "threshold": spec.probability_threshold,
        "expected_accuracy": spec.expected_accuracy,
        "expected_precision_over": spec.expected_precision_over,
        "expected_recall_over": spec.expected_recall_over,
        "policy_version": LINE_MARKET_SIGNAL_POLICY_VERSION,
    }
