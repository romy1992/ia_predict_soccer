"""Segnale "Bet Over" per Under/Over 1.5/2.5/3.5/4.5 (richiesto esplicitamente
dall'operatore, 2026-09-08: "voglio puntare al modello che dica over, ma
anche per 1.5, 2.5, 3.5 e 4.5").

Distinto DELIBERATAMENTE dal pick mostrato in dashboard
(`_pick_and_odd_for_prediction`/`_extract_probability` in `dashboard_service.py`,
soglia fissa 0.5, MAI toccata da questo modulo) e dalla `DecisionPolicy`
generica (BET-04, soglie PLAY/BORDERLINE uguali per ogni mercato/outcome):
qui la soglia e' quella orientata alla PRECISIONE sulla classe "Over"
(`scripts/analysis/find_betting_thresholds.py`, non alla statistica di
Youden usata per il pick), specifica per mercato, perche' un falso positivo
su una scommessa costa direttamente - conta la precisione, non l'equilibrio
tra le due classi.

Il segnale e' quindi INDIPENDENTE dal pick: puo' attivarsi (P(Over) >=
soglia_mercato) anche quando il pick mostrato resta "Under" (tipico sui
mercati con base rate Over bassa, es. 4.5: soglia 0.1776, ben sotto 0.5) -
e viceversa il pick puo' mostrare "Over" senza che il segnale sia attivo, se
la soglia di precisione per quel mercato e' piu' alta di quella di Youden
usata per il pick (es. 1.5: soglia segnale 0.7925 contro soglia pick 0.5).
Nessuna delle due logiche esistenti viene modificata da questo modulo.

Valori derivati da `scripts/analysis/find_betting_thresholds.py` (dataset
reale, champion post-SMOTE gia' promossi a `production` il 2026-09-08),
soglia al recall floor 30% (buon compromesso volume/affidabilita' tra i 5
livelli calcolati, 10/20/30/40/50%) - risultato completo in
`best_models/betting_thresholds.json`. Una ritaratura futura (nuovo recall
floor, nuovo champion) richiede una nuova versione qui, MAI un edit
silenzioso di questi valori (stesso principio di `DecisionPolicy`/BET-04 e
di `ModelRegistry`: nessuna promozione o ritaratura implicita)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

OVER_SIGNAL_POLICY_VERSION = "over_signal_policy_v1"


@dataclass(frozen=True)
class OverSignalThresholds:
    market: str
    probability_threshold: float
    expected_precision: float
    expected_recall: float
    n_oof: int
    source_recall_floor: str = "30%"


# fmt: off
OVER_SIGNAL_THRESHOLDS: dict[str, OverSignalThresholds] = {
    "under_over_1_5": OverSignalThresholds("under_over_1_5", 0.7925, 0.8382, 0.3636, n_oof=4170),
    "under_over_2_5": OverSignalThresholds("under_over_2_5", 0.5657, 0.6533, 0.3006, n_oof=7620),
    "under_over_3_5": OverSignalThresholds("under_over_3_5", 0.3566, 0.4000, 0.3021, n_oof=4165),
    "under_over_4_5": OverSignalThresholds("under_over_4_5", 0.1776, 0.2163, 0.3214, n_oof=4140),
}
# fmt: on


def evaluate_over_signal(market: str, p_over: Optional[float]) -> Optional[dict]:
    """`None` se il mercato non e' un under/over con soglia calibrata, o se
    `p_over` non e' disponibile (mai un segnale inventato). Altrimenti un
    dict con `signal` (bool) + i valori di riferimento della soglia, cosi'
    il chiamante puo' sempre mostrare "a quale precisione/recall attesi
    corrisponde" invece di un booleano nudo."""
    spec = OVER_SIGNAL_THRESHOLDS.get(market)
    if spec is None or p_over is None:
        return None
    return {
        "signal": bool(float(p_over) >= spec.probability_threshold),
        "threshold": spec.probability_threshold,
        "expected_precision": spec.expected_precision,
        "expected_recall": spec.expected_recall,
        "policy_version": OVER_SIGNAL_POLICY_VERSION,
    }
