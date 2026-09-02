"""Double Chance derivato da 1X2 coerente (MARKET-02).

Vincolo del task: la Double Chance NON e' un modello addestrato a parte,
ma una DERIVAZIONE matematica deterministica delle tre probabilita' 1X2
gia' prodotte dal vero mercato multiclasse (MARKET-01,
`src/ml/markets/market_1x2.py`). Non c'e' training, non c'e' leakage
temporale da gestire qui: la correttezza si riduce interamente a coerenza
aritmetica, verificata esplicitamente dai test.

Formule (nomenclatura bookmaker tra parentesi):
- P("Home/Draw") = P(HOME) + P(DRAW)   [== "1X"]
- P("Home/Away") = P(HOME) + P(AWAY)   [== "12"]
- P("Draw/Away") = P(DRAW) + P(AWAY)   [== "X2"]

Le chiavi stringa ("Home/Draw", "Draw/Away", "Home/Away") riprendono la
stessa convenzione gia' usata in `src/api/dashboard_service.py` per il
mercato "dc" legacy (binario, 1X vs 2): qui pero' i tre outcome sono
tutti e tre disponibili simultaneamente, come richiesto dall'acceptance
criteria "Tre outcome DC disponibili" (il mercato "dc" binario esistente
resta INVARIATO e continua a modellare solo P(1X) vs P(2), vedi EXP-05).

Nota su "somma": a differenza del vero 1X2 (esclusivo, somma=1), i tre
outcome Double Chance NON sono mutuamente esclusivi (ogni esito base e'
incluso in esattamente due DC outcome su tre), quindi la loro somma e'
sempre 2.0 quando le probabilita' 1X2 di partenza sommano a 1 (verificato
nei test), MAI 1.0.
"""

from __future__ import annotations

from typing import Any, Optional

from src.ml.markets.market_1x2 import Market1x2Expert, OUTCOME_LABELS

DOUBLE_CHANCE_OUTCOMES: tuple[str, str, str] = ("Home/Draw", "Home/Away", "Draw/Away")

# Tolleranza sulla somma P(HOME)+P(DRAW)+P(AWAY) prima di considerare il
# 1X2 di partenza "non coerente" (titolo del task: "da 1X2 coerente").
_SUM_TOLERANCE = 1e-6


def derive_double_chance_probabilities(
    p_home: float,
    p_draw: float,
    p_away: float,
) -> dict[str, float]:
    """Deriva le tre probabilita' Double Chance da un 1X2 coerente (somma=1).

    Solleva ValueError se p_home+p_draw+p_away non e' ~1: la Double Chance
    derivata da un 1X2 incoerente non e' un dato affidabile da esporre.
    """
    total = float(p_home) + float(p_draw) + float(p_away)
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise ValueError(
            f"1X2 non coerente: P(HOME)+P(DRAW)+P(AWAY)={total} (atteso 1.0). "
            "Double Chance derivata solo da un 1X2 gia' valido."
        )

    return {
        "Home/Draw": float(p_home) + float(p_draw),
        "Home/Away": float(p_home) + float(p_away),
        "Draw/Away": float(p_draw) + float(p_away),
    }


def derive_double_chance_from_dict(probabilities_1x2: dict[str, float]) -> dict[str, float]:
    """Variante comoda: accetta il dict {"HOME":.., "DRAW":.., "AWAY":..}
    prodotto da `Market1x2Expert.predict_proba_dict` (stesso ordine/label
    di `OUTCOME_LABELS`, nessuna assunzione di posizione)."""
    missing = [label for label in OUTCOME_LABELS if label not in probabilities_1x2]
    if missing:
        raise ValueError(f"Probabilita' 1X2 incomplete, mancano: {missing}")

    return derive_double_chance_probabilities(
        p_home=probabilities_1x2["HOME"],
        p_draw=probabilities_1x2["DRAW"],
        p_away=probabilities_1x2["AWAY"],
    )


def fair_odds_from_probabilities(probabilities: dict[str, float]) -> dict[str, Optional[float]]:
    """Fair odd = 1/P per outcome. None (non inf) quando P<=0: una quota
    infinita non e' un dato utile da esporre a valle (API/betting)."""
    fair_odds: dict[str, Optional[float]] = {}
    for outcome, probability in probabilities.items():
        fair_odds[outcome] = (1.0 / probability) if probability and probability > 0 else None
    return fair_odds


def build_double_chance_output(probabilities_1x2: dict[str, float]) -> dict[str, Any]:
    """Output standard per un singolo match: probabilita' + fair odds DC."""
    dc_probabilities = derive_double_chance_from_dict(probabilities_1x2)
    return {
        "source_1x2": dict(probabilities_1x2),
        "probabilities": dc_probabilities,
        "fair_odds": fair_odds_from_probabilities(dc_probabilities),
    }


class DoubleChanceExpert:
    """Wrapper batch: deriva Double Chance per ogni riga a partire da un
    `Market1x2Expert` gia' addestrato/caricato (MARKET-01). Nessun modello
    proprio: solo derivazione deterministica, coerente per costruzione."""

    def __init__(self, market_1x2_expert: Market1x2Expert):
        self.market_1x2_expert = market_1x2_expert

    def derive(self, X: Any) -> list[dict[str, Any]]:
        rows_1x2 = self.market_1x2_expert.predict_proba_dict(X)
        return [build_double_chance_output(row) for row in rows_1x2]

