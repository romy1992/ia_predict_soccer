"""Value Engine (BET-02, Fase BETTING).

Sostituisce `DashboardService._value_decision` (REWRITE), che chiamava
impropriamente "edge" quello che in realta' e' un Expected Value (EV).
Distingue esplicitamente (acceptance criteria "Edge ed EV distinti"):

- `prob_edge = p_model - p_market_fair`: differenza di PROBABILITA' tra
  Oracle e mercato (fair, overround gia' rimosso da BET-01) — indipendente
  dalla quota, non varia se la quota cambia a parita' di probabilita'.
- `ev = p_model * odd - 1`: valore atteso IN TERMINI DI QUOTA — questo era
  il calcolo chiamato (impropriamente) "edge" nel codice precedente.

Consuma direttamente l'output di BET-01 (`FairOddsOutcome`):
`evaluate_value_from_fair_odds_outcome` garantisce per costruzione che
`p_model`/`p_market_fair`/`odd` si riferiscano allo STESSO outcome, cosa
che il vecchio codice non garantiva (bug "outcome scorretto" corretto in
`DashboardService._pick_and_odd_for_prediction`: rimosso il fallback che
usava la quota "Draw" per calcolare il value di un pick "Away").

Soglie VERSIONATE (mai hardcoded inline nel corpo della funzione): stesso
principio di versionamento gia' in uso per i modelli (`ModelRegistry`) e
per la calibrazione, qui applicato alla policy di decisione betting
(06_BETTING_INTELLIGENCE.md: "le soglie non devono essere hardcoded
globalmente: vanno versionate e tarate tramite backtest out-of-sample").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.oracle.fair_odds.fair_odds_engine import FairOddsOutcome

PLAY = "PLAY"
BORDERLINE = "BORDERLINE"
NO_BET = "NO BET"


@dataclass(frozen=True)
class ValueDecisionPolicy:
    """Soglie versionate per la decisione PLAY/BORDERLINE/NO BET.

    Stessi valori numerici gia' in uso prima di BET-02 (nessuna modifica al
    comportamento "as-is" per non introdurre un cambio di policy non
    richiesto da questo task): la differenza e' che ora sono un oggetto
    esplicito e VERSIONATO (`version`), non tre numeri hardcoded nel corpo
    del metodo — una nuova taratura (es. da backtest, BET-03) richiedera'
    una nuova istanza con una nuova `version`, mai un edit silenzioso di
    questa.
    """

    version: str
    play_min_probability: float
    play_min_ev: float
    borderline_min_probability: float
    borderline_min_ev: float


DEFAULT_POLICY = ValueDecisionPolicy(
    version="value_policy_v1",
    play_min_probability=0.62,
    play_min_ev=0.03,
    borderline_min_probability=0.55,
    borderline_min_ev=0.0,
)


@dataclass
class ValueDecision:
    """Output STANDARD della valutazione (acceptance criteria "Edge ed EV
    distinti"): entrambe le metriche sempre presenti come campi separati,
    mai un unico numero ambiguo."""

    market: str
    outcome: str
    p_model: Optional[float]
    p_market_fair: Optional[float]
    odd: Optional[float]
    prob_edge: Optional[float]
    ev: Optional[float]
    decision: str
    reason: str
    policy_version: str


def compute_prob_edge(p_model: Optional[float], p_market_fair: Optional[float]) -> Optional[float]:
    """prob_edge = p_model - p_market_fair (acceptance criteria).

    `None` se uno dei due valori non e' disponibile: mai una differenza
    calcolata rispetto a un valore mancante/sostituito."""
    if p_model is None or p_market_fair is None:
        return None
    return float(p_model) - float(p_market_fair)


def compute_expected_value(p_model: Optional[float], odd: Optional[float]) -> Optional[float]:
    """ev = p_model*odd - 1 (acceptance criteria).

    Gestisce esplicitamente la quota mancante/non valida (acceptance
    criteria "Gestire quota mancante"): `None`, mai 0 o un altro valore
    fittizio che verrebbe silenziosamente interpretato come "EV nullo"."""
    if p_model is None or odd is None:
        return None
    try:
        odd_value = float(odd)
    except (TypeError, ValueError):
        return None
    if odd_value <= 0.0:
        return None
    return float(p_model) * odd_value - 1.0


def evaluate_value(
    market: str,
    outcome: str,
    p_model: Optional[float],
    p_market_fair: Optional[float],
    odd: Optional[float],
    policy: ValueDecisionPolicy = DEFAULT_POLICY,
) -> ValueDecision:
    """Valuta un outcome (stesso market/outcome per p_model/p_market_fair/odd
    — responsabilita' del chiamante, garantita per costruzione da
    `evaluate_value_from_fair_odds_outcome`) e produce `prob_edge`, `ev` e
    la decisione PLAY/BORDERLINE/NO BET secondo `policy`."""
    prob_edge = compute_prob_edge(p_model, p_market_fair)
    ev = compute_expected_value(p_model, odd)

    if p_model is None:
        decision, reason = NO_BET, "Probabilita' modello non disponibile"
    elif odd is None or float(odd) <= 0.0:
        decision, reason = NO_BET, "Quota non disponibile"
    elif p_model >= policy.play_min_probability and ev is not None and ev >= policy.play_min_ev:
        decision, reason = PLAY, "Confidenza alta e EV positivo"
    elif p_model >= policy.borderline_min_probability and ev is not None and ev >= policy.borderline_min_ev:
        decision, reason = BORDERLINE, "Confidenza media o EV ridotto"
    else:
        decision, reason = NO_BET, "Confidenza/EV insufficienti"

    return ValueDecision(
        market=market,
        outcome=outcome,
        p_model=p_model,
        p_market_fair=p_market_fair,
        odd=odd,
        prob_edge=prob_edge,
        ev=ev,
        decision=decision,
        reason=reason,
        policy_version=policy.version,
    )


def evaluate_value_from_fair_odds_outcome(
    fair_odds_outcome: FairOddsOutcome,
    policy: ValueDecisionPolicy = DEFAULT_POLICY,
) -> ValueDecision:
    """Comodo: consuma direttamente l'output di BET-01 (`FairOddsOutcome`),
    garantendo per costruzione che `p_model`/`p_market_fair`/`odd` si
    riferiscano allo STESSO outcome (acceptance criteria "Usare outcome
    corretto")."""
    return evaluate_value(
        market=fair_odds_outcome.market,
        outcome=fair_odds_outcome.outcome,
        p_model=fair_odds_outcome.p_model,
        p_market_fair=fair_odds_outcome.p_market_fair,
        odd=fair_odds_outcome.odd,
        policy=policy,
    )
