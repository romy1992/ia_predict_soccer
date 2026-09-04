"""Decision Policy versionata (BET-04, Fase BETTING).

BET-02 (`value_engine.py`) introduce `ValueDecisionPolicy`: un'unica istanza
GLOBALE (`DEFAULT_POLICY`) applicata a QUALUNQUE mercato/outcome. Questo
modulo la estende con soglie che possono variare ESPLICITAMENTE per mercato
(e opzionalmente per singolo outcome di un mercato), acceptance criteria
"Soglie per mercato/outcome", piu' due nuovi filtri:

- `min_samples`: numero minimo di bookmaker che quotano l'outcome
  (`FairOddsOutcome.bookmakers`, gia' calcolato da BET-01 da
  `compute_market_baseline`/ML-04) sotto il quale la fair probability e'
  considerata troppo rumorosa per decidere - NO BET indipendentemente da
  edge/EV (acceptance criteria "Min samples").
- `min_odd`/`max_odd` (opzionali): range di quota accettabile, per
  escludere quote estreme (spesso outlier o mercati illiquidi) a
  prescindere da edge/EV (acceptance criteria "Min/max odd opzionali").

Riusa DIRETTAMENTE (mai duplicati) `compute_prob_edge`/`compute_expected_value`
(BET-02): la definizione di prob_edge/EV resta UNA SOLA in tutto il
progetto, sia per il Value Engine "semplice" sia per questa Decision Policy.

Compatibilita' deliberata (stesso principio gia' seguito in BET-02: "non
introdurre un cambio di policy non richiesto da questo task"):
`DEFAULT_DECISION_POLICY` riproduce ESATTAMENTE la stessa decisione di
`DEFAULT_POLICY` (BET-02) per lo stesso input quando i nuovi filtri non
sono vincolanti (`min_samples=0`, `min_prob_edge=0.0`, `min/max odd=None`)
- il comportamento cambia SOLO se il chiamante configura esplicitamente un
override per un mercato (nuova capacita' richiesta da questo task, non
attivata di default: nessun override e' precaricato in produzione finche'
una taratura per-mercato non sia esplicitamente richiesta).

Acceptance criteria "Niente soglie globali hardcoded nel dashboard
service": `dashboard_service.py` chiama
`evaluate_decision_from_fair_odds_outcome(fair_odds_outcome)` passando solo
dati (market/outcome/probabilita'/quota/samples), MAI un numero di soglia
inline nel corpo del metodo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.oracle.fair_odds.fair_odds_engine import FairOddsOutcome
from src.oracle.value_engine.value_engine import (
    BORDERLINE,
    NO_BET,
    PLAY,
    compute_expected_value,
    compute_prob_edge,
)


@dataclass(frozen=True)
class DecisionThresholds:
    """Soglie applicabili a UN mercato (o mercato+outcome): stessa
    granularita' PLAY/BORDERLINE gia' in uso in `ValueDecisionPolicy`
    (BET-02) - stessi valori numerici di default (nessun cambio di
    comportamento "as-is") - estesa con `min_prob_edge`/`min_samples`/
    `min_odd`/`max_odd`."""

    play_min_probability: float = 0.62
    play_min_prob_edge: float = 0.0
    play_min_ev: float = 0.03
    borderline_min_probability: float = 0.55
    borderline_min_prob_edge: float = 0.0
    borderline_min_ev: float = 0.0
    min_samples: int = 0
    min_odd: Optional[float] = None
    max_odd: Optional[float] = None

    def __post_init__(self) -> None:
        # Fail-fast su una configurazione palesemente inconsistente (mai
        # silenziosa): un override futuro per mercato con min_odd>max_odd
        # escluderebbe SEMPRE ogni quota, quasi certamente un errore di
        # digitazione nella taratura, non un comportamento voluto.
        if self.min_odd is not None and self.max_odd is not None and self.min_odd > self.max_odd:
            raise ValueError(f"min_odd ({self.min_odd}) non puo' essere maggiore di max_odd ({self.max_odd})")
        if self.min_samples < 0:
            raise ValueError(f"min_samples non puo' essere negativo: {self.min_samples}")


DEFAULT_THRESHOLDS = DecisionThresholds()


@dataclass(frozen=True)
class DecisionPolicy:
    """Policy versionata (acceptance criteria "policy_version registrata"):
    soglie di default piu' override ESPLICITI per mercato o per singolo
    outcome di un mercato (acceptance criteria "Soglie per mercato/outcome")
    - mai una singola istanza globale indistinta come prima di questo task.

    Una nuova taratura (es. da backtest BET-03, task futuro) richiede una
    nuova istanza con una nuova `version`, mai un edit silenzioso di
    questa (stesso principio di `ValueDecisionPolicy`/ORACLE-03)."""

    version: str
    default: DecisionThresholds = DEFAULT_THRESHOLDS
    per_market: dict[str, DecisionThresholds] = field(default_factory=dict)
    per_market_outcome: dict[tuple[str, str], DecisionThresholds] = field(default_factory=dict)

    def thresholds_for(self, market: str, outcome: str) -> DecisionThresholds:
        """Risolve le soglie per un market/outcome: priorita' override
        outcome specifico > override mercato > default (acceptance
        criteria "Soglie per mercato/outcome")."""
        by_outcome = self.per_market_outcome.get((market, outcome))
        if by_outcome is not None:
            return by_outcome
        by_market = self.per_market.get(market)
        if by_market is not None:
            return by_market
        return self.default


# Nessun override precaricato (acceptance criteria non richiede una taratura
# specifica per mercato in questo task): la POLICY di default resta
# equivalente a `DEFAULT_POLICY` (BET-02) per ogni mercato finche' un
# override non viene esplicitamente configurato dal chiamante.
DEFAULT_DECISION_POLICY = DecisionPolicy(version="decision_policy_v1")


@dataclass
class Decision:
    """Output STANDARD della valutazione: stesso shape di `ValueDecision`
    (BET-02) + `samples` (nuovo campo, acceptance criteria "Min samples"),
    per restare compatibile con i consumer esistenti (dashboard) mentre si
    espone il dato aggiuntivo."""

    market: str
    outcome: str
    p_model: Optional[float]
    p_market_fair: Optional[float]
    odd: Optional[float]
    samples: int
    prob_edge: Optional[float]
    ev: Optional[float]
    decision: str
    reason: str
    policy_version: str


def evaluate_decision(
    market: str,
    outcome: str,
    p_model: Optional[float],
    p_market_fair: Optional[float],
    odd: Optional[float],
    samples: int = 0,
    policy: DecisionPolicy = DEFAULT_DECISION_POLICY,
) -> Decision:
    """Valuta un outcome applicando le soglie di `policy` PER QUEL
    market/outcome (acceptance criteria "Soglie per mercato/outcome"),
    riusando `compute_prob_edge`/`compute_expected_value` (BET-02, MAI
    duplicati: la formula resta unica in tutto il progetto)."""
    thresholds = policy.thresholds_for(market=market, outcome=outcome)
    prob_edge = compute_prob_edge(p_model, p_market_fair)
    ev = compute_expected_value(p_model, odd)

    if p_model is None:
        decision, reason = NO_BET, "Probabilita' modello non disponibile"
    elif odd is None or float(odd) <= 0.0:
        decision, reason = NO_BET, "Quota non disponibile"
    elif thresholds.min_odd is not None and float(odd) < thresholds.min_odd:
        decision, reason = NO_BET, f"Quota sotto il minimo consentito ({thresholds.min_odd})"
    elif thresholds.max_odd is not None and float(odd) > thresholds.max_odd:
        decision, reason = NO_BET, f"Quota sopra il massimo consentito ({thresholds.max_odd})"
    elif samples < thresholds.min_samples:
        decision, reason = NO_BET, f"Campione insufficiente ({samples} < {thresholds.min_samples})"
    elif (
        p_model >= thresholds.play_min_probability
        and prob_edge is not None
        and prob_edge >= thresholds.play_min_prob_edge
        and ev is not None
        and ev >= thresholds.play_min_ev
    ):
        decision, reason = PLAY, "Confidenza alta e EV positivo"
    elif (
        p_model >= thresholds.borderline_min_probability
        and prob_edge is not None
        and prob_edge >= thresholds.borderline_min_prob_edge
        and ev is not None
        and ev >= thresholds.borderline_min_ev
    ):
        decision, reason = BORDERLINE, "Confidenza media o EV ridotto"
    else:
        decision, reason = NO_BET, "Confidenza/EV insufficienti"

    return Decision(
        market=market,
        outcome=outcome,
        p_model=p_model,
        p_market_fair=p_market_fair,
        odd=odd,
        samples=samples,
        prob_edge=prob_edge,
        ev=ev,
        decision=decision,
        reason=reason,
        policy_version=policy.version,
    )


def evaluate_decision_from_fair_odds_outcome(
    fair_odds_outcome: FairOddsOutcome,
    policy: DecisionPolicy = DEFAULT_DECISION_POLICY,
) -> Decision:
    """Comodo: consuma direttamente `FairOddsOutcome` (BET-01), incluso
    `bookmakers` come `samples` (acceptance criteria "Min samples") - stessa
    garanzia gia' presente in BET-02 che market/outcome/p_model/quota si
    riferiscano allo STESSO outcome."""
    return evaluate_decision(
        market=fair_odds_outcome.market,
        outcome=fair_odds_outcome.outcome,
        p_model=fair_odds_outcome.p_model,
        p_market_fair=fair_odds_outcome.p_market_fair,
        odd=fair_odds_outcome.odd,
        samples=fair_odds_outcome.bookmakers,
        policy=policy,
    )
