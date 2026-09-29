"""Fair Odds Engine (BET-01, Fase BETTING).

Per ciascun outcome di un mercato, standardizza il confronto tra:
- la probabilita' del bookmaker: `p_market_raw` (implied probability dalla
  quota media) e `p_market_fair` (overround rimosso) — gia' calcolate da
  `src/ml/baselines/bookmaker_baseline.py` (ML-04), qui RIUSATE senza
  duplicare la logica di rimozione overround;
- `fair_odd = 1 / p_market_fair`;
- la probabilita' dell'Oracle (`p_model`), quando disponibile (tipicamente
  da `src/ml/ensemble/model_consensus.py`, ORACLE-04, o da un qualunque
  Direct Expert/meta-model gia' calcolato a monte).

Scope DELIBERATAMENTE limitato (dipendenza tecnica di BET-02, "non
anticipare task successivi"): questo modulo produce SOLO probabilita'/quote
per outcome. NON calcola `prob_edge`/`EV`/`decision` (Value Engine, BET-02,
REWRITE) ne' sostituisce `DashboardService._value_decision` (quello e'
esplicitamente compito di BET-02, che dipende da questo modulo).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.ml.baselines.bookmaker_baseline import compute_market_baseline


@dataclass
class FairOddsOutcome:
    """Output STANDARD per outcome (acceptance criteria "Output standard
    per outcome"): stesse chiavi per QUALUNQUE mercato/outcome, anche
    quando un valore non e' disponibile (mai un campo mancante, semmai
    `None` — nessuna spiegazione/valore inventato)."""

    market: str
    outcome: str
    odd: Optional[float]
    p_market_raw: Optional[float]
    p_market_fair: Optional[float]
    fair_odd: Optional[float]
    p_model: Optional[float]
    bookmakers: int = 0


def fair_odd_from_probability(probability: Optional[float]) -> Optional[float]:
    """fair_odd = 1/p (acceptance criteria "fair odd = 1/p").

    `None` se la probabilita' non e' disponibile o non e' positiva (mai una
    divisione per zero/quota infinita spacciata per un numero valido)."""
    if probability is None:
        return None
    try:
        value = float(probability)
    except (TypeError, ValueError):
        return None
    if value <= 0.0:
        return None
    return 1.0 / value


def build_fair_odds_outcome(
    market: str,
    outcome: str,
    market_baseline_row: Optional[dict[str, Any]],
    p_model: Optional[float] = None,
) -> FairOddsOutcome:
    """Costruisce l'output standard per UN outcome da una riga gia'
    prodotta da `compute_market_baseline`/`build_fixture_baseline` (ML-04).

    `market_baseline_row` puo' essere `None` (nessuna quota disponibile per
    questo outcome): in quel caso `p_market_raw`/`p_market_fair`/`fair_odd`
    restano `None`, ma se `p_model` e' comunque noto il confronto resta
    parzialmente disponibile (Oracle senza controparte di mercato) invece
    di sparire silenziosamente.
    """
    row = market_baseline_row or {}
    p_market_fair = row.get("fair_probability")
    return FairOddsOutcome(
        market=market,
        outcome=outcome,
        odd=row.get("avg_odd"),
        p_market_raw=row.get("implied_raw"),
        p_market_fair=p_market_fair,
        fair_odd=fair_odd_from_probability(p_market_fair),
        p_model=p_model,
        bookmakers=int(row.get("bookmakers") or 0),
    )


def build_fair_odds_for_market(
    market: str,
    odds_rows: list[dict[str, Any]],
    p_model_by_outcome: Optional[dict[str, float]] = None,
) -> list[FairOddsOutcome]:
    """Confronto Oracle vs market (acceptance criteria) per TUTTI gli
    outcome quotati di un mercato: una riga per outcome, `p_model` allineato
    quando il chiamante lo fornisce per quello stesso outcome."""
    baseline = compute_market_baseline(market=market, odds_rows=odds_rows)
    p_model_by_outcome = p_model_by_outcome or {}

    outcomes: list[FairOddsOutcome] = [
        build_fair_odds_outcome(
            market=market,
            outcome=row["outcome"],
            market_baseline_row=row,
            p_model=p_model_by_outcome.get(row["outcome"]),
        )
        for row in baseline.get("outcomes", [])
    ]

    # Outcome con p_model noto ma SENZA quota di mercato (es. Oracle calcola
    # un mercato/linea non quotata da alcun bookmaker): incluso comunque,
    # mai scartato solo perche' manca la controparte di mercato.
    quoted_outcomes = {item.outcome for item in outcomes}
    for outcome, p_model in p_model_by_outcome.items():
        if outcome in quoted_outcomes:
            continue
        outcomes.append(
            build_fair_odds_outcome(market=market, outcome=outcome, market_baseline_row=None, p_model=p_model)
        )

    return outcomes


def build_fair_odds_for_fixture(
    odds_summary: dict[str, list[dict[str, Any]]],
    p_model_by_market_outcome: Optional[dict[str, dict[str, float]]] = None,
) -> dict[str, list[FairOddsOutcome]]:
    """Confronto Oracle vs market per TUTTI i mercati di una fixture, dallo
    stesso `odds_summary` gia' prodotto da `DashboardService`
    (`_aggregate_odds_from_api`/`_aggregate_odds_from_db`): nessuna nuova
    query, nessuna nuova estrazione feature."""
    p_model_by_market_outcome = p_model_by_market_outcome or {}
    result: dict[str, list[FairOddsOutcome]] = {}

    markets = set(odds_summary or {}) | set(p_model_by_market_outcome)
    for market in markets:
        rows = (odds_summary or {}).get(market) or []
        if not isinstance(rows, list):
            rows = []
        result[market] = build_fair_odds_for_market(
            market=market, odds_rows=rows, p_model_by_outcome=p_model_by_market_outcome.get(market)
        )

    return result
