"""Betting Backtester (BET-03, Fase BETTING) — parte "pura".

Data una sequenza di bet gia' valutate (`BacktestBet`, tipicamente costruite
da `market_backtest.py` con probabilita' OUT-OF-SAMPLE e quote storiche reali),
calcola le metriche di performance di uno stake flat (acceptance criteria):

- ROI / yield / profit
- hit rate
- avg odds
- max drawdown
- performance per edge bucket / mercato / lega

Questo modulo NON tocca DB ne' modelli ML: e' deliberatamente una libreria di
sole funzioni pure (stesso stile di `fair_odds_engine.py` BET-01 e
`value_engine.py` BET-02), cosi' da poter essere testato con numeri fissi
("Test numerici", acceptance criteria) senza alcuna dipendenza esterna.

Riusa BET-01/BET-02 SENZA modificarli: `build_backtest_bet` chiama
direttamente `evaluate_value` (BET-02) per ottenere `prob_edge`/`ev`/
`decision`, garantendo per costruzione la stessa distinzione edge/EV e la
stessa gestione della quota mancante gia' validate in BET-02.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from src.oracle.value_engine.value_engine import (
    DEFAULT_POLICY,
    PLAY,
    ValueDecisionPolicy,
    evaluate_value,
)

# "Stake flat iniziale" (acceptance criteria): stake costante per bet, non
# proporzionale/Kelly. Una taratura futura dello staking e' esplicitamente
# fuori scope (BET-03 si limita a misurare la performance, non a ottimizzare
# la size della bet).
STAKE_DEFAULT = 1.0

# Di default il backtest piazza SOLO le bet con decisione PLAY (BET-02): il
# segnale piu' forte. BORDERLINE/NO BET restano fuori dal calcolo ROI a meno
# che il chiamante non allarghi esplicitamente `include_decisions`.
DEFAULT_PLACED_DECISIONS: frozenset[str] = frozenset({PLAY})

# Bucket di prob_edge (BET-02) di default per la scomposizione "per edge
# bucket" richiesta dall'acceptance criteria. Confini scelti come pura
# convenzione di reporting (percentuali di edge "basso/medio/alto"): NON è
# una policy di decisione (quella resta BET-04), solo una chiave di
# raggruppamento per il report.
DEFAULT_EDGE_BUCKET_EDGES: tuple[float, ...] = (0.0, 0.03, 0.06, 0.10)


@dataclass
class BacktestBet:
    """Una singola bet storica gia' valutata (acceptance criteria "Usare
    outcome corretto": market/outcome/p_model/p_market_fair/odd si
    riferiscono sempre allo stesso outcome, stessa garanzia di BET-01/02)."""

    market: str
    outcome: str
    p_model: Optional[float]
    p_market_fair: Optional[float]
    odd: Optional[float]
    prob_edge: Optional[float]
    ev: Optional[float]
    decision: str
    policy_version: str
    won: Optional[bool]
    fixture_id: Optional[int] = None
    league: Optional[Any] = None
    season: Optional[Any] = None
    kickoff_at: Optional[str] = None


def build_backtest_bet(
    market: str,
    outcome: str,
    p_model: Optional[float],
    p_market_fair: Optional[float],
    odd: Optional[float],
    won: Optional[bool],
    fixture_id: Optional[int] = None,
    league: Optional[Any] = None,
    season: Optional[Any] = None,
    kickoff_at: Optional[str] = None,
    policy: ValueDecisionPolicy = DEFAULT_POLICY,
) -> BacktestBet:
    """Costruisce una `BacktestBet` riusando DIRETTAMENTE `evaluate_value`
    (BET-02, non duplicato): edge/EV/decisione calcolati esattamente come nel
    resto del progetto (dashboard compresa)."""
    decision = evaluate_value(
        market=market,
        outcome=outcome,
        p_model=p_model,
        p_market_fair=p_market_fair,
        odd=odd,
        policy=policy,
    )
    return BacktestBet(
        market=market,
        outcome=outcome,
        p_model=p_model,
        p_market_fair=p_market_fair,
        odd=odd,
        prob_edge=decision.prob_edge,
        ev=decision.ev,
        decision=decision.decision,
        policy_version=decision.policy_version,
        won=won,
        fixture_id=fixture_id,
        league=league,
        season=season,
        kickoff_at=kickoff_at,
    )


def bet_profit(bet: BacktestBet, stake: float = STAKE_DEFAULT) -> Optional[float]:
    """Risultato IN UNITA' DI STAKE di una singola bet con stake flat:
    `stake*(odd-1)` se vinta, `-stake` se persa.

    Gestisce esplicitamente quota/esito mancante (acceptance criteria
    "Gestire quota mancante"): `None`, mai un profitto fittizio."""
    if bet.odd is None or bet.won is None:
        return None
    try:
        odd_value = float(bet.odd)
    except (TypeError, ValueError):
        return None
    if odd_value <= 0.0:
        return None
    return float(stake) * (odd_value - 1.0) if bet.won else -float(stake)


def _is_placed(bet: BacktestBet, include_decisions: frozenset[str]) -> bool:
    """Una bet e' "piazzata" (e quindi entra nel calcolo ROI) solo se la
    decisione rientra tra quelle incluse E odd/won sono entrambi noti."""
    return bet.decision in include_decisions and bet.odd is not None and bet.won is not None


def max_drawdown_from_cumulative(cumulative_profit: list[float]) -> float:
    """Max drawdown STANDARD (differenza peak-to-trough) sulla curva di
    profitto cumulato, in unita' di stake. Sempre >= 0, sempre definito
    (nessuna divisione, nessun concetto di 'bankroll' richiesto)."""
    peak = float("-inf")
    worst = 0.0
    for value in cumulative_profit:
        peak = max(peak, value)
        worst = max(worst, peak - value)
    return float(worst)


def edge_bucket_label(prob_edge: Optional[float], edges: tuple[float, ...] = DEFAULT_EDGE_BUCKET_EDGES) -> str:
    """Etichetta leggibile e deterministica per il bucket di `prob_edge`
    (acceptance criteria "Performance per edge bucket")."""
    if prob_edge is None:
        return "unknown"

    sorted_edges = sorted(set(float(e) for e in edges))
    value = float(prob_edge)
    if not sorted_edges:
        return "all"
    if value < sorted_edges[0]:
        return f"<{sorted_edges[0]:.2f}"
    for lower, upper in zip(sorted_edges[:-1], sorted_edges[1:]):
        if lower <= value < upper:
            return f"[{lower:.2f},{upper:.2f})"
    return f">={sorted_edges[-1]:.2f}"


@dataclass
class BacktestStats:
    """Metriche STANDARD (stesse chiavi per overall e per ciascun
    breakdown): ROI/yield/profit distinti (con stake flat sono
    numericamente equivalenti a meno di scala %, ma restano campi separati
    per chiarezza di reporting), hit rate, avg odds, max drawdown."""

    bets: int
    wins: int
    losses: int
    stake_per_bet: float
    total_staked: float
    profit: float
    roi: Optional[float]
    yield_pct: Optional[float]
    hit_rate: Optional[float]
    avg_odds: Optional[float]
    avg_prob_edge: Optional[float]
    max_drawdown: float


def _empty_stats(stake: float) -> BacktestStats:
    return BacktestStats(
        bets=0,
        wins=0,
        losses=0,
        stake_per_bet=float(stake),
        total_staked=0.0,
        profit=0.0,
        roi=None,
        yield_pct=None,
        hit_rate=None,
        avg_odds=None,
        avg_prob_edge=None,
        max_drawdown=0.0,
    )


def _stats_from_placed_bets(placed: list[BacktestBet], stake: float) -> BacktestStats:
    if not placed:
        return _empty_stats(stake)

    # Ordine cronologico deterministico (kickoff_at, poi fixture_id come
    # tie-break stabile): la curva di equity/drawdown e le statistiche
    # aggregate NON dipendono dall'ordine con cui il chiamante ha passato le
    # bet (acceptance criteria "Report riproducibile").
    ordered = sorted(
        placed,
        key=lambda item: (item.kickoff_at or "", item.fixture_id if item.fixture_id is not None else 0),
    )

    results = [bet_profit(bet, stake) for bet in ordered]
    # _is_placed garantisce odd/won noti: bet_profit non dovrebbe restituire
    # None qui, ma un eventuale odd non positivo viene comunque escluso senza
    # sollevare eccezioni (difesa aggiuntiva, mai un profitto fittizio).
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return _empty_stats(stake)

    n = len(valid_results)
    wins = sum(1 for bet, r in zip(ordered, results) if r is not None and bet.won)
    profit = float(sum(valid_results))
    total_staked = float(stake) * n

    cumulative: list[float] = []
    running = 0.0
    for r in valid_results:
        running += r
        cumulative.append(running)

    odds_values = [float(bet.odd) for bet, r in zip(ordered, results) if r is not None]
    edge_values = [float(bet.prob_edge) for bet, r in zip(ordered, results) if r is not None and bet.prob_edge is not None]

    return BacktestStats(
        bets=n,
        wins=wins,
        losses=n - wins,
        stake_per_bet=float(stake),
        total_staked=total_staked,
        profit=profit,
        roi=(profit / total_staked) if total_staked > 0 else None,
        yield_pct=(profit / total_staked * 100.0) if total_staked > 0 else None,
        hit_rate=(wins / n) if n > 0 else None,
        avg_odds=(sum(odds_values) / len(odds_values)) if odds_values else None,
        avg_prob_edge=(sum(edge_values) / len(edge_values)) if edge_values else None,
        max_drawdown=max_drawdown_from_cumulative(cumulative),
    )


@dataclass
class BacktestReport:
    """Report riproducibile (acceptance criteria): overall + breakdown per
    edge bucket/mercato/lega, con conteggio esplicito di quante bet sono
    state escluse (e perche' restano escluse e' desumibile da `decision`
    sulle singole `BacktestBet`, non nascosto)."""

    overall: BacktestStats
    by_edge_bucket: dict[str, BacktestStats]
    by_market: dict[str, BacktestStats]
    by_league: dict[str, BacktestStats]
    total_bets_considered: int
    placed_bets: int
    skipped_bets: int
    stake_per_bet: float
    included_decisions: list[str] = field(default_factory=list)


def compute_backtest_report(
    bets: list[BacktestBet],
    stake: float = STAKE_DEFAULT,
    include_decisions: frozenset[str] = DEFAULT_PLACED_DECISIONS,
    edge_bucket_edges: tuple[float, ...] = DEFAULT_EDGE_BUCKET_EDGES,
) -> BacktestReport:
    """Aggrega una lista di `BacktestBet` (ordine qualunque: il risultato non
    dipende dall'ordine di input, acceptance criteria "Report riproducibile")
    nel report di performance flat-stake."""
    placed = [bet for bet in bets if _is_placed(bet, include_decisions)]

    by_edge_bucket: dict[str, list[BacktestBet]] = {}
    by_market: dict[str, list[BacktestBet]] = {}
    by_league: dict[str, list[BacktestBet]] = {}

    for bet in placed:
        by_edge_bucket.setdefault(edge_bucket_label(bet.prob_edge, edge_bucket_edges), []).append(bet)
        by_market.setdefault(bet.market, []).append(bet)
        league_key = "unknown" if bet.league is None else str(bet.league)
        by_league.setdefault(league_key, []).append(bet)

    return BacktestReport(
        overall=_stats_from_placed_bets(placed, stake),
        by_edge_bucket={key: _stats_from_placed_bets(value, stake) for key, value in sorted(by_edge_bucket.items())},
        by_market={key: _stats_from_placed_bets(value, stake) for key, value in sorted(by_market.items())},
        by_league={key: _stats_from_placed_bets(value, stake) for key, value in sorted(by_league.items())},
        total_bets_considered=len(bets),
        placed_bets=len(placed),
        skipped_bets=len(bets) - len(placed),
        stake_per_bet=float(stake),
        included_decisions=sorted(include_decisions),
    )


def persist_backtest_report(
    market: str,
    report: BacktestReport,
    output_dir: str = os.path.join("best_models", "backtests"),
) -> str:
    """Persiste il report su disco (stesso pattern di `persist_fixture_baseline`,
    BET-01): utile per ispezione/riproducibilita', nessun impatto sul calcolo."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.abspath(os.path.join(output_dir, f"{market}_backtest.json"))
    with open(file_path, "w", encoding="utf-8") as file_handle:
        json.dump(dataclasses.asdict(report), file_handle, ensure_ascii=False, indent=2)
    return file_path
