"""Prediction Ledger / Paper Betting (BET-06, Fase BETTING).

Principio guida (`06_BETTING_INTELLIGENCE.md`, "Paper Betting"): "Ogni
prediction deve essere salvata PRIMA del kickoff e poi settled SENZA
ALTERARE il record originale". Questo modulo e' la parte "pura" (nessun
DB, stesso stile di `fair_odds_engine.py`/`value_engine.py`/
`betting_backtester.py`): rappresenta una prediction come `PredictionRecord`
e separa ESPLICITAMENTE:

- i campi "originali" scritti UNA VOLA (al momento della creazione, PRIMA
  del kickoff): market/outcome/model_run_id/p_model/p_market_fair/odd/
  fair_odd/prob_edge/ev/decision/policy_version/stake/kickoff_at/created_at;
- i campi di SETTLEMENT (scritti SOLO DOPO, quando il risultato reale e'
  noto): is_settled/settled_at/actual_outcome/won/pnl/settlement_status.

`settle_prediction_record` non modifica MAI in-place i campi originali di
un `PredictionRecord` (immutabilita' logica): usa `dataclasses.replace` per
produrre un NUOVO record con solo i campi di settlement aggiornati.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.oracle.decision_engine.decision_policy import Decision
from src.oracle.ledger.settlement_rules import outcome_wins, resolve_settlement

# Stake "paper" di default: stessa convenzione di STAKE_DEFAULT (BET-03),
# qui duplicata come costante indipendente perche' il ledger e' un modulo
# concettualmente distinto dal backtest storico (prediction realtime vs
# ricostruzione OOF) - stesso VALORE (1.0), nessun cambio di convenzione.
DEFAULT_STAKE = 1.0

# Stessi motivi di "void" gia' visti nel resto della Fase BETTING (BET-01/02:
# "mai un valore fittizio, semmai un motivo esplicito"): un settlement puo'
# non produrre un PnL numerico se manca la quota o se l'esito reale non e'
# determinabile dai dati disponibili (es. statistiche incomplete).
SETTLED = "settled"  # compatibilità lettura record legacy
SETTLED_WIN = "settled_win"
SETTLED_LOSS = "settled_loss"
VOID_CANCELLED = "void_cancelled"
VOID_POSTPONED = "void_postponed"
VOID_ABANDONED = "void_abandoned"
VOID_PUSH = "void_push"
VOID_NO_RESULT = "void_no_result"
VOID_MARKET_RULE = "void_market_rule"
NOT_PLACED_MISSING_ODD = "not_placed_missing_odd"
# Alias di import per compatibilità con consumer legacy; il valore semantico
# non è più un VOID.
VOID_MISSING_ODD = NOT_PLACED_MISSING_ODD
VOID_STATUSES = frozenset({
    VOID_CANCELLED,
    VOID_POSTPONED,
    VOID_ABANDONED,
    VOID_PUSH,
    VOID_NO_RESULT,
    VOID_MARKET_RULE,
})


@dataclass
class PredictionRecord:
    """Rappresentazione PURA di una riga del Prediction Ledger (nessuna
    dipendenza da SQLAlchemy): stesse chiavi della tabella `prediction_ledger`
    (`src/service_ia/model/match.py::PredictionLedger`)."""

    fixture_id: int
    market: str
    outcome: str
    decision: str
    stake: float = DEFAULT_STAKE
    model_run_id: Optional[str] = None
    model_name: Optional[str] = None
    policy_version: Optional[str] = None
    p_model: Optional[float] = None
    p_market_fair: Optional[float] = None
    odd: Optional[float] = None
    fair_odd: Optional[float] = None
    model_void_odd: Optional[float] = None
    market_fair_odd: Optional[float] = None
    odds_edge_absolute: Optional[float] = None
    odds_edge_percent: Optional[float] = None
    prob_edge: Optional[float] = None
    ev: Optional[float] = None
    expected_roi_percent: Optional[float] = None
    play_threshold_odd: Optional[float] = None
    min_edge_percent: Optional[float] = None
    value_label: Optional[str] = None
    value_reason: Optional[str] = None
    p_market_raw: Optional[float] = None
    period: str = "full_time"
    line: Optional[str] = None
    source: str = "manual"
    cohort: str = "manual"
    captured_at: Optional[datetime] = None
    odds_captured_at: Optional[datetime] = None
    bookmaker_count: int = 0
    league: Optional[str] = None
    capture_key: Optional[str] = None
    kickoff_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    id_prediction: Optional[str] = None

    # Settlement (mai popolati alla creazione)
    is_settled: bool = False
    settled_at: Optional[datetime] = None
    settlement_status: Optional[str] = None
    actual_outcome: Optional[str] = None
    won: Optional[bool] = None
    pnl: Optional[float] = None


def build_prediction_record(
    fixture_id: int,
    decision: Decision,
    model_run_id: Optional[str] = None,
    model_name: Optional[str] = None,
    kickoff_at: Optional[datetime] = None,
    stake: float = DEFAULT_STAKE,
    p_market_raw: Optional[float] = None,
    period: str = "full_time",
    line: Optional[str] = None,
    source: str = "manual",
    cohort: str = "manual",
    captured_at: Optional[datetime] = None,
    odds_captured_at: Optional[datetime] = None,
    bookmaker_count: int = 0,
    league: Optional[str] = None,
    capture_key: Optional[str] = None,
) -> PredictionRecord:
    """Costruisce un `PredictionRecord` riusando DIRETTAMENTE l'output gia'
    calcolato da `evaluate_decision`/`evaluate_decision_from_fair_odds_outcome`
    (BET-04, mai duplicato): market/outcome/p_model/p_market_fair/odd/
    prob_edge/ev/decision/policy_version provengono TUTTI dalla stessa
    `Decision`, garanzia che si riferiscano allo stesso outcome (stessa
    garanzia gia' presente in BET-01/02/04)."""
    from src.oracle.fair_odds.fair_odds_engine import fair_odd_from_probability

    captured_at = captured_at or datetime.now(timezone.utc)
    return PredictionRecord(
        fixture_id=int(fixture_id),
        market=decision.market,
        outcome=decision.outcome,
        decision=decision.decision,
        stake=float(stake),
        model_run_id=model_run_id,
        model_name=model_name,
        policy_version=decision.policy_version,
        p_model=decision.p_model,
        p_market_fair=decision.p_market_fair,
        odd=decision.odd,
        fair_odd=fair_odd_from_probability(decision.p_market_fair),
        model_void_odd=decision.model_void_odd,
        market_fair_odd=decision.market_fair_odd,
        odds_edge_absolute=decision.odds_edge_absolute,
        odds_edge_percent=decision.odds_edge_percent,
        prob_edge=decision.prob_edge,
        ev=decision.ev,
        expected_roi_percent=decision.expected_roi_percent,
        play_threshold_odd=decision.play_threshold_odd,
        min_edge_percent=decision.min_edge_percent,
        value_label=decision.decision,
        value_reason=decision.reason,
        p_market_raw=p_market_raw,
        period=period,
        line=line,
        source=source,
        cohort=cohort,
        captured_at=captured_at,
        odds_captured_at=odds_captured_at,
        bookmaker_count=int(bookmaker_count),
        league=league,
        capture_key=capture_key,
        kickoff_at=kickoff_at,
        created_at=captured_at,
    )


def resolve_actual_outcome(market: str, stat_home: Optional[dict], stat_away: Optional[dict]) -> Optional[str]:
    """Esito REALE (nome outcome canonico) di `market` dato il risultato
    finale del match, riusando SENZA duplicare:
    - `FilterMarketService._label_by_market` (gia' validato dal training:
      stessa etichetta 0/1 usata per costruire `y`);
    - `_canonical_outcome_for_prediction` (BET-03: stessa mappa 0/1 -> nome
      outcome gia' usata dal backtest storico, cosi' un outcome "Over 2.5"
      salvato dal ledger e uno prodotto dal backtest sono SEMPRE confrontabili).

    `None` se il risultato non e' determinabile (punteggi mancanti,
    mercato non supportato): mai un esito inventato."""
    if not stat_home or not stat_away:
        return None
    resolution = resolve_settlement(
        market=market,
        match_status="FT",
        home_score=stat_home.get("score_ft"),
        away_score=stat_away.get("score_ft"),
        stat_home=stat_home,
        stat_away=stat_away,
    )
    return resolution.actual_outcome


def compute_paper_pnl(odd: Optional[float], won: Optional[bool], stake: float) -> Optional[float]:
    """PnL "paper" con stake flat: `stake*(odd-1)` se vinta, `-stake` se
    persa. STESSA formula di `bet_profit` (BET-03) - duplicata qui (3 righe)
    per mantenere questo modulo indipendente dal backtest storico (concetti
    distinti: paper trading realtime vs ricostruzione OOF), coerenza
    numerica garantita da `tests/service/prediction_ledger_test.py`.

    Gestisce esplicitamente quota/esito mancante: `None`, mai un profitto
    fittizio (stesso principio BET-02/03)."""
    if odd is None or won is None:
        return None
    try:
        odd_value = float(odd)
    except (TypeError, ValueError):
        return None
    if odd_value <= 0.0:
        return None
    return float(stake) * (odd_value - 1.0) if won else -float(stake)


def settle_prediction_record(
    record: PredictionRecord,
    actual_outcome: Optional[str],
    settled_at: Optional[datetime] = None,
    void_status: Optional[str] = None,
    is_push: bool = False,
) -> PredictionRecord:
    """Produce un NUOVO `PredictionRecord` con i campi di settlement
    popolati, senza mai modificare in-place i campi originali (acceptance
    criteria "Immutabilita' logica della prediction originale") - lo stesso
    `dataclasses.replace` garantisce per costruzione che market/outcome/
    p_model/odd/decision/... restino IDENTICI all'oggetto di partenza.

    - `actual_outcome=None` (risultato non determinabile): settlement_status
      = VOID_NO_RESULT, `won`/`pnl` restano `None` (mai un esito inventato).
    - `record.odd is None` (nessuna quota disponibile al momento della
      prediction): settlement_status = VOID_MISSING_ODD, `won` e' comunque
      calcolabile (utile per l'hit-rate) ma `pnl` resta `None`.
    - Altrimenti: SETTLED, `won` = confronto outcome predetto vs reale,
      `pnl` = `compute_paper_pnl`.
    """
    settled_at = settled_at or datetime.now(timezone.utc)

    if void_status or is_push:
        status = VOID_PUSH if is_push else str(void_status)
        return dataclasses.replace(
            record,
            is_settled=True,
            settled_at=settled_at,
            settlement_status=status,
            actual_outcome=actual_outcome,
            won=None,
            pnl=0.0,
        )

    if actual_outcome is None:
        return dataclasses.replace(
            record,
            is_settled=True,
            settled_at=settled_at,
            settlement_status=VOID_NO_RESULT,
            actual_outcome=None,
            won=None,
            pnl=0.0,
        )

    won = outcome_wins(record.market, record.outcome, actual_outcome)
    if record.odd is None:
        return dataclasses.replace(
            record,
            is_settled=True,
            settled_at=settled_at,
            settlement_status=NOT_PLACED_MISSING_ODD,
            actual_outcome=actual_outcome,
            won=None,
            pnl=0.0,
        )
    if won is None:
        return dataclasses.replace(
            record,
            is_settled=True,
            settled_at=settled_at,
            settlement_status=VOID_MARKET_RULE,
            actual_outcome=actual_outcome,
            won=None,
            pnl=0.0,
        )

    pnl = compute_paper_pnl(odd=record.odd, won=won, stake=record.stake)
    return dataclasses.replace(
        record,
        is_settled=True,
        settled_at=settled_at,
        settlement_status=SETTLED_WIN if won else SETTLED_LOSS,
        actual_outcome=actual_outcome,
        won=won,
        pnl=pnl,
    )
