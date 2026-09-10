"""Closing Line Value — CLV (BET-05, Fase BETTING).

Il CLV misura quanto la quota presa al momento della decisione
(`odd_at_bet`) sia stata migliore o peggiore della quota di CHIUSURA dello
stesso mercato (`closing_odd`): il valore di mercato subito prima del
kickoff, quando (quasi) ogni informazione pubblica e' gia' incorporata nel
prezzo. E' lo standard "de facto" per valutare se una strategia di betting
ha identificato valore reale, indipendentemente dall'esito della singola
partita (acceptance criteria implicito del task, titolo "Calcolare CLV
quando disponibile").

Metrica di VALUTAZIONE EX-POST (calcolabile SOLO dopo che la vera closing
odd esiste, quindi dopo il kickoff): MAI un input per un modello di
previsione pre-match (acceptance criteria "Nessun uso del closing price
come feature pre-match illegittima") — questo modulo non produce feature
di training, solo un report descrittivo consumato a posteriori.

"Quando disponibile" (titolo del task): il CLV e' calcolabile SOLO per le
fixture per cui esistono `OddsSnapshot` (DATA-06) con `captured_at` fino al
kickoff incluso. Il backtest storico (BET-03) usa invece un singolo valore
quota per fixture (`match.odds`, JSON legacy, senza serie temporale): per
QUELLE bet il CLV resta semplicemente non disponibile (`available=False`),
mai un valore inventato o un fallback silenzioso su un dato che non e'
davvero la chiusura.

Riusa (mai duplica):
- `OddsSnapshotRepository` (DATA-06) per gli snapshot storici nel tempo.
- `build_opening_latest_closing` (EXP-04, `market_odds_expert.py`) per la
  STESSA regola gia' validata "closing valorizzato solo se
  as_of >= kickoff_at": chiamata con `as_of=kickoff_at`, restituisce per
  costruzione l'ultimo snapshot con `captured_at <= kickoff_at` per ogni
  bookmaker/outcome — la vera closing odd pre-match, mai una quota "live"
  successiva al calcio d'inizio.
- `compute_market_baseline`/`get_market_outcome_baseline` (ML-04/BET-01)
  per la fair probability aggregata al closing (overround rimosso), stessa
  identica logica gia' usata per `p_market_fair` in tutto il progetto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from src.ml.baselines.bookmaker_baseline import compute_market_baseline, get_market_outcome_baseline
from src.ml.experts.market.market_odds_expert import build_opening_latest_closing
from src.repository.odds_snapshot_repository import OddsSnapshotRepository


def compute_clv_odd_pct(odd_at_bet: Optional[float], closing_odd: Optional[float]) -> Optional[float]:
    """CLV in termini di quota: `odd_at_bet/closing_odd - 1`.

    Positivo se la quota presa era PIU' ALTA (migliore) della chiusura: il
    mercato si e' mosso a favore di chi ha scommesso prima. `None` se uno
    dei due valori manca o non e' una quota valida (mai un CLV calcolato su
    un dato mancante/inventato)."""
    if odd_at_bet is None or closing_odd is None:
        return None
    try:
        odd_at_bet_f = float(odd_at_bet)
        closing_odd_f = float(closing_odd)
    except (TypeError, ValueError):
        return None
    if odd_at_bet_f <= 0.0 or closing_odd_f <= 0.0:
        return None
    return odd_at_bet_f / closing_odd_f - 1.0


def compute_clv_prob(p_fair_at_bet: Optional[float], p_fair_closing: Optional[float]) -> Optional[float]:
    """CLV in termini di PROBABILITA' fair: `p_fair_at_bet - p_fair_closing`.

    Positivo se il mercato, al momento della bet, dava GIA' una probabilita'
    fair piu' alta di quella confermata alla chiusura (raro/atteso solo per
    rumore); negativo (piu' comune per una buona bet "anticipatrice") se la
    chiusura ha CONFERMATO una probabilita' fair piu' alta di quella
    disponibile al momento della bet. `None` se un valore manca."""
    if p_fair_at_bet is None or p_fair_closing is None:
        return None
    return float(p_fair_at_bet) - float(p_fair_closing)


def fetch_closing_market_baseline(
    fixture_id: int,
    market: str,
    kickoff_at: Optional[datetime],
    period: str = "full_time",
    line: Optional[str] = None,
    snapshot_repo: Optional[OddsSnapshotRepository] = None,
) -> Optional[dict[str, Any]]:
    """Baseline di mercato (stesso shape di `compute_market_baseline`,
    ML-04) calcolata SOLO sugli snapshot di chiusura (l'ultimo per
    bookmaker/outcome con `captured_at <= kickoff_at`).

    `None` quando il CLV non e' calcolabile (acceptance criteria implicito
    "quando disponibile"): kickoff sconosciuto, nessuno snapshot registrato
    per questa fixture/mercato, o nessuno snapshot esiste PRIMA del
    kickoff (solo dati futuri/live, mai usabili come "chiusura")."""
    if kickoff_at is None:
        return None

    repo = snapshot_repo or OddsSnapshotRepository()
    rows = repo.list_for_fixture(fixture_id=fixture_id, market=market, period=period, line=line)
    if not rows:
        return None

    snapshots = [row.to_dict() for row in rows]
    # SQLite (test/locale) perde il tzinfo dei DateTime(timezone=True);
    # normalizziamo entrambe le parti a UTC prima del confronto point-in-time.
    kickoff_at = kickoff_at if kickoff_at.tzinfo else kickoff_at.replace(tzinfo=timezone.utc)
    for snapshot in snapshots:
        captured = snapshot.get("captured_at")
        if isinstance(captured, str):
            captured = datetime.fromisoformat(captured.replace("Z", "+00:00"))
        if isinstance(captured, datetime) and captured.tzinfo is None:
            captured = captured.replace(tzinfo=timezone.utc)
        if isinstance(captured, datetime):
            snapshot["captured_at"] = captured.isoformat()
    # as_of=kickoff_at: riusa EXP-04 con la STESSA regola gia' validata
    # (closing_allowed = kickoff_at is not None and as_of >= kickoff_at,
    # qui SEMPRE vera per costruzione) — "closing" per ogni bookmaker/outcome
    # e' per costruzione l'ultimo snapshot con captured_at <= kickoff_at.
    per_bookmaker = build_opening_latest_closing(snapshots, as_of=kickoff_at, kickoff_at=kickoff_at)

    closing_odds_by_outcome: dict[str, list[float]] = {}
    for row in per_bookmaker:
        closing = row.get("closing")
        if not closing or closing.get("odd") is None:
            continue
        try:
            odd_value = float(closing["odd"])
        except (TypeError, ValueError):
            continue
        if odd_value <= 0.0:
            continue
        closing_odds_by_outcome.setdefault(row["outcome"], []).append(odd_value)

    if not closing_odds_by_outcome:
        return None

    odds_rows = [
        {"outcome": outcome, "avg_odd": float(np.mean(odds)), "bookmakers": len(odds)}
        for outcome, odds in closing_odds_by_outcome.items()
    ]
    return compute_market_baseline(market=market, odds_rows=odds_rows)


@dataclass
class ClvResult:
    """Output STANDARD per una singola bet/prediction (stesso stile
    "sempre gli stessi campi, None se non disponibile" di `FairOddsOutcome`
    BET-01 e `ValueDecision` BET-02)."""

    market: str
    outcome: str
    fixture_id: Optional[int]
    model_name: Optional[str]
    odd_at_bet: Optional[float]
    p_fair_at_bet: Optional[float]
    closing_odd: Optional[float]
    p_fair_closing: Optional[float]
    clv_odd_pct: Optional[float]
    clv_prob: Optional[float]
    available: bool
    reason: str


def build_clv_result(
    market: str,
    outcome: str,
    odd_at_bet: Optional[float],
    p_fair_at_bet: Optional[float],
    fixture_id: int,
    kickoff_at: Optional[datetime],
    period: str = "full_time",
    line: Optional[str] = None,
    model_name: Optional[str] = None,
    snapshot_repo: Optional[OddsSnapshotRepository] = None,
) -> ClvResult:
    """Costruisce il CLV per un singolo outcome, recuperando la closing
    odd/fair probability dagli `OddsSnapshot` (DATA-06). Non solleva mai
    un'eccezione per dati mancanti: `available=False` con `reason`
    esplicito (stesso pattern di fallback non silenzioso gia' in uso in
    tutto il progetto)."""
    closing_baseline = fetch_closing_market_baseline(
        fixture_id=fixture_id, market=market, kickoff_at=kickoff_at, period=period, line=line, snapshot_repo=snapshot_repo
    )
    if closing_baseline is None:
        return ClvResult(
            market=market, outcome=outcome, fixture_id=fixture_id, model_name=model_name,
            odd_at_bet=odd_at_bet, p_fair_at_bet=p_fair_at_bet,
            closing_odd=None, p_fair_closing=None, clv_odd_pct=None, clv_prob=None,
            available=False, reason="Closing odd non disponibile (nessuno snapshot pre-kickoff per questa fixture)",
        )

    closing_row = get_market_outcome_baseline(
        fixture_baseline={"markets": {market: closing_baseline}}, market=market, outcome=outcome
    )
    if closing_row is None:
        return ClvResult(
            market=market, outcome=outcome, fixture_id=fixture_id, model_name=model_name,
            odd_at_bet=odd_at_bet, p_fair_at_bet=p_fair_at_bet,
            closing_odd=None, p_fair_closing=None, clv_odd_pct=None, clv_prob=None,
            available=False, reason=f"Outcome '{outcome}' non quotato al closing",
        )

    closing_odd = closing_row.get("avg_odd")
    p_fair_closing = closing_row.get("fair_probability")
    clv_odd_pct = compute_clv_odd_pct(odd_at_bet, closing_odd)
    clv_prob = compute_clv_prob(p_fair_at_bet, p_fair_closing)
    available = clv_odd_pct is not None or clv_prob is not None

    return ClvResult(
        market=market,
        outcome=outcome,
        fixture_id=fixture_id,
        model_name=model_name,
        odd_at_bet=odd_at_bet,
        p_fair_at_bet=p_fair_at_bet,
        closing_odd=closing_odd,
        p_fair_closing=p_fair_closing,
        clv_odd_pct=clv_odd_pct,
        clv_prob=clv_prob,
        available=available,
        reason="" if available else "Dati insufficienti per calcolare il CLV (odd_at_bet/p_fair_at_bet mancanti)",
    )


@dataclass
class ClvReportRow:
    """Aggregazione STANDARD (stesse chiavi per overall/per-market/per-model,
    stesso stile di `BacktestStats` in BET-03)."""

    count: int
    avg_clv_odd_pct: Optional[float]
    avg_clv_prob: Optional[float]
    positive_clv_rate: Optional[float]


def _report_row_from_results(results: list[ClvResult]) -> ClvReportRow:
    if not results:
        return ClvReportRow(count=0, avg_clv_odd_pct=None, avg_clv_prob=None, positive_clv_rate=None)

    odd_pct_values = [r.clv_odd_pct for r in results if r.clv_odd_pct is not None]
    prob_values = [r.clv_prob for r in results if r.clv_prob is not None]

    return ClvReportRow(
        count=len(results),
        avg_clv_odd_pct=float(np.mean(odd_pct_values)) if odd_pct_values else None,
        avg_clv_prob=float(np.mean(prob_values)) if prob_values else None,
        positive_clv_rate=(sum(1 for v in odd_pct_values if v > 0) / len(odd_pct_values)) if odd_pct_values else None,
    )


@dataclass
class ClvReport:
    """Report per modello/mercato (acceptance criteria "Report per
    modello/mercato"): overall + breakdown, con conteggio esplicito di
    quante bet avevano CLV disponibile/non disponibile (mai nascosto)."""

    overall: ClvReportRow
    by_market: dict[str, ClvReportRow]
    by_model: dict[str, ClvReportRow]
    total_considered: int
    available_count: int
    unavailable_count: int


def compute_clv_report(results: list[ClvResult]) -> ClvReport:
    """Aggrega una lista di `ClvResult` (ordine qualunque) in un report per
    mercato/modello. Solo le righe con CLV disponibile (`available=True`)
    contribuiscono alle medie, ma il conteggio totale/non disponibili resta
    sempre esplicito (acceptance criteria implicito di trasparenza, stesso
    principio di `BacktestReport.skipped_bets` in BET-03)."""
    available = [r for r in results if r.available]

    by_market: dict[str, list[ClvResult]] = {}
    by_model: dict[str, list[ClvResult]] = {}
    for result in available:
        by_market.setdefault(result.market, []).append(result)
        model_key = result.model_name or "unknown"
        by_model.setdefault(model_key, []).append(result)

    return ClvReport(
        overall=_report_row_from_results(available),
        by_market={key: _report_row_from_results(value) for key, value in sorted(by_market.items())},
        by_model={key: _report_row_from_results(value) for key, value in sorted(by_model.items())},
        total_considered=len(results),
        available_count=len(available),
        unavailable_count=len(results) - len(available),
    )
