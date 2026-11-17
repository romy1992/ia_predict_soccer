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

from src.ml.baselines.bookmaker_baseline import (
    _normalize_outcome_for_market,
    _normalize_text,
    compute_market_baseline,
    get_market_outcome_baseline,
)
from src.ml.experts.market.market_odds_expert import build_opening_latest_closing
from src.repository.odds_snapshot_repository import OddsSnapshotRepository

# Mercato degli `OddsSnapshot` su cui e' prezzata una decisione del ledger,
# quando il nome della decisione differisce da quello degli snapshot: la
# cattura ufficiale salva il vero 1X2 come `market="1x2"`
# (`official_capture_service._capture_true_1x2`), ma le quote "Match Winner"
# sono salvate come `h2h` (`download_match_service._switch_market_name`).
# Senza questa corrispondenza nessuna PLAY 1X2 avrebbe mai una chiusura.
_PRICING_MARKET_ALIASES: dict[str, str] = {"1x2": "h2h"}


def pricing_market_for(market: str) -> str:
    """Nome del mercato negli `OddsSnapshot` per una decisione `market`."""
    value = str(market or "")
    return _PRICING_MARKET_ALIASES.get(value, value)


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


def _as_utc(value: Any) -> Optional[datetime]:
    """`datetime` aware in UTC da datetime/ISO string. SQLite (test/locale)
    perde il tzinfo dei DateTime(timezone=True): un valore naive e' UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _outcome_key(market: str, outcome: Any) -> str:
    """Chiave di confronto di un esito (stessa normalizzazione di
    `get_market_outcome_baseline`), per abbinare esiti scritti diversamente."""
    return _normalize_text(_normalize_outcome_for_market(market=market, outcome=str(outcome or "")))


def _restrict_to_bet_bookmakers(
    per_bookmaker: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    market: str,
    bet_captured_at: Optional[datetime],
) -> list[dict[str, Any]]:
    """Tiene, per ogni esito, solo i bookmaker che lo quotavano gia' al
    momento della giocata (`captured_at <= bet_captured_at`).

    La quota presa e' la MEDIA dei bookmaker disponibili alla cattura
    (`OfficialPredictionCaptureService._odds_summary`): confrontarla con una
    media di chiusura che include bookmaker arrivati dopo (con margini
    diversi) sposterebbe il CLV anche a mercato fermo. Se per un esito
    nessun bookmaker era presente alla giocata, l'esito resta com'e'
    (serve comunque alla normalizzazione della fair probability)."""
    bet_at = _as_utc(bet_captured_at)
    if bet_at is None:
        return per_bookmaker
    at_bet: set[tuple[str, str]] = set()
    for snapshot in snapshots:
        captured = _as_utc(snapshot.get("captured_at"))
        if captured is not None and captured <= bet_at:
            at_bet.add((str(snapshot.get("bookmaker") or "unknown"), _outcome_key(market, snapshot.get("outcome"))))
    quoted_outcomes = {outcome for _, outcome in at_bet}
    return [
        row
        for row in per_bookmaker
        if _outcome_key(market, row.get("outcome")) not in quoted_outcomes
        or (str(row.get("bookmaker") or "unknown"), _outcome_key(market, row.get("outcome"))) in at_bet
    ]


@dataclass
class _ClosingState:
    """Baseline di chiusura piu' il motivo per cui manca e, per ogni esito
    (chiave normalizzata), l'istante dello snapshot di chiusura piu' recente."""

    baseline: Optional[dict[str, Any]]
    reason_code: Optional[str]
    captured_at_by_outcome: dict[str, datetime] = field(default_factory=dict)


def _closing_state(
    fixture_id: int,
    market: str,
    kickoff_at: Optional[datetime],
    period: str = "full_time",
    line: Optional[str] = None,
    snapshot_repo: Optional[OddsSnapshotRepository] = None,
    bet_captured_at: Optional[datetime] = None,
) -> _ClosingState:
    if kickoff_at is None:
        return _ClosingState(baseline=None, reason_code=REASON_MISSING_KICKOFF)

    repo = snapshot_repo or OddsSnapshotRepository()
    rows = repo.list_for_fixture(fixture_id=fixture_id, market=market, period=period, line=line)
    if not rows:
        return _ClosingState(baseline=None, reason_code=REASON_NO_SNAPSHOTS)

    snapshots = [row.to_dict() for row in rows]
    # Entrambe le parti normalizzate a UTC prima del confronto point-in-time.
    kickoff_at = _as_utc(kickoff_at)
    for snapshot in snapshots:
        captured = _as_utc(snapshot.get("captured_at"))
        if captured is not None:
            snapshot["captured_at"] = captured.isoformat()
    # as_of=kickoff_at: riusa EXP-04 con la STESSA regola gia' validata
    # (closing_allowed = kickoff_at is not None and as_of >= kickoff_at,
    # qui SEMPRE vera per costruzione) — "closing" per ogni bookmaker/outcome
    # e' per costruzione l'ultimo snapshot con captured_at <= kickoff_at.
    per_bookmaker = build_opening_latest_closing(snapshots, as_of=kickoff_at, kickoff_at=kickoff_at)
    per_bookmaker = _restrict_to_bet_bookmakers(per_bookmaker, snapshots, market, bet_captured_at)

    closing_odds_by_outcome: dict[str, list[float]] = {}
    captured_at_by_outcome: dict[str, datetime] = {}
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
        captured = _as_utc(closing.get("captured_at"))
        key = _outcome_key(market, row["outcome"])
        if captured is not None and (key not in captured_at_by_outcome or captured > captured_at_by_outcome[key]):
            captured_at_by_outcome[key] = captured

    if not closing_odds_by_outcome:
        return _ClosingState(baseline=None, reason_code=REASON_NO_PRE_KICKOFF_SNAPSHOT)

    odds_rows = [
        {"outcome": outcome, "avg_odd": float(np.mean(odds)), "bookmakers": len(odds)}
        for outcome, odds in closing_odds_by_outcome.items()
    ]
    return _ClosingState(
        baseline=compute_market_baseline(market=market, odds_rows=odds_rows),
        reason_code=None,
        captured_at_by_outcome=captured_at_by_outcome,
    )


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
    return _closing_state(
        fixture_id=fixture_id,
        market=market,
        kickoff_at=kickoff_at,
        period=period,
        line=line,
        snapshot_repo=snapshot_repo,
    ).baseline


# Doppia chance: la cattura ufficiale calcola `p_market_fair` come SOMMA
# delle fair probability h2h dei due esiti (`_capture_true_1x2`), perche'
# "dc" non e' un mercato a esiti esclusivi (`EXCLUSIVE_MARKETS`) e la sua
# probabilita' implicita contiene ancora il margine. La fair di chiusura va
# derivata allo stesso modo, altrimenti clv_prob confronta grandezze diverse.
_DC_H2H_COMPONENTS: dict[str, tuple[str, str]] = {
    "homedraw": ("Home", "Draw"),
    "1x": ("Home", "Draw"),
    "homeaway": ("Home", "Away"),
    "12": ("Home", "Away"),
    "drawaway": ("Draw", "Away"),
    "x2": ("Draw", "Away"),
}


def _dc_fair_closing_from_h2h(h2h_baseline: Optional[dict[str, Any]], outcome: str) -> Optional[float]:
    components = _DC_H2H_COMPONENTS.get(_normalize_text(outcome))
    if components is None or h2h_baseline is None:
        return None
    fair_values = []
    for component in components:
        row = get_market_outcome_baseline(
            fixture_baseline={"markets": {"h2h": h2h_baseline}}, market="h2h", outcome=component
        )
        if row is None or row.get("fair_probability") is None:
            return None
        fair_values.append(float(row["fair_probability"]))
    return float(sum(fair_values))


# Motivi (stabili, per i conteggi di copertura) per cui il CLV non e'
# disponibile. `reason` resta il testo leggibile gia' esposto.
REASON_MISSING_KICKOFF = "missing_kickoff"
REASON_NO_SNAPSHOTS = "no_snapshots"
REASON_NO_PRE_KICKOFF_SNAPSHOT = "no_pre_kickoff_snapshot"
REASON_OUTCOME_NOT_QUOTED = "outcome_not_quoted_at_closing"
REASON_CLOSING_NOT_AFTER_BET = "closing_not_after_bet"
REASON_INSUFFICIENT_BET_DATA = "insufficient_bet_data"

_REASON_TEXT = {
    REASON_MISSING_KICKOFF: "Closing odd non disponibile (kickoff sconosciuto)",
    REASON_NO_SNAPSHOTS: "Closing odd non disponibile (nessuno snapshot per questa fixture/mercato/linea)",
    REASON_NO_PRE_KICKOFF_SNAPSHOT: "Closing odd non disponibile (nessuno snapshot pre-kickoff per questa fixture)",
    REASON_CLOSING_NOT_AFTER_BET: (
        "Closing odd non disponibile (nessuno snapshot successivo alla quota presa: "
        "la chiusura coinciderebbe con la quota della giocata)"
    ),
    REASON_INSUFFICIENT_BET_DATA: "Dati insufficienti per calcolare il CLV (odd_at_bet/p_fair_at_bet mancanti)",
}


@dataclass
class ClvResult:
    """Output STANDARD per una singola bet/prediction (stesso stile
    "sempre gli stessi campi, None se non disponibile" di `FairOddsOutcome`
    BET-01 e `ValueDecision` BET-02).

    Campi additivi (default None): `reason_code` (motivo stabile se non
    disponibile), `pricing_market` (mercato degli snapshot usato),
    `closing_captured_at` e `closing_minutes_before_kickoff` (istante dello
    snapshot di chiusura piu' recente dell'esito e suo anticipo sul kickoff),
    `closing_bookmakers` (bookmaker nella media di chiusura)."""

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
    reason_code: Optional[str] = None
    pricing_market: Optional[str] = None
    closing_captured_at: Optional[str] = None
    closing_minutes_before_kickoff: Optional[float] = None
    closing_bookmakers: Optional[int] = None


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
    bet_captured_at: Optional[datetime] = None,
    bet_odds_captured_at: Optional[datetime] = None,
) -> ClvResult:
    """Costruisce il CLV per un singolo outcome, recuperando la closing
    odd/fair probability dagli `OddsSnapshot` (DATA-06). Non solleva mai
    un'eccezione per dati mancanti: `available=False` con `reason`
    esplicito (stesso pattern di fallback non silenzioso gia' in uso in
    tutto il progetto).

    `bet_odds_captured_at` (istante delle quote usate per la giocata, nel
    ledger `odds_captured_at`; se manca `bet_captured_at`): se nessuno
    snapshot di chiusura dell'esito e' PIU' RECENTE, la "chiusura" sarebbe
    la stessa quota della giocata (CLV = 0 per costruzione, nessuna
    informazione sul mercato): il CLV risulta non disponibile
    (`closing_not_after_bet`) invece di un falso zero. Con
    `bet_captured_at` la media di chiusura usa gli stessi bookmaker che
    quotavano l'esito alla giocata (`_restrict_to_bet_bookmakers`). Senza
    nessuno dei due (backtest) il comportamento resta quello storico."""
    pricing_market = pricing_market_for(market)
    base = {
        "market": market,
        "outcome": outcome,
        "fixture_id": fixture_id,
        "model_name": model_name,
        "odd_at_bet": odd_at_bet,
        "p_fair_at_bet": p_fair_at_bet,
        "pricing_market": pricing_market,
    }

    def _unavailable(reason_code: str, reason: Optional[str] = None, **extra: Any) -> ClvResult:
        payload = {
            "closing_odd": None,
            "p_fair_closing": None,
            **extra,
        }
        return ClvResult(
            **base,
            **payload,
            clv_odd_pct=None,
            clv_prob=None,
            available=False,
            reason=reason or _REASON_TEXT[reason_code],
            reason_code=reason_code,
        )

    state = _closing_state(
        fixture_id=fixture_id,
        market=pricing_market,
        kickoff_at=kickoff_at,
        period=period,
        line=line,
        snapshot_repo=snapshot_repo,
        bet_captured_at=bet_captured_at,
    )
    if state.baseline is None:
        return _unavailable(state.reason_code or REASON_NO_SNAPSHOTS)

    closing_row = get_market_outcome_baseline(
        fixture_baseline={"markets": {pricing_market: state.baseline}}, market=pricing_market, outcome=outcome
    )
    if closing_row is None:
        return _unavailable(REASON_OUTCOME_NOT_QUOTED, reason=f"Outcome '{outcome}' non quotato al closing")

    closing_odd = closing_row.get("avg_odd")
    p_fair_closing = closing_row.get("fair_probability")
    if pricing_market == "dc":
        h2h_state = _closing_state(
            fixture_id=fixture_id,
            market="h2h",
            kickoff_at=kickoff_at,
            period=period,
            line=None,
            snapshot_repo=snapshot_repo,
            bet_captured_at=bet_captured_at,
        )
        p_fair_closing = _dc_fair_closing_from_h2h(h2h_state.baseline, outcome)
    closing_captured = state.captured_at_by_outcome.get(_outcome_key(pricing_market, outcome))
    kickoff_utc = _as_utc(kickoff_at)
    closing_meta = {
        "closing_captured_at": closing_captured.isoformat() if closing_captured else None,
        "closing_minutes_before_kickoff": (
            (kickoff_utc - closing_captured).total_seconds() / 60.0
            if closing_captured is not None and kickoff_utc is not None
            else None
        ),
        "closing_bookmakers": int(closing_row.get("bookmakers") or 0),
    }

    bet_reference = _as_utc(bet_odds_captured_at) or _as_utc(bet_captured_at)
    if bet_reference is not None and (closing_captured is None or closing_captured <= bet_reference):
        return _unavailable(
            REASON_CLOSING_NOT_AFTER_BET,
            closing_odd=closing_odd,
            p_fair_closing=p_fair_closing,
            **closing_meta,
        )

    clv_odd_pct = compute_clv_odd_pct(odd_at_bet, closing_odd)
    clv_prob = compute_clv_prob(p_fair_at_bet, p_fair_closing)
    available = clv_odd_pct is not None or clv_prob is not None

    return ClvResult(
        **base,
        closing_odd=closing_odd,
        p_fair_closing=p_fair_closing,
        clv_odd_pct=clv_odd_pct,
        clv_prob=clv_prob,
        available=available,
        reason="" if available else _REASON_TEXT[REASON_INSUFFICIENT_BET_DATA],
        reason_code=None if available else REASON_INSUFFICIENT_BET_DATA,
        **closing_meta,
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
