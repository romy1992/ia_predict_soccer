"""Betting Backtester — orchestrazione end-to-end (BET-03, Fase BETTING).

Collega, per un singolo mercato di `FilterMarketService.SUPPORTED_MARKETS`,
i mattoni gia' validati nel progetto SENZA duplicarne la logica:

- `FilterMarketService._build_row` (EXP-05): stesse feature/target gia' usate
  da `train_multi_market.py`/`totals_market.py`/`btts_market.py`.
- `expanding_window_splits` + `temporal_oof_probabilities` (ML-05): STESSO
  meccanismo walk-forward gia' in uso in tutto il progetto. E' quello che
  garantisce l'acceptance criteria "Backtest solo out-of-sample" — `p_model`
  per ogni fixture proviene SEMPRE da un fold di validazione MAI usato per il
  training di quel fold (nessuna nuova logica di split).
- `compute_market_baseline` / `build_fair_odds_outcome` (ML-04/BET-01): fair
  odds standard per lo stesso outcome.
- `evaluate_value`/`build_backtest_bet` (BET-02/BET-03): edge/EV/decisione.

Nota deliberata sullo scope (per non anticipare task successivi ne'
riscrivere componenti esistenti):
- Il backtest NON allena/seleziona un nuovo "modello campione" (quello resta
  compito di `train_multi_market.py`/ORACLE-02): l'`estimator` di default e'
  una semplice pipeline Logistic Regression, sostituibile dal chiamante con
  QUALUNQUE stimatore sklearn-compatibile (stessa interfaccia generica gia'
  usata da `temporal_oof_probabilities`).
- Per corners/cards viene usata ESCLUSIVAMENTE la linea che coincide
  esattamente con la soglia target di `FilterMarketService` (9.5 corner /
  4.5 cartellini): se quella riga non e' quotata per una fixture, la bet
  resta "NO BET" (quota mancante, BET-02) invece di sostituire silenziosamente
  la quota di una linea diversa (che distorcerebbe ROI/EV) — a differenza del
  pick "nearest line" usato in `dashboard_service.py` solo a scopo di
  visualizzazione.
- Nota nota (data quality, FUORI SCOPE per questo modulo): l'ingestion
  storica (`download_match_service.map_odds`) ha un bug pre-esistente per cui
  le quote 'Yes'/'No' di 'goal_no_goal' finiscono entrambe sulla stessa
  chiave 'no_goal_'; questo modulo non lo corregge (non e' un file
  `src/oracle/backtest/`), semplicemente qualunque riga senza quota valida
  degrada a NO BET (BET-02) invece di produrre un numero inventato.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.baselines.bookmaker_baseline import compute_market_baseline, get_market_outcome_baseline
from src.ml.evaluation.probability_metrics import temporal_oof_probabilities
from src.ml.validation.temporal_split import expanding_window_splits
from src.oracle.backtest.betting_backtester import (
    DEFAULT_EDGE_BUCKET_EDGES,
    DEFAULT_PLACED_DECISIONS,
    STAKE_DEFAULT,
    BacktestBet,
    BacktestReport,
    build_backtest_bet,
    compute_backtest_report,
    persist_backtest_report,
)
from src.oracle.fair_odds.fair_odds_engine import build_fair_odds_outcome
from src.oracle.value_engine.value_engine import DEFAULT_POLICY, ValueDecisionPolicy
from src.repository.match_repository import MatchRepository
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.utility.utils import convert_orm_match_to_dict

_META_COLUMNS = ["id_fixture", "season", "league", "market", "prediction_at"]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _outcome_odds_rows_from_raw_market_odds(market_odds: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """Converte il dict grezzo di `Odds.<market>` (chiavi 'outcome_bookmaker',
    stesso formato scritto da `download_match_service.map_odds`) nelle righe
    outcome/avg_odd/bookmakers richieste da `compute_market_baseline`
    (BET-01/ML-04, riusato senza modifiche)."""
    buckets: dict[str, list[float]] = {}
    for key, raw_odd in (market_odds or {}).items():
        odd = _safe_float(raw_odd)
        if odd is None:
            continue
        outcome, separator, _bookmaker = str(key).rpartition("_")
        outcome = outcome if separator else str(key)
        buckets.setdefault(outcome, []).append(odd)

    return [
        {"outcome": outcome, "avg_odd": float(np.mean(odds)), "bookmakers": len(odds)}
        for outcome, odds in buckets.items()
    ]


def _canonical_outcome_for_prediction(market: str, prediction: int) -> str:
    """Nome outcome canonico coerente CON LA STESSA soglia usata da
    `FilterMarketService._label_by_market` per costruire `y` (stesso
    mercato -> stesso confine): garantisce che `won` (calcolato confrontando
    `prediction` con `y_true`) corrisponda davvero all'outcome quotato."""
    if market == "h2h":
        return "Home" if prediction == 1 else "Not Home"
    if market == "goal_no_goal":
        return "Yes" if prediction == 1 else "No"
    if market == "dc":
        return "1X" if prediction == 1 else "Away"
    if market.startswith("under_over_"):
        threshold = market.replace("under_over_", "").replace("_", ".")
        return f"Over {threshold}" if prediction == 1 else f"Under {threshold}"
    if market == "corners":
        return "Over 9.5" if prediction == 1 else "Under 9.5"
    if market == "cards":
        return "Over 4.5" if prediction == 1 else "Under 4.5"
    raise ValueError(f"Mercato non supportato dal backtest: {market}")


def _default_estimator() -> Pipeline:
    """Stimatore di default per generare `p_model` OOF: semplice, veloce,
    deterministico (`random_state` fisso). Il chiamante puo' sostituirlo con
    qualunque altro stimatore sklearn-compatibile."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )


def _temporal_cv_splits(frame: pd.DataFrame) -> list[tuple[list[int], list[int]]]:
    """Stessi default di walk-forward gia' usati altrove nel progetto
    (`train_multi_market._build_temporal_cv`, `totals_market.run_totals_benchmark`)."""
    min_train = max(30, int(len(frame) * 0.45))
    min_valid = max(10, int(len(frame) * 0.1))
    return expanding_window_splits(
        frame=frame, time_col="prediction_at", n_splits=5, min_train_size=min_train, min_valid_size=min_valid
    )


def _filter_valid_splits(y: pd.Series, splits: list[tuple[list[int], list[int]]]) -> list[tuple[list[int], list[int]]]:
    """Scarta i fold con train/valid a classe singola (stesso controllo gia'
    presente in `train_multi_market._filter_valid_splits`): evita che
    `temporal_oof_probabilities` fallisca fittando uno stimatore su un fold
    dove `y` non ha almeno 2 classi."""
    valid_splits: list[tuple[list[int], list[int]]] = []
    for train_idx, valid_idx in splits:
        if not train_idx or not valid_idx:
            continue
        if y.iloc[train_idx].nunique() < 2 or y.iloc[valid_idx].nunique() < 2:
            continue
        valid_splits.append((train_idx, valid_idx))
    return valid_splits


def build_backtest_dataset(
    matches: list[dict[str, Any]], market: str
) -> tuple[pd.DataFrame, dict[int, list[dict[str, Any]]]]:
    """Dataset (feature + target reale) allineato all'ordine temporale, piu'
    le quote storiche GREZZE per fixture (usate SOLO per il fair-odds
    lookup, non come feature ML): riusa `FilterMarketService._build_row`
    SENZA duplicarne la logica di estrazione feature/target."""
    if market not in FilterMarketService.SUPPORTED_MARKETS:
        raise ValueError(f"Mercato non supportato: {market}")

    service = FilterMarketService()
    rows: list[dict[str, Any]] = []
    odds_rows_by_fixture: dict[int, list[dict[str, Any]]] = {}

    for match in matches:
        row = service._build_row(match=match, market=market, with_target=True)
        if not row or row.get("id_fixture") is None:
            continue

        odds_list = match.get("odds") or []
        raw_market_odds = (odds_list[0] or {}).get(market) if odds_list else None
        fixture_id = int(row["id_fixture"])
        odds_rows_by_fixture[fixture_id] = _outcome_odds_rows_from_raw_market_odds(raw_market_odds)
        rows.append(row)

    if not rows:
        return pd.DataFrame(), {}

    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)
    frame["id_fixture"] = frame["id_fixture"].astype(int)
    frame["prediction_at"] = pd.to_datetime(frame["prediction_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)
    return frame, odds_rows_by_fixture


def build_backtest_bets_from_oof(
    frame: pd.DataFrame,
    odds_rows_by_fixture: dict[int, list[dict[str, Any]]],
    oof: pd.DataFrame,
    market: str,
    policy: ValueDecisionPolicy = DEFAULT_POLICY,
) -> list[BacktestBet]:
    """Da probabilita' OOF (out-of-sample per costruzione) a `BacktestBet`:
    stessa convenzione pick/probabilita' di `DashboardService._build_decision_cards`
    (predizione = classe con probabilita' >= 0.5, `p_model` = probabilita'
    ASSOCIATA a quella classe), cosi' il backtest riflette come il sistema
    piazzerebbe realmente la bet."""
    bets: list[BacktestBet] = []

    for _, oof_row in oof.iterrows():
        idx = int(oof_row["index"])
        if idx < 0 or idx >= len(frame):
            continue
        source_row = frame.iloc[idx]

        p1 = float(oof_row["probability"])
        y_true = int(oof_row["y_true"])
        prediction = 1 if p1 >= 0.5 else 0
        predicted_probability = p1 if prediction == 1 else (1.0 - p1)
        won = bool(y_true == prediction)

        outcome = _canonical_outcome_for_prediction(market=market, prediction=prediction)
        fixture_id = int(source_row["id_fixture"])
        odds_rows = odds_rows_by_fixture.get(fixture_id) or []
        baseline = compute_market_baseline(market=market, odds_rows=odds_rows)
        baseline_row = get_market_outcome_baseline(
            fixture_baseline={"markets": {market: baseline}}, market=market, outcome=outcome
        )

        fair_odds_outcome = build_fair_odds_outcome(
            market=market, outcome=outcome, market_baseline_row=baseline_row, p_model=predicted_probability
        )

        kickoff = source_row.get("prediction_at")
        kickoff_at = kickoff.isoformat() if pd.notna(kickoff) else None

        bets.append(
            build_backtest_bet(
                market=market,
                outcome=outcome,
                p_model=fair_odds_outcome.p_model,
                p_market_fair=fair_odds_outcome.p_market_fair,
                odd=fair_odds_outcome.odd,
                won=won,
                fixture_id=fixture_id,
                league=source_row.get("league"),
                season=source_row.get("season"),
                kickoff_at=kickoff_at,
                policy=policy,
            )
        )

    return bets


@dataclass
class MarketBacktestResult:
    market: str
    rows: int
    status: str
    report: Optional[BacktestReport]
    persisted_path: Optional[str]


def run_market_backtest(
    matches: list[dict[str, Any]],
    market: str,
    estimator: Optional[Any] = None,
    stake: float = STAKE_DEFAULT,
    include_decisions: frozenset[str] = DEFAULT_PLACED_DECISIONS,
    edge_bucket_edges: tuple[float, ...] = DEFAULT_EDGE_BUCKET_EDGES,
    policy: ValueDecisionPolicy = DEFAULT_POLICY,
    persist: bool = False,
) -> MarketBacktestResult:
    """Esegue il backtest per un mercato su una lista di match gia' in
    memoria (nessun accesso DB: comodo per test e per riuso da altri
    orchestratori). Ritorna sempre un risultato (mai un'eccezione per dati
    insufficienti: status esplicito, stesso stile di `run_totals_benchmark`)."""
    frame, odds_rows_by_fixture = build_backtest_dataset(matches=matches, market=market)
    if frame.empty:
        return MarketBacktestResult(market=market, rows=0, status="skipped_no_data", report=None, persisted_path=None)

    feature_columns = [col for col in frame.columns if col not in {*_META_COLUMNS, "y"}]
    X = frame[feature_columns]
    y = frame["y"].astype(int)

    cv_splits = _filter_valid_splits(y=y, splits=_temporal_cv_splits(frame))
    if not cv_splits:
        return MarketBacktestResult(
            market=market,
            rows=len(frame),
            status="skipped_insufficient_rows_for_temporal_cv",
            report=None,
            persisted_path=None,
        )

    oof = temporal_oof_probabilities(estimator=estimator or _default_estimator(), X=X, y=y, cv_splits=cv_splits)
    if oof.empty:
        return MarketBacktestResult(
            market=market, rows=len(frame), status="skipped_no_oof_rows", report=None, persisted_path=None
        )

    bets = build_backtest_bets_from_oof(
        frame=frame, odds_rows_by_fixture=odds_rows_by_fixture, oof=oof, market=market, policy=policy
    )
    report = compute_backtest_report(
        bets=bets, stake=stake, include_decisions=include_decisions, edge_bucket_edges=edge_bucket_edges
    )

    persisted_path = persist_backtest_report(market=market, report=report) if persist else None

    return MarketBacktestResult(market=market, rows=len(frame), status="backtested", report=report, persisted_path=persisted_path)


def run_market_backtest_from_db(
    market: str,
    seasons: Optional[list[int]] = None,
    leagues: Optional[list[int]] = None,
    estimator: Optional[Any] = None,
    stake: float = STAKE_DEFAULT,
    include_decisions: frozenset[str] = DEFAULT_PLACED_DECISIONS,
    edge_bucket_edges: tuple[float, ...] = DEFAULT_EDGE_BUCKET_EDGES,
    policy: ValueDecisionPolicy = DEFAULT_POLICY,
    persist: bool = False,
) -> MarketBacktestResult:
    """Variante DB reale: stesso filtro (statistics/mean_statistics/odds
    disponibili, status FT) gia' usato da `run_totals_benchmark_from_db`."""
    match_repo = MatchRepository()
    filters: dict[str, Any] = {
        "statistics": "not None",
        "mean_statistics": "not None",
        "odds": "not None",
        "status": ["FT"],
    }
    if seasons:
        filters["season"] = seasons
    if leagues:
        filters["current_league"] = leagues

    matches = convert_orm_match_to_dict(match_repo.search_filter(filters=filters))
    return run_market_backtest(
        matches=matches,
        market=market,
        estimator=estimator,
        stake=stake,
        include_decisions=include_decisions,
        edge_bucket_edges=edge_bucket_edges,
        policy=policy,
        persist=persist,
    )








