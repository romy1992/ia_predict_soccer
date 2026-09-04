"""Cards O/U specializzato con linea configurabile (MARKET-06).

Analogo a Corners O/U (MARKET-05): generalizza il mercato 'cards' esistente
(soglia FISSA a 4.5, cioe' `total_cards >= 5`, in
`FilterMarketService._label_by_market` — non modificata qui per preservare
compatibilita' con quel mercato legacy) introducendo la linea come PARAMETRO
esplicito (es. 3.5, 4.5, 5.5, 6.5).

Aggiunge, oltre alle feature generiche gia' estratte da `FilterMarketService`
(che includono gia' 'Yellow Cards'/'Red Cards'/'Fouls' medi per squadra —
le feature "team/style" richieste dal task, gia' disponibili in
`mean_statistics` senza bisogno di nuova logica):
- feature ARBITRO (NUOVE, calcolate qui): media storica cartellini totali
  (gialli+rossi, entrambe le squadre) elargiti nelle partite PRECEDENTI
  arbitrate dallo stesso arbitro, point-in-time (stesso principio anti-leakage
  di `TeamStrengthExpert`, EXP-01, ma raggruppato per `Match.referee` invece
  che per squadra), con fallback a un prior neutro quando l'arbitro non ha
  ancora storico sufficiente nel dataset;
- calibrazione (Platt/isotonic) via `CalibrationService` (ML-06, riusata,
  non duplicata) per il modello di ciascuna linea;
- report con metriche PER LINEA (acceptance criteria).
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.ml.calibration.calibration_service import CalibrationResult, CalibrationService
from src.ml.validation.temporal_split import expanding_window_splits
from src.repository.match_repository import MatchRepository
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.model_registry import ModelRegistry
from src.service_ia.utility.utils import convert_orm_match_to_dict

MARKET_NAME = "cards"
ODDS_MARKET = "cards"  # colonna Odds.cards esistente, riusata come fonte feature quote
DEFAULT_LINES: tuple[float, ...] = (3.5, 4.5, 5.5, 6.5)
DEFAULT_REFEREE_PRIOR_CARDS = 4.0  # cartellini totali medi neutri quando l'arbitro non ha storico

_META_COLUMNS = ["id_fixture", "season", "league", "market", "prediction_at"]
_RF_KWARGS = dict(n_estimators=150, max_depth=10, min_samples_leaf=2, random_state=42, class_weight="balanced", n_jobs=-1)


def _line_label(line: float) -> str:
    return f"line_{str(float(line)).replace('.', '_')}"


def label_cards_over(total_cards: Any, line: float) -> Any:
    """Target reale (dal risultato, MAI dalle quote): 1 se cartellini totali > line.

    Accetta sia uno scalare sia un array; la linea e' un PARAMETRO libero,
    non una soglia hardcoded (acceptance criteria 'Linee configurabili')."""
    result = (np.asarray(total_cards, dtype=float) > float(line)).astype(int)
    return result if result.shape else int(result)


def _team_total_cards(stat: dict[str, Any]) -> int:
    yellow = int(stat.get("yellow_cards") or 0)
    red = int(stat.get("red_cards") or 0)
    return yellow + red


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _sorted_matches_chronologically(matches: list[dict[str, Any]]) -> list[tuple[datetime, dict[str, Any]]]:
    parsed: list[tuple[datetime, dict[str, Any]]] = []
    for match in matches:
        prediction_at = _parse_datetime(match.get("date_match"))
        if prediction_at is None:
            continue
        parsed.append((prediction_at, match))
    parsed.sort(key=lambda item: (item[0], item[1].get("id_fixture") or 0))
    return parsed


@dataclass
class _RefereeState:
    matches_officiated: int = 0
    total_cards_sum: float = 0.0

    @property
    def average_cards(self) -> Optional[float]:
        if self.matches_officiated == 0:
            return None
        return self.total_cards_sum / self.matches_officiated

    def update(self, total_cards: int) -> None:
        self.total_cards_sum += float(total_cards)
        self.matches_officiated += 1


def build_referee_features_dataset(
    matches: list[dict[str, Any]],
    default_prior_cards: float = DEFAULT_REFEREE_PRIOR_CARDS,
) -> pd.DataFrame:
    """1 riga per fixture: statistiche ARBITRO point-in-time (PRE-match).

    Nessun leakage: per la riga N si usano SOLO le partite arbitrate dallo
    stesso arbitro (`Match.referee`, stringa) con indice cronologico < N.
    Se l'arbitro e' assente/sconosciuto o non ha ancora storico, si usa
    `default_prior_cards` (prior neutro) e `referee_has_history=0`.
    """
    states: dict[str, _RefereeState] = defaultdict(_RefereeState)
    rows: list[dict[str, Any]] = []

    for prediction_at, match in _sorted_matches_chronologically(matches):
        referee = str(match.get("referee") or "").strip()
        stats = match.get("statistics") or []
        home_id = match.get("id_team_home")
        away_id = match.get("id_team_away")
        stat_home = next((s for s in stats if s.get("statistics_team_id") == home_id), None)
        stat_away = next((s for s in stats if s.get("statistics_team_id") == away_id), None)
        if stat_home is None or stat_away is None:
            continue

        state = states[referee] if referee else _RefereeState()
        prior_average = state.average_cards

        rows.append(
            {
                "id_fixture": match.get("id_fixture"),
                "prediction_at": prediction_at.isoformat(),
                "referee_avg_cards_prior": float(prior_average) if prior_average is not None else float(default_prior_cards),
                "referee_matches_officiated_prior": int(state.matches_officiated),
                "referee_has_history": int(prior_average is not None),
            }
        )

        if referee:
            total_cards = _team_total_cards(stat_home) + _team_total_cards(stat_away)
            state.update(total_cards)

    if not rows:
        return pd.DataFrame(
            columns=["id_fixture", "referee_avg_cards_prior", "referee_matches_officiated_prior", "referee_has_history"]
        )

    return pd.DataFrame(rows)


def build_cards_frame_from_records(
    matches: list[dict[str, Any]],
    lines: tuple[float, ...] = DEFAULT_LINES,
    odds_market: str = ODDS_MARKET,
    default_prior_cards: float = DEFAULT_REFEREE_PRIOR_CARDS,
) -> pd.DataFrame:
    """Dataset (feature odds/mean_stats generiche "team/style" + feature
    arbitro dedicate + target reale PER OGNI linea) sulle fixture con quote
    'cards' disponibili."""
    service = FilterMarketService()
    rows: list[dict[str, Any]] = []
    for match in matches:
        row = service._build_row(match=match, market=odds_market, with_target=False)
        if not row:
            continue

        stat_home, stat_away = FilterMarketService._resolve_team_stats(match=match, with_full_stats=True)
        if not stat_home or not stat_away:
            continue

        total_cards = _team_total_cards(stat_home) + _team_total_cards(stat_away)
        row["total_cards"] = total_cards
        for line in lines:
            row[f"y_{_line_label(line)}"] = int(label_cards_over(total_cards, line))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)
    frame["id_fixture"] = frame["id_fixture"].astype(int)
    frame["prediction_at"] = pd.to_datetime(frame["prediction_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)

    referee_frame = build_referee_features_dataset(matches, default_prior_cards=default_prior_cards)
    referee_columns = ["referee_avg_cards_prior", "referee_matches_officiated_prior", "referee_has_history"]
    if referee_frame.empty:
        for col in referee_columns:
            frame[col] = default_prior_cards if col == "referee_avg_cards_prior" else 0
        return frame

    referee_frame = referee_frame[["id_fixture", *referee_columns]].copy()
    referee_frame["id_fixture"] = referee_frame["id_fixture"].astype(int)
    frame = frame.merge(referee_frame, on="id_fixture", how="left")
    frame["referee_avg_cards_prior"] = frame["referee_avg_cards_prior"].fillna(default_prior_cards)
    frame["referee_matches_officiated_prior"] = frame["referee_matches_officiated_prior"].fillna(0)
    frame["referee_has_history"] = frame["referee_has_history"].fillna(0)
    return frame


def _feature_columns_for(frame: pd.DataFrame, lines: tuple[float, ...]) -> list[str]:
    y_columns = {f"y_{_line_label(line)}" for line in lines}
    excluded = set(_META_COLUMNS) | {"total_cards"} | y_columns
    return [col for col in frame.columns if col not in excluded]


@dataclass
class CardsLineTrainResult:
    """Esito training+calibrazione per UNA linea specifica."""

    line: float
    sample_size: int
    feature_names: list[str]
    calibration: CalibrationResult


def train_cards_line(
    frame: pd.DataFrame,
    line: float,
    cv_splits: list[tuple[list[int], list[int]]],
    lines_in_frame: tuple[float, ...] = DEFAULT_LINES,
) -> CardsLineTrainResult:
    """Addestra + calibra (`CalibrationService`, ML-06) il modello per una
    singola linea, con validazione temporale OOF (mai split random)."""
    label = f"y_{_line_label(line)}"
    if label not in frame.columns:
        raise ValueError(f"Linea non presente nel dataset: {line}")

    feature_columns = _feature_columns_for(frame, lines_in_frame)
    X = frame[feature_columns]
    y = frame[label].astype(int)
    if y.nunique() < 2:
        raise ValueError(f"Target a classe unica per la linea {line}: impossibile addestrare")

    estimator = RandomForestClassifier(**_RF_KWARGS)
    calibration = CalibrationService.calibrate_estimator(estimator=estimator, X=X, y=y, cv_splits=cv_splits)

    return CardsLineTrainResult(line=float(line), sample_size=int(len(y)), feature_names=feature_columns, calibration=calibration)


@dataclass
class CardsBenchmarkReport:
    """Report multi-linea (acceptance criteria 'Metriche per linea')."""

    lines: tuple[float, ...]
    results: dict[str, CardsLineTrainResult]

    def metrics_summary(self) -> dict[str, dict[str, Any]]:
        return {
            label: {
                "line": result.line,
                "sample_size": result.sample_size,
                "pre_metrics": result.calibration.pre_metrics,
                "post_metrics": result.calibration.post_metrics,
                "calibration_method": result.calibration.method,
            }
            for label, result in self.results.items()
        }


def train_cards_all_lines(
    frame: pd.DataFrame,
    lines: tuple[float, ...] = DEFAULT_LINES,
    cv_splits: Optional[list[tuple[list[int], list[int]]]] = None,
) -> CardsBenchmarkReport:
    """Addestra+calibra ciascuna linea sullo STESSO walk-forward. Una linea
    con classe unica (dati insufficienti) viene saltata SENZA bloccare le
    altre linee addestrabili."""
    if cv_splits is None:
        min_train = max(30, int(len(frame) * 0.45))
        min_valid = max(10, int(len(frame) * 0.1))
        cv_splits = expanding_window_splits(
            frame=frame, time_col="prediction_at", n_splits=5, min_train_size=min_train, min_valid_size=min_valid
        )
    if not cv_splits:
        raise ValueError("Dataset insufficiente per validazione temporale (nessuno split valido)")

    results: dict[str, CardsLineTrainResult] = {}
    for line in lines:
        try:
            results[_line_label(line)] = train_cards_line(frame=frame, line=line, cv_splits=cv_splits, lines_in_frame=lines)
        except ValueError:
            continue

    if not results:
        raise ValueError("Nessuna linea addestrabile con questo dataset (classi troppo sbilanciate per tutte le linee)")

    return CardsBenchmarkReport(lines=lines, results=results)


@dataclass
class CardsExpert:
    """Interfaccia comune (`predict_proba`) parametrica sulla linea: OGNI
    linea e' un mercato/modello indipendente nel registry (es.
    'cards_line_4_5'), analogamente a `CornersExpert` (MARKET-05)."""

    line: float
    estimator: Any
    feature_names: list[str] = field(default_factory=list)
    run_id: Optional[str] = None
    stage: Optional[str] = None

    def __post_init__(self) -> None:
        if not hasattr(self.estimator, "predict_proba"):
            raise TypeError("L'estimator caricato non espone predict_proba: interfaccia comune non rispettata")

    def predict_proba(self, X: Any) -> np.ndarray:
        ordered = X[self.feature_names] if self.feature_names else X
        raw = self.estimator.predict_proba(ordered)
        return CalibrationService._class1_probability(raw)

    def predict_proba_dict(self, X: Any) -> dict[str, np.ndarray]:
        p_over = self.predict_proba(X)
        return {"over": p_over, "under": 1.0 - p_over}

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    @staticmethod
    def _market_for_line(line: float) -> str:
        return f"{MARKET_NAME}_{_line_label(line)}"

    @classmethod
    def _from_run(cls, line: float, run: dict[str, Any]) -> "CardsExpert":
        model_path = run.get("model_path")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path non trovato per il run {run.get('run_id')}: {model_path}")

        estimator = joblib.load(model_path)
        return cls(
            line=float(line),
            estimator=estimator,
            feature_names=list(run.get("feature_names") or []),
            run_id=run.get("run_id"),
            stage=run.get("current_stage") or run.get("stage"),
        )

    @classmethod
    def load_production(cls, line: float, registry: Optional[Any] = None) -> "CardsExpert":
        """Carica lo stage 'production' per la linea richiesta (vincolo
        generale: latest != production)."""
        registry = registry or ModelRegistry()
        market = cls._market_for_line(line)
        run = registry.get_production(market=market)
        if run is None:
            raise LookupError(f"Nessun modello in stage 'production' per '{market}'")
        return cls._from_run(line=line, run=run)

    @classmethod
    def load_latest(cls, line: float, registry: Optional[Any] = None) -> "CardsExpert":
        """ATTENZIONE: solo per debug/validazione, mai per servire predizioni reali."""
        registry = registry or ModelRegistry()
        market = cls._market_for_line(line)
        run = registry.get_latest(market=market)
        if run is None:
            raise LookupError(f"Nessun modello registrato per '{market}'")
        return cls._from_run(line=line, run=run)

    @classmethod
    def from_estimator(cls, line: float, estimator: Any, feature_names: Optional[list[str]] = None) -> "CardsExpert":
        """Costruzione diretta (utile nei test, senza toccare registry/disco)."""
        return cls(line=float(line), estimator=estimator, feature_names=list(feature_names or []))


@dataclass
class CardsBenchmarkRunResult:
    market: str
    rows: int
    status: str
    lines_trained: list[float]
    details: dict[str, Any]


def _save_line_model(result: CardsLineTrainResult) -> dict[str, Any]:
    line_label = _line_label(result.line)
    model_path = os.path.abspath(os.path.join("best_models", f"{MARKET_NAME}_{line_label}_champion.pkl"))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(result.calibration.calibrator, model_path)

    registry = ModelRegistry()
    return registry.register(
        model_path=model_path,
        market=f"{MARKET_NAME}_{line_label}",
        model_name="calibrated_random_forest",
        feature_names=result.feature_names,
        metrics={
            "pre_log_loss": result.calibration.pre_metrics.get("log_loss"),
            "post_log_loss": result.calibration.post_metrics.get("log_loss"),
            "pre_brier": result.calibration.pre_metrics.get("brier"),
            "post_brier": result.calibration.post_metrics.get("brier"),
        },
        extra={
            "line": result.line,
            "calibration_method": result.calibration.method,
            "sample_size": result.sample_size,
        },
        stage="candidate",
    )


def run_cards_benchmark(
    matches: list[dict[str, Any]],
    lines: tuple[float, ...] = DEFAULT_LINES,
    odds_market: str = ODDS_MARKET,
    save_model: bool = True,
) -> CardsBenchmarkRunResult:
    """Costruisce il dataset reale, addestra+calibra ciascuna linea e,
    se richiesto, registra ciascun modello come 'candidate' separato."""
    frame = build_cards_frame_from_records(matches, lines=lines, odds_market=odds_market)
    if frame.empty:
        return CardsBenchmarkRunResult(market=MARKET_NAME, rows=0, status="skipped_no_data", lines_trained=[], details={})

    try:
        report = train_cards_all_lines(frame, lines=lines)
    except ValueError as exc:
        return CardsBenchmarkRunResult(
            market=MARKET_NAME,
            rows=len(frame),
            status="skipped_insufficient_data_for_all_lines",
            lines_trained=[],
            details={"error": str(exc)},
        )

    run_metadata: dict[str, Any] = {}
    if save_model:
        for label, result in report.results.items():
            run_metadata[label] = _save_line_model(result)

    return CardsBenchmarkRunResult(
        market=MARKET_NAME,
        rows=len(frame),
        status="benchmarked",
        lines_trained=[result.line for result in report.results.values()],
        details={"metrics_summary": report.metrics_summary(), "runs": run_metadata},
    )


def run_cards_benchmark_from_db(
    seasons: Optional[list[int]] = None,
    lines: tuple[float, ...] = DEFAULT_LINES,
    odds_market: str = ODDS_MARKET,
    save_model: bool = True,
) -> CardsBenchmarkRunResult:
    """Variante DB reale: filtra le fixture con statistiche/mean_statistics/odds
    disponibili (stesso requisito degli altri dataset builder di mercato)."""
    match_repo = MatchRepository()
    filters: dict[str, Any] = {
        "statistics": "not None",
        "mean_statistics": "not None",
        "odds": "not None",
        "status": ["FT"],
    }
    if seasons:
        filters["season"] = seasons
    matches = convert_orm_match_to_dict(match_repo.search_filter(filters=filters))

    return run_cards_benchmark(matches=matches, lines=lines, odds_market=odds_market, save_model=save_model)
