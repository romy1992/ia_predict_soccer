"""Corners O/U specializzato con linea configurabile (MARKET-05).

Generalizza il mercato 'corners' esistente (soglia FISSA a 9.5, cioe'
`total_corners >= 10`, in `FilterMarketService._label_by_market` — non
modificata qui per preservare compatibilita' con quel mercato legacy)
introducendo la linea come PARAMETRO esplicito (es. 8.5, 9.5, 10.5, 11.5),
stesso pattern gia' adottato per U/O gol in
`src/ml/markets/totals/totals_market.py` (MARKET-04): il TARGET reale (dal
risultato, mai dalle quote) e' calcolabile per qualunque linea, mentre la
FONTE feature (odds + mean_statistics) resta quella gia' disponibile nella
colonna `Odds.corners`.

Aggiunge:
- feature DEDICATE ai corner (oltre alle generiche odds/mean_stats gia'
  estratte da `FilterMarketService`): media storica corner fatti/concessi
  per squadra (`mean_statistics['Corner Kicks']`), totale medio atteso e
  differenziale — nessuna nuova query DB, solo un secondo utilizzo mirato
  di un campo gia' disponibile;
- calibrazione (Platt/isotonic) via `CalibrationService` (ML-06, gia'
  esistente, non duplicata) per il modello di CIASCUNA linea;
- report con metriche PER LINEA (acceptance criteria).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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

MARKET_NAME = "corners"
ODDS_MARKET = "corners"  # colonna Odds.corners esistente, riusata come fonte feature quote
DEFAULT_LINES: tuple[float, ...] = (8.5, 9.5, 10.5, 11.5)

_META_COLUMNS = ["id_fixture", "season", "league", "market", "prediction_at"]
_RF_KWARGS = dict(n_estimators=150, max_depth=10, min_samples_leaf=2, random_state=42, class_weight="balanced", n_jobs=-1)


def _line_label(line: float) -> str:
    return f"line_{str(float(line)).replace('.', '_')}"


def label_corners_over(total_corners: Any, line: float) -> Any:
    """Target reale (dal risultato, MAI dalle quote): 1 se corner totali > line.

    Accetta sia uno scalare sia un array; la linea e' un PARAMETRO libero,
    non una soglia hardcoded (acceptance criteria 'Linee configurabili')."""
    result = (np.asarray(total_corners, dtype=float) > float(line)).astype(int)
    return result if result.shape else int(result)


def _corner_dedicated_features(match: dict[str, Any]) -> dict[str, float]:
    """Feature DEDICATE ai corner: media storica corner per squadra (gia'
    disponibile in `mean_statistics['Corner Kicks']`, calcolata da
    `download_match_service.py`), totale medio atteso e differenziale.
    Nessuna nuova query DB: riusa `_resolve_mean_stats` gia' esistente."""
    mean_home, mean_away = FilterMarketService._resolve_mean_stats(match)
    if not mean_home or not mean_away:
        return {}

    home_mean = FilterMarketService._safe_float(mean_home.get("Corner Kicks"))
    away_mean = FilterMarketService._safe_float(mean_away.get("Corner Kicks"))

    return {
        "corner_mean_home_dedicated": home_mean,
        "corner_mean_away_dedicated": away_mean,
        "corner_mean_total_dedicated": home_mean + away_mean,
        "corner_mean_diff_dedicated": home_mean - away_mean,
    }


def build_corners_frame_from_records(
    matches: list[dict[str, Any]],
    lines: tuple[float, ...] = DEFAULT_LINES,
    odds_market: str = ODDS_MARKET,
) -> pd.DataFrame:
    """Dataset (feature odds/mean_stats generiche + feature dedicate corner +
    target reale PER OGNI linea) sulle fixture con quote 'corners' disponibili."""
    service = FilterMarketService()
    rows: list[dict[str, Any]] = []
    for match in matches:
        row = service._build_row(match=match, market=odds_market, with_target=False)
        if not row:
            continue

        stat_home, stat_away = FilterMarketService._resolve_team_stats(match=match, with_full_stats=True)
        if not stat_home or not stat_away:
            continue
        home_corners = stat_home.get("corners")
        away_corners = stat_away.get("corners")
        if home_corners is None or away_corners is None:
            continue

        total_corners = int(home_corners) + int(away_corners)
        row["total_corners"] = total_corners
        row.update(_corner_dedicated_features(match))
        for line in lines:
            row[f"y_{_line_label(line)}"] = int(label_corners_over(total_corners, line))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)
    frame["id_fixture"] = frame["id_fixture"].astype(int)
    frame["prediction_at"] = pd.to_datetime(frame["prediction_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)
    return frame


def _feature_columns_for(frame: pd.DataFrame, lines: tuple[float, ...]) -> list[str]:
    y_columns = {f"y_{_line_label(line)}" for line in lines}
    excluded = set(_META_COLUMNS) | {"total_corners"} | y_columns
    return [col for col in frame.columns if col not in excluded]


@dataclass
class CornersLineTrainResult:
    """Esito training+calibrazione per UNA linea specifica."""

    line: float
    sample_size: int
    feature_names: list[str]
    calibration: CalibrationResult


def train_corners_line(
    frame: pd.DataFrame,
    line: float,
    cv_splits: list[tuple[list[int], list[int]]],
    lines_in_frame: tuple[float, ...] = DEFAULT_LINES,
) -> CornersLineTrainResult:
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

    return CornersLineTrainResult(line=float(line), sample_size=int(len(y)), feature_names=feature_columns, calibration=calibration)


@dataclass
class CornersBenchmarkReport:
    """Report multi-linea (acceptance criteria 'Metriche per linea')."""

    lines: tuple[float, ...]
    results: dict[str, CornersLineTrainResult]

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


def train_corners_all_lines(
    frame: pd.DataFrame,
    lines: tuple[float, ...] = DEFAULT_LINES,
    cv_splits: Optional[list[tuple[list[int], list[int]]]] = None,
) -> CornersBenchmarkReport:
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

    results: dict[str, CornersLineTrainResult] = {}
    for line in lines:
        try:
            results[_line_label(line)] = train_corners_line(frame=frame, line=line, cv_splits=cv_splits, lines_in_frame=lines)
        except ValueError:
            continue

    if not results:
        raise ValueError("Nessuna linea addestrabile con questo dataset (classi troppo sbilanciate per tutte le linee)")

    return CornersBenchmarkReport(lines=lines, results=results)


@dataclass
class CornersExpert:
    """Interfaccia comune (`predict_proba`) parametrica sulla linea: OGNI
    linea e' un mercato/modello indipendente nel registry (es.
    'corners_line_9_5'), analogamente a come 'under_over_1_5/2_5/...' sono
    gia' mercati separati (EXP-05)."""

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
    def _from_run(cls, line: float, run: dict[str, Any]) -> "CornersExpert":
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
    def load_production(cls, line: float, registry: Optional[Any] = None) -> "CornersExpert":
        """Carica lo stage 'production' per la linea richiesta (vincolo
        generale: latest != production)."""
        registry = registry or ModelRegistry()
        market = cls._market_for_line(line)
        run = registry.get_production(market=market)
        if run is None:
            raise LookupError(f"Nessun modello in stage 'production' per '{market}'")
        return cls._from_run(line=line, run=run)

    @classmethod
    def load_latest(cls, line: float, registry: Optional[Any] = None) -> "CornersExpert":
        """ATTENZIONE: solo per debug/validazione, mai per servire predizioni reali."""
        registry = registry or ModelRegistry()
        market = cls._market_for_line(line)
        run = registry.get_latest(market=market)
        if run is None:
            raise LookupError(f"Nessun modello registrato per '{market}'")
        return cls._from_run(line=line, run=run)

    @classmethod
    def from_estimator(cls, line: float, estimator: Any, feature_names: Optional[list[str]] = None) -> "CornersExpert":
        """Costruzione diretta (utile nei test, senza toccare registry/disco)."""
        return cls(line=float(line), estimator=estimator, feature_names=list(feature_names or []))


@dataclass
class CornersBenchmarkRunResult:
    market: str
    rows: int
    status: str
    lines_trained: list[float]
    details: dict[str, Any]


def _save_line_model(result: CornersLineTrainResult) -> dict[str, Any]:
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


def run_corners_benchmark(
    matches: list[dict[str, Any]],
    lines: tuple[float, ...] = DEFAULT_LINES,
    odds_market: str = ODDS_MARKET,
    save_model: bool = True,
) -> CornersBenchmarkRunResult:
    """Costruisce il dataset reale, addestra+calibra ciascuna linea e,
    se richiesto, registra ciascun modello come 'candidate' separato."""
    frame = build_corners_frame_from_records(matches, lines=lines, odds_market=odds_market)
    if frame.empty:
        return CornersBenchmarkRunResult(market=MARKET_NAME, rows=0, status="skipped_no_data", lines_trained=[], details={})

    try:
        report = train_corners_all_lines(frame, lines=lines)
    except ValueError as exc:
        return CornersBenchmarkRunResult(
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

    return CornersBenchmarkRunResult(
        market=MARKET_NAME,
        rows=len(frame),
        status="benchmarked",
        lines_trained=[result.line for result in report.results.values()],
        details={"metrics_summary": report.metrics_summary(), "runs": run_metadata},
    )


def run_corners_benchmark_from_db(
    seasons: Optional[list[int]] = None,
    lines: tuple[float, ...] = DEFAULT_LINES,
    odds_market: str = ODDS_MARKET,
    save_model: bool = True,
) -> CornersBenchmarkRunResult:
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

    return run_corners_benchmark(matches=matches, lines=lines, odds_market=odds_market, save_model=save_model)

