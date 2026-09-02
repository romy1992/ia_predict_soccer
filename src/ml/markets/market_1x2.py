"""Market 1X2 multiclass (MARKET-01).

Vero mercato 1X2: tre classi esclusive HOME/DRAW/AWAY le cui probabilita'
sommano sempre a 1 (nessun mapping draw->away, a differenza del
`DirectMarketExpert` per 'h2h', che resta VOLUTAMENTE binario, vedi EXP-05
e il relativo test `test_h2h_semantics_explicitly_marks_binary_not_1x2`).

Riusa mattoni gia' validati:
- estrazione feature quote/mean_statistics gia' presente in
  `FilterMarketService` (stessa fonte dati del mercato 'h2h' binario);
- validazione temporale (`expanding_window_splits`, ML-02);
- metriche/calibrazione MULTICLASSE dedicate (questo task), perche' i
  moduli binari (ML-05/ML-06) non sono applicabili a 3 classi;
- bookmaker baseline gia' esistente (ML-04, gia' pensata per 3 outcome sul
  mercato 'h2h': Home/Draw/Away).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.baselines.bookmaker_baseline import compute_market_baseline
from src.ml.calibration.multiclass_calibration_service import MulticlassCalibrationService
from src.ml.evaluation.multiclass_probability_metrics import (
    compute_multiclass_probability_metrics,
    multiclass_champion_score,
    reorder_probabilities_to_labels,
    temporal_oof_multiclass_probabilities,
)
from src.ml.validation.temporal_split import expanding_window_splits
from src.service_ia.pre_processing.feature_selection import FeatureSelectionService
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.model_registry import ModelRegistry

MARKET_NAME = "1x2"

# Ordine canonico ESPLICITO delle classi. Non e' un mapping numerico
# implicito 0/1/2 e non dipende dall'ordinamento alfabetico interno di
# sklearn (`estimator.classes_`): ogni funzione che produce una matrice di
# probabilita' la rimappa esplicitamente su questo ordine tramite
# `reorder_probabilities_to_labels`. Nessun mapping draw->away e' presente
# in nessun punto di questo modulo.
OUTCOME_LABELS: tuple[str, str, str] = ("HOME", "DRAW", "AWAY")

_META_COLUMNS = ["y", "market", "id_fixture", "season", "league", "prediction_at", "raw_h2h_odds"]


def label_1x2(home_goals: Any, away_goals: Any) -> Optional[str]:
    """Etichetta 1X2 dal risultato reale (mai dalle quote). HOME/DRAW/AWAY."""
    if home_goals is None or away_goals is None:
        return None
    home_goals, away_goals = int(home_goals), int(away_goals)
    if home_goals > away_goals:
        return "HOME"
    if home_goals < away_goals:
        return "AWAY"
    return "DRAW"


def bookmaker_baseline_from_h2h_odds(market_odds: dict[str, Any]) -> dict[str, Any]:
    """Fair probabilities Home/Draw/Away dal dict 'h2h' grezzo (riusa ML-04).

    Le chiavi del dict odds seguono lo schema storico `{outcome}_{bookmaker}`
    (es. "home_bet365"): l'outcome e' il prefisso fino all'ULTIMO underscore.
    """
    buckets: dict[str, list[float]] = {}
    for key, raw_odd in (market_odds or {}).items():
        outcome_prefix, sep, _bookmaker = str(key).rpartition("_")
        outcome_key = (outcome_prefix if sep else key).strip().lower()
        try:
            odd = float(raw_odd)
        except (TypeError, ValueError):
            continue
        if odd <= 0:
            continue
        buckets.setdefault(outcome_key, []).append(odd)

    odds_rows = [
        {"outcome": outcome, "avg_odd": float(np.mean(values)), "bookmakers": len(values)}
        for outcome, values in buckets.items()
        if outcome in {"home", "draw", "away"}
    ]
    return compute_market_baseline(market="h2h", odds_rows=odds_rows)


def _build_row(match: dict[str, Any]) -> Optional[dict[str, Any]]:
    odds_list = match.get("odds") or []
    if not odds_list:
        return None

    market_odds = (odds_list[0] or {}).get("h2h")
    if not isinstance(market_odds, dict) or len(market_odds) == 0:
        return None

    stat_home, stat_away = FilterMarketService._resolve_team_stats(match=match, with_full_stats=True)
    if not stat_home or not stat_away:
        return None

    outcome = label_1x2(stat_home.get("score_ft"), stat_away.get("score_ft"))
    if outcome is None:
        return None

    row: dict[str, Any] = {
        "id_fixture": match.get("id_fixture"),
        "season": match.get("season"),
        "league": match.get("current_league"),
        "market": MARKET_NAME,
        "prediction_at": match.get("date_match"),
        "raw_h2h_odds": market_odds,
    }
    row.update(FilterMarketService._extract_market_odds_features(market_odds))
    row.update(FilterMarketService._extract_mean_features(match))

    # Senza feature utili (solo colonne di metadati) non ha senso produrre la riga.
    if len(row) <= 6:
        return None

    row["y"] = outcome
    return row


def _finalize_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    numeric_columns = [col for col in frame.columns if col not in _META_COLUMNS]
    frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

    frame["prediction_at"] = pd.to_datetime(frame["prediction_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)
    return frame


def build_1x2_dataset_from_records(matches: list[dict[str, Any]]) -> pd.DataFrame:
    """Dataset 1X2 da record 'match' gia' in memoria (nessuna query DB: usato dai test)."""
    rows = [row for row in (_build_row(match) for match in matches) if row]
    return _finalize_frame(rows)


def build_1x2_dataset_from_db(seasons: Optional[list[int]] = None) -> pd.DataFrame:
    """Dataset 1X2 dal DB reale (stessa query gia' validata di FilterMarketService.build_dataset)."""
    service = FilterMarketService()
    matches = service._search_matches(seasons=seasons, status="FT")
    rows = [row for row in (_build_row(match) for match in matches) if row]
    return _finalize_frame(rows)


@dataclass
class Market1X2TrainResult:
    market: str
    rows: int
    status: str
    champion: Optional[str]
    best_score: Optional[float]
    selected_features: list[str]
    details: dict[str, Any]


def _build_temporal_cv(df: pd.DataFrame) -> Optional[list[tuple[list[int], list[int]]]]:
    if df.empty or "prediction_at" not in df.columns:
        return None

    min_train = max(30, int(len(df) * 0.45))
    min_valid = max(10, int(len(df) * 0.1))
    splits = expanding_window_splits(
        frame=df, time_col="prediction_at", n_splits=5, min_train_size=min_train, min_valid_size=min_valid
    )
    return splits or None


def _filter_valid_splits(y: pd.Series, splits: list[tuple[list[int], list[int]]]) -> list[tuple[list[int], list[int]]]:
    valid_splits: list[tuple[list[int], list[int]]] = []
    for train_idx, valid_idx in splits:
        if not train_idx or not valid_idx:
            continue
        if y.iloc[train_idx].nunique() < 2:
            continue
        valid_splits.append((train_idx, valid_idx))
    return valid_splits


def _model_space(feature_count: int, selection_method: str) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    selector_logistic = FeatureSelectionService.build_selector(selection_method, feature_count)
    selector_rf = FeatureSelectionService.build_selector(selection_method, feature_count)

    def _selector_grid(selector: Any) -> dict[str, list[Any]]:
        if hasattr(selector, "k"):
            return {"selector__k": sorted(set(max(1, min(feature_count, v)) for v in [10, 20]))}
        if hasattr(selector, "n_features_to_select"):
            return {"selector__n_features_to_select": sorted(set(max(1, min(feature_count, v)) for v in [8, 15]))}
        return {}

    return {
        "logistic": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("selector", selector_logistic),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
                ]
            ),
            {**_selector_grid(selector_logistic), "model__C": [0.1, 1.0]},
        ),
        "random_forest": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("selector", selector_rf),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=150, random_state=42, n_jobs=-1, class_weight="balanced"
                        ),
                    ),
                ]
            ),
            {**_selector_grid(selector_rf), "model__max_depth": [None, 12], "model__min_samples_leaf": [1, 4]},
        ),
    }


def _bookmaker_baseline_report(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "raw_h2h_odds" not in df.columns:
        return {"available_rows": 0}

    baselines = [bookmaker_baseline_from_h2h_odds(row) for row in df["raw_h2h_odds"].tolist()]
    baselines = [b for b in baselines if b.get("outcomes")]
    if not baselines:
        return {"available_rows": 0}

    overrounds = [b["overround"] for b in baselines if b.get("overround") is not None]
    return {
        "available_rows": len(baselines),
        "mean_overround": float(np.mean(overrounds)) if overrounds else None,
        "sample": baselines[0],
    }


def train_market_1x2(
    seasons: Optional[list[int]] = None,
    selection_method: str = "kbest",
    save_model: bool = True,
    frame: Optional[pd.DataFrame] = None,
) -> Market1X2TrainResult:
    """Addestra il vero modello 1X2 multiclasse (HOME/DRAW/AWAY).

    `frame` permette di iniettare un dataset gia' costruito (usato nei
    test): se assente, viene costruito dal DB reale.
    """
    df = frame if frame is not None else build_1x2_dataset_from_db(seasons=seasons)

    if df.empty:
        return Market1X2TrainResult(
            market=MARKET_NAME, rows=0, status="skipped_no_data", champion=None,
            best_score=None, selected_features=[], details={},
        )

    y = df["y"].astype(str)
    unexpected = set(y.unique()) - set(OUTCOME_LABELS)
    if unexpected:
        # Acceptance criteria: nessun mapping ammesso al di fuori di HOME/DRAW/AWAY.
        raise ValueError(f"Etichette 1X2 inattese: {unexpected}")

    X = df.drop(columns=_META_COLUMNS, errors="ignore")
    feature_names = X.columns.tolist()

    if X.empty:
        return Market1X2TrainResult(
            market=MARKET_NAME, rows=len(df), status="skipped_no_feature_columns", champion=None,
            best_score=None, selected_features=[], details={},
        )

    raw_splits = _build_temporal_cv(df)
    if raw_splits is None:
        return Market1X2TrainResult(
            market=MARKET_NAME, rows=len(df), status="skipped_insufficient_rows_for_temporal_cv", champion=None,
            best_score=None, selected_features=[], details={"classes": y.value_counts().to_dict()},
        )

    cv_splits = _filter_valid_splits(y=y, splits=raw_splits)
    if len(cv_splits) < 2:
        return Market1X2TrainResult(
            market=MARKET_NAME, rows=len(df), status="skipped_invalid_temporal_folds", champion=None,
            best_score=None, selected_features=[], details={"classes": y.value_counts().to_dict()},
        )

    scorer = make_scorer(f1_score, average="weighted", zero_division=0)
    model_results: dict[str, dict[str, Any]] = {}
    fitted_estimators: dict[str, Any] = {}

    for model_name, (pipeline, grid) in _model_space(feature_count=X.shape[1], selection_method=selection_method).items():
        # backend "threading" (non il default "loky" a processi): evita di
        # nidificare due livelli di parallelismo a PROCESSI separati
        # (GridSearchCV + RandomForestClassifier, entrambi n_jobs=-1), che su
        # questo ambiente causa TerminatedWorkerError/MemoryError nei worker.
        # Con thread condivisi il parallelismo resta attivo (niente fit
        # sequenziale) senza spawn di nuovi processi Python.
        search = GridSearchCV(estimator=pipeline, param_grid=grid, scoring=scorer, cv=cv_splits, n_jobs=-1, verbose=0)
        with parallel_backend("threading", n_jobs=-1):
            search.fit(X, y)
        best_estimator = search.best_estimator_
        fitted_estimators[model_name] = best_estimator

        oof = temporal_oof_multiclass_probabilities(
            estimator=best_estimator, X=X, y=y, cv_splits=cv_splits, class_labels=OUTCOME_LABELS
        )
        if oof.empty:
            continue

        prob_cols = [f"prob_{label}" for label in OUTCOME_LABELS]
        prob_matrix = oof[prob_cols].to_numpy()
        metrics = compute_multiclass_probability_metrics(
            y_true=oof["y_true"].tolist(), probabilities=prob_matrix, class_labels=OUTCOME_LABELS
        )
        predicted_labels = [OUTCOME_LABELS[i] for i in prob_matrix.argmax(axis=1)]
        f1_weighted = float(f1_score(oof["y_true"].tolist(), predicted_labels, average="weighted", zero_division=0))
        score = multiclass_champion_score(metrics=metrics, f1_weighted=f1_weighted)

        model_results[model_name] = {
            "probability_metrics": metrics,
            "f1_weighted": f1_weighted,
            "selection_score": score,
            "best_params": {
                k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v))
                for k, v in search.best_params_.items()
            },
        }

    if not model_results:
        return Market1X2TrainResult(
            market=MARKET_NAME, rows=len(df), status="skipped_no_valid_model", champion=None,
            best_score=None, selected_features=[], details={},
        )

    champion_name, champion_payload = max(model_results.items(), key=lambda kv: kv[1]["selection_score"])
    champion = fitted_estimators[champion_name]

    calibration_payload: dict[str, Any] = {"enabled": False}
    champion_estimator = champion
    try:
        calibration_result = MulticlassCalibrationService.calibrate_estimator(
            estimator=champion, X=X, y=y, cv_splits=cv_splits, class_labels=OUTCOME_LABELS
        )
        champion_estimator = calibration_result.calibrator
        calibration_payload = {
            "enabled": True,
            "method": calibration_result.method,
            "sample_size": calibration_result.sample_size,
            "class_counts": calibration_result.class_counts,
            "pre_metrics": calibration_result.pre_metrics,
            "post_metrics": calibration_result.post_metrics,
        }
    except Exception as calibration_exc:
        calibration_payload = {"enabled": False, "error": str(calibration_exc)}

    baseline_report = _bookmaker_baseline_report(df)

    run_metadata = None
    if save_model:
        model_path = os.path.abspath(os.path.join("best_models", f"{MARKET_NAME}_champion.pkl"))
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(champion_estimator, model_path)

        registry = ModelRegistry()
        run_metadata = registry.register(
            model_path=model_path,
            market=MARKET_NAME,
            model_name=champion_name,
            feature_names=feature_names,
            metrics={
                "selection_score": champion_payload["selection_score"],
                "f1_weighted": champion_payload["f1_weighted"],
                "pre_log_loss": (calibration_payload.get("pre_metrics") or {}).get("log_loss"),
                "post_log_loss": (calibration_payload.get("post_metrics") or {}).get("log_loss"),
                "pre_brier": (calibration_payload.get("pre_metrics") or {}).get("brier"),
                "post_brier": (calibration_payload.get("post_metrics") or {}).get("brier"),
            },
            extra={
                "classification_type": "multiclass",
                "classes": list(OUTCOME_LABELS),
                "calibration": calibration_payload,
                "bookmaker_baseline": baseline_report,
                "model_family": champion_name,
                "rows": len(df),
            },
            stage="candidate",
        )

    return Market1X2TrainResult(
        market=MARKET_NAME,
        rows=len(df),
        status="trained",
        champion=champion_name,
        best_score=champion_payload["selection_score"],
        selected_features=feature_names,
        details={
            "cv_strategy": "expanding_window",
            "cv_folds": len(cv_splits),
            "classification_type": "multiclass",
            "classes": list(OUTCOME_LABELS),
            "models": model_results,
            "calibration": calibration_payload,
            "bookmaker_baseline": baseline_report,
            "run": run_metadata,
        },
    )


@dataclass
class Market1x2Expert:
    """Interfaccia comune multiclasse: `predict_proba` ritorna SEMPRE 3
    colonne nell'ordine canonico `OUTCOME_LABELS` (HOME, DRAW, AWAY), che
    sommano a 1 per costruzione (garantito da `CalibratedClassifierCV` /
    qualunque classificatore sklearn con `predict_proba`)."""

    estimator: Any
    feature_names: list[str] = field(default_factory=list)
    run_id: Optional[str] = None
    stage: Optional[str] = None

    def __post_init__(self) -> None:
        if not hasattr(self.estimator, "predict_proba"):
            raise TypeError("L'estimator caricato non espone predict_proba: interfaccia comune non rispettata")

    def predict_proba(self, X: Any) -> np.ndarray:
        ordered_X = X[self.feature_names] if self.feature_names else X
        raw = self.estimator.predict_proba(ordered_X)
        classes = getattr(self.estimator, "classes_", OUTCOME_LABELS)
        return reorder_probabilities_to_labels(classes, raw, OUTCOME_LABELS)

    def predict_proba_dict(self, X: Any) -> list[dict[str, float]]:
        """Tre probabilita' esplicite per riga: {"HOME": .., "DRAW": .., "AWAY": ..}."""
        matrix = self.predict_proba(X)
        return [
            {label: float(matrix[row_idx, col_idx]) for col_idx, label in enumerate(OUTCOME_LABELS)}
            for row_idx in range(matrix.shape[0])
        ]

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = proba.argmax(axis=1)
        return np.asarray([OUTCOME_LABELS[i] for i in idx])

    @classmethod
    def load_production(cls, registry: Optional[Any] = None) -> "Market1x2Expert":
        """Carica lo stage 'production' (vincolo generale: latest != production)."""
        registry = registry or ModelRegistry()
        run = registry.get_production(market=MARKET_NAME)
        if run is None:
            raise LookupError(f"Nessun modello in stage 'production' per il mercato '{MARKET_NAME}'")
        return cls._from_run(run)

    @classmethod
    def load_latest(cls, registry: Optional[Any] = None) -> "Market1x2Expert":
        """ATTENZIONE: solo per debug/validazione, mai per servire predizioni reali."""
        registry = registry or ModelRegistry()
        run = registry.get_latest(market=MARKET_NAME)
        if run is None:
            raise LookupError(f"Nessun modello registrato per il mercato '{MARKET_NAME}'")
        return cls._from_run(run)

    @classmethod
    def _from_run(cls, run: dict[str, Any]) -> "Market1x2Expert":
        model_path = run.get("model_path")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path non trovato per il run {run.get('run_id')}: {model_path}")

        estimator = joblib.load(model_path)
        return cls(
            estimator=estimator,
            feature_names=list(run.get("feature_names") or []),
            run_id=run.get("run_id"),
            stage=run.get("current_stage") or run.get("stage"),
        )

    @classmethod
    def from_estimator(cls, estimator: Any, feature_names: Optional[list[str]] = None) -> "Market1x2Expert":
        """Costruzione diretta (utile nei test, senza toccare registry/disco)."""
        return cls(estimator=estimator, feature_names=list(feature_names or []))






