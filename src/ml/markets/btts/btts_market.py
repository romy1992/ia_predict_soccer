"""BTTS (Both Teams To Score) consolidato (MARKET-03).

Confronta due modi di ottenere P(BTTS=Yes) gia' presenti nel progetto,
senza introdurre un nuovo training "from scratch":
- "score_distribution": derivato dalla score distribution matrix Poisson
  del Goal Distribution Expert (EXP-02): P(Yes) = P(home>=1 AND away>=1).
- "direct_expert": il classificatore binario "goal_no_goal" gia'
  addestrato/calibrato da `train_multi_market.py` ed esposto tramite
  `DirectMarketExpert` (EXP-05).

Questo modulo SOLO confronta (report benchmark), eventualmente fa
ensemble (media pesata) e ricalibra (Platt/isotonic 1D, OOF temporale)
il vincitore. Coerente con l'acceptance criteria "P(Yes)+P(No)=1": ogni
funzione che espone le due classi lo fa per costruzione (No = 1 - Yes),
mai tramite un secondo modello indipendente che potrebbe rompere la
somma a 1.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.ml.evaluation.probability_metrics import champion_probability_score, compute_probability_metrics
from src.ml.experts.direct.direct_market_expert import DirectMarketExpert
from src.ml.experts.goal_distribution.goal_distribution_expert import (
    LEAGUE_AVG_GOALS_DEFAULT,
    GoalDistributionExpert,
    score_distribution_matrix,
)
from src.ml.experts.team_strength.team_strength_expert import TeamStrengthExpert
from src.ml.validation.temporal_split import expanding_window_splits
from src.repository.match_repository import MatchRepository
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.model_registry import ModelRegistry
from src.service_ia.utility.utils import convert_orm_match_to_dict

BTTS_OUTCOMES: tuple[str, str] = ("Yes", "No")

# Mercato "diretto" gia' addestrato/calibrato (EXP-05/`train_multi_market.py`)
# che rappresenta la sorgente "direct_expert" del benchmark BTTS.
DIRECT_MARKET_NAME = "goal_no_goal"
MARKET_NAME = "btts"

# Colonne di metadati (non feature) prodotte da `FilterMarketService._build_row`.
_DIRECT_META_COLUMNS = ["id_fixture", "season", "league", "market", "prediction_at", "y"]


def score_distribution_btts_probability(home_lambda: float, away_lambda: float, max_goals: int = 10) -> float:
    """P(BTTS=Yes) = somma P(home=i, away=j) per i>=1 e j>=1 (score distribution, EXP-02)."""
    matrix = score_distribution_matrix(home_lambda=home_lambda, away_lambda=away_lambda, max_goals=max_goals)
    return float(matrix.iloc[1:, 1:].to_numpy().sum())


def btts_probabilities(p_yes: float) -> dict[str, float]:
    """P(Yes)+P(No)=1 per costruzione: No e' sempre il complemento di Yes."""
    p_yes = float(np.clip(p_yes, 0.0, 1.0))
    return {"Yes": p_yes, "No": 1.0 - p_yes}


def ensemble_probability(p_a: Any, p_b: Any, weight_a: float = 0.5) -> np.ndarray:
    """Media pesata tra due vettori di probabilita' P(Yes) (stesso ordine righe)."""
    weight_a = float(np.clip(weight_a, 0.0, 1.0))
    return weight_a * np.asarray(p_a, dtype=float) + (1.0 - weight_a) * np.asarray(p_b, dtype=float)


@dataclass
class RawProbabilityCalibrationResult:
    """Calibrazione (Platt/isotonic) di una probabilita' GIA' calcolata
    (non di un estimator sklearn multi-feature): stesso principio pre/post
    OOF temporale di ML-06, applicato a un punteggio scalare."""

    method: str
    sample_size: int
    pre_metrics: dict[str, Any]
    post_metrics: dict[str, Any]
    calibrator: Any


def calibrate_raw_probability_oof(
    raw_probability: Sequence[float],
    y_true: Sequence[int],
    cv_splits: list[tuple[list[int], list[int]]],
    method: str = "isotonic",
) -> RawProbabilityCalibrationResult:
    """Ricalibra una probabilita' scalare gia' calcolata, con validazione
    temporale (mai fit su dati futuri rispetto al fold di valutazione)."""
    raw = np.asarray(raw_probability, dtype=float).reshape(-1, 1)
    y = np.asarray(y_true, dtype=int)

    if raw.shape[0] == 0:
        raise ValueError("Serie vuota: impossibile calibrare BTTS")

    pre_metrics = compute_probability_metrics(y_true=y, probabilities=raw.reshape(-1), n_bins=10)

    def _fit_predict(method_name: str, train_idx: list[int], valid_idx: list[int]) -> np.ndarray:
        if method_name == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            calibrator.fit(raw[train_idx].reshape(-1), y[train_idx])
            return calibrator.predict(raw[valid_idx].reshape(-1))
        calibrator = LogisticRegression(max_iter=1000)
        calibrator.fit(raw[train_idx], y[train_idx])
        return calibrator.predict_proba(raw[valid_idx])[:, -1]

    oof_rows: list[dict[str, Any]] = []
    for train_idx, valid_idx in cv_splits:
        if not train_idx or not valid_idx:
            continue
        if len(np.unique(y[train_idx])) < 2:
            continue

        current_method = method
        try:
            calibrated = _fit_predict(current_method, train_idx, valid_idx)
        except Exception:
            current_method = "sigmoid"
            calibrated = _fit_predict(current_method, train_idx, valid_idx)

        for idx, p in zip(valid_idx, calibrated):
            oof_rows.append({"index": int(idx), "probability": float(np.clip(p, 0.0, 1.0)), "y_true": int(y[idx])})

    if not oof_rows:
        raise ValueError("Nessun fold valido per calibrazione BTTS")

    oof_frame = (
        pd.DataFrame(oof_rows)
        .groupby("index", as_index=False)
        .agg({"probability": "mean", "y_true": "first"})
        .sort_values(by=["index"])
    )
    post_metrics = compute_probability_metrics(
        y_true=oof_frame["y_true"].to_numpy(),
        probabilities=oof_frame["probability"].to_numpy(),
        n_bins=10,
    )

    # Calibratore finale fittato su tutto lo storico, per l'uso in produzione.
    if method == "isotonic":
        final_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        final_calibrator.fit(raw.reshape(-1), y)
    else:
        final_calibrator = LogisticRegression(max_iter=1000)
        final_calibrator.fit(raw, y)

    return RawProbabilityCalibrationResult(
        method=method,
        sample_size=int(len(y)),
        pre_metrics=pre_metrics,
        post_metrics=post_metrics,
        calibrator=final_calibrator,
    )


@dataclass
class BttsBenchmarkReport:
    """Report benchmark (acceptance criteria): metriche per ciascun
    approccio, il migliore selezionato e l'eventuale calibrazione finale."""

    approach_metrics: dict[str, dict[str, Any]]
    best_approach: str
    calibration: Optional[RawProbabilityCalibrationResult]


def benchmark_btts_approaches(
    y_true: Sequence[int],
    p_score_distribution: Sequence[float],
    p_direct_expert: Sequence[float],
    cv_splits: list[tuple[list[int], list[int]]],
    ensemble_weight: float = 0.5,
    calibrate_best: bool = True,
    calibration_method: str = "isotonic",
) -> BttsBenchmarkReport:
    """Confronta score_distribution vs direct_expert vs ensemble; sceglie
    il migliore per `champion_probability_score` (ML-05) e lo calibra."""
    y = np.asarray(y_true, dtype=int)
    p_sd = np.asarray(p_score_distribution, dtype=float)
    p_de = np.asarray(p_direct_expert, dtype=float)
    p_ens = ensemble_probability(p_sd, p_de, weight_a=ensemble_weight)

    candidates: dict[str, np.ndarray] = {
        "score_distribution": p_sd,
        "direct_expert": p_de,
        "ensemble": p_ens,
    }

    approach_metrics: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for name, probs in candidates.items():
        metrics = compute_probability_metrics(y_true=y, probabilities=probs, n_bins=10)
        predicted = (probs >= 0.5).astype(int)
        f1_weighted = float(f1_score(y, predicted, average="weighted", zero_division=0))
        selection_score = champion_probability_score(metrics=metrics, f1_weighted=f1_weighted)
        approach_metrics[name] = {**metrics, "f1_weighted": f1_weighted, "selection_score": selection_score}
        scores[name] = selection_score

    best_approach = max(scores.items(), key=lambda kv: kv[1])[0]

    calibration_result: Optional[RawProbabilityCalibrationResult] = None
    if calibrate_best:
        calibration_result = calibrate_raw_probability_oof(
            raw_probability=candidates[best_approach],
            y_true=y,
            cv_splits=cv_splits,
            method=calibration_method,
        )

    return BttsBenchmarkReport(
        approach_metrics=approach_metrics,
        best_approach=best_approach,
        calibration=calibration_result,
    )


# ---------------------------------------------------------------------------
# Dataset builder end-to-end: allinea le due sorgenti di P(BTTS=Yes) su dati
# reali, riusando SOLO mattoni gia' validati (nessuna nuova logica di feature
# extraction):
# - rating point-in-time (`TeamStrengthExpert`, EXP-01);
# - score distribution Poisson (`GoalDistributionExpert`, EXP-02);
# - dataset/feature 'goal_no_goal' gia' usato per il direct expert (`FilterMarketService`,
#   stesso builder di `train_multi_market.py`/EXP-05).
# ---------------------------------------------------------------------------


def build_goal_no_goal_frame_from_records(matches: list[dict[str, Any]]) -> pd.DataFrame:
    """Dataset (feature odds/mean_stats + target BTTS reale) per il mercato
    diretto 'goal_no_goal': STESSA estrazione feature di `FilterMarketService`
    (nessuna duplicazione di logica), ordinato temporalmente."""
    service = FilterMarketService()
    rows = [service._build_row(match=match, market=DIRECT_MARKET_NAME, with_target=True) for match in matches]
    rows = [row for row in rows if row]
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)
    frame["id_fixture"] = frame["id_fixture"].astype(int)
    frame["prediction_at"] = pd.to_datetime(frame["prediction_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)
    return frame


def _score_distribution_probabilities_from_ratings(
    ratings_frame: pd.DataFrame,
    league_avg_goals: float = LEAGUE_AVG_GOALS_DEFAULT,
    max_goals: int = 10,
) -> pd.DataFrame:
    """P(BTTS=Yes) via score distribution (EXP-02) da rating point-in-time
    gia' calcolati (EXP-01): nessun training, nessun leakage aggiuntivo."""
    if ratings_frame.empty:
        return pd.DataFrame(columns=["id_fixture", "p_score_distribution"])

    probabilities: list[float] = []
    for _, row in ratings_frame.iterrows():
        home_lambda, away_lambda = GoalDistributionExpert.estimate_lambdas_from_ratings(
            home_attack_rating=row["home_team_attack_rating"],
            away_defense_rating=row["away_team_defense_rating"],
            away_attack_rating=row["away_team_attack_rating"],
            home_defense_rating=row["home_team_defense_rating"],
            league_avg_goals=league_avg_goals,
        )
        probabilities.append(score_distribution_btts_probability(home_lambda, away_lambda, max_goals=max_goals))

    return pd.DataFrame(
        {"id_fixture": ratings_frame["id_fixture"].astype(int).to_numpy(), "p_score_distribution": probabilities}
    )


def build_btts_evaluation_frame(
    matches: list[dict[str, Any]],
    direct_expert: Any,
    league_avg_goals: float = LEAGUE_AVG_GOALS_DEFAULT,
    max_goals: int = 10,
) -> pd.DataFrame:
    """Allinea (per fixture) le due sorgenti di P(BTTS=Yes) + target reale.

    `matches` dovrebbe includere l'intero storico disponibile (non solo il
    sottoinsieme con odds 'goal_no_goal'): i rating point-in-time (EXP-01)
    sono piu' affidabili con piu' storico, mentre il dataset 'goal_no_goal'
    scarta comunque da solo le fixture senza quote disponibili.

    `direct_expert` deve esporre `.predict_proba(X) -> array 1D` (stessa
    interfaccia di `DirectMarketExpert`, EXP-05) ed eventualmente
    `.feature_names` per il riordino colonne.
    """
    direct_frame = build_goal_no_goal_frame_from_records(matches)
    if direct_frame.empty:
        return pd.DataFrame()

    ratings_frame = TeamStrengthExpert().build_ratings_dataset(matches)
    score_frame = _score_distribution_probabilities_from_ratings(
        ratings_frame=ratings_frame, league_avg_goals=league_avg_goals, max_goals=max_goals
    )
    if score_frame.empty:
        return pd.DataFrame()

    feature_names = list(getattr(direct_expert, "feature_names", None) or [])
    if not feature_names:
        feature_names = [col for col in direct_frame.columns if col not in _DIRECT_META_COLUMNS]
    X_direct = direct_frame.reindex(columns=feature_names, fill_value=0.0)
    direct_frame = direct_frame.assign(
        p_direct_expert=np.asarray(direct_expert.predict_proba(X_direct), dtype=float)
    )

    merged = direct_frame.merge(score_frame, on="id_fixture", how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged = merged.sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)
    return merged[["id_fixture", "season", "league", "prediction_at", "y", "p_score_distribution", "p_direct_expert"]]


@dataclass
class BttsBenchmarkRunResult:
    """Esito dell'orchestrazione end-to-end (dataset reale -> report benchmark),
    acceptance criteria 'Report benchmark' prodotto su dati veri (non solo su
    vettori di probabilita' gia' calcolati a mano)."""

    market: str
    rows: int
    status: str
    best_approach: Optional[str]
    details: dict[str, Any]


def run_btts_benchmark(
    matches: list[dict[str, Any]],
    direct_expert: Any,
    ensemble_weight: float = 0.5,
    calibration_method: str = "isotonic",
    save_model: bool = True,
) -> BttsBenchmarkRunResult:
    """Costruisce il dataset reale, esegue `benchmark_btts_approaches` e, se
    richiesto, registra il calibratore vincente come 'candidate' (mai
    'production' automaticamente, coerente col vincolo generale del progetto:
    la promozione resta un passo separato/manuale, vedi OPS-02)."""
    frame = build_btts_evaluation_frame(matches=matches, direct_expert=direct_expert)
    if frame.empty:
        return BttsBenchmarkRunResult(
            market=MARKET_NAME, rows=0, status="skipped_no_data", best_approach=None, details={}
        )

    min_train = max(30, int(len(frame) * 0.45))
    min_valid = max(10, int(len(frame) * 0.1))
    cv_splits = expanding_window_splits(
        frame=frame, time_col="prediction_at", n_splits=5, min_train_size=min_train, min_valid_size=min_valid
    )
    if not cv_splits:
        return BttsBenchmarkRunResult(
            market=MARKET_NAME,
            rows=len(frame),
            status="skipped_insufficient_rows_for_temporal_cv",
            best_approach=None,
            details={},
        )

    report = benchmark_btts_approaches(
        y_true=frame["y"].astype(int).to_numpy(),
        p_score_distribution=frame["p_score_distribution"].to_numpy(),
        p_direct_expert=frame["p_direct_expert"].to_numpy(),
        cv_splits=cv_splits,
        ensemble_weight=ensemble_weight,
        calibrate_best=True,
        calibration_method=calibration_method,
    )

    calibration_payload = None
    run_metadata = None
    if report.calibration is not None:
        calibration_payload = {
            "method": report.calibration.method,
            "sample_size": report.calibration.sample_size,
            "pre_metrics": report.calibration.pre_metrics,
            "post_metrics": report.calibration.post_metrics,
        }

        if save_model:
            calibrator_path = os.path.abspath(os.path.join("best_models", f"{MARKET_NAME}_champion_calibrator.pkl"))
            os.makedirs(os.path.dirname(calibrator_path), exist_ok=True)
            joblib.dump(report.calibration.calibrator, calibrator_path)

            registry = ModelRegistry()
            run_metadata = registry.register(
                model_path=calibrator_path,
                market=MARKET_NAME,
                model_name=report.best_approach,
                feature_names=[],
                metrics={
                    "selection_score": report.approach_metrics[report.best_approach]["selection_score"],
                    "pre_log_loss": report.calibration.pre_metrics.get("log_loss"),
                    "post_log_loss": report.calibration.post_metrics.get("log_loss"),
                    "pre_brier": report.calibration.pre_metrics.get("brier"),
                    "post_brier": report.calibration.post_metrics.get("brier"),
                },
                extra={
                    "classification_type": "binary",
                    "classes": list(BTTS_OUTCOMES),
                    "best_approach": report.best_approach,
                    "approach_metrics": report.approach_metrics,
                    "calibration_method": report.calibration.method,
                    "rows": len(frame),
                },
                stage="candidate",
            )

    return BttsBenchmarkRunResult(
        market=MARKET_NAME,
        rows=len(frame),
        status="benchmarked",
        best_approach=report.best_approach,
        details={
            "approach_metrics": report.approach_metrics,
            "calibration": calibration_payload,
            "run": run_metadata,
        },
    )


def run_btts_benchmark_from_db(
    seasons: Optional[list[int]] = None,
    direct_expert: Optional[Any] = None,
    ensemble_weight: float = 0.5,
    calibration_method: str = "isotonic",
    save_model: bool = True,
) -> BttsBenchmarkRunResult:
    """Variante DB reale: `direct_expert` di default e' il modello 'goal_no_goal'
    in stage 'production' (EXP-05); solleva `LookupError` se non ancora promosso
    (nessuna promozione automatica, vedi vincolo generale/OPS-02).

    Nota: i match vengono filtrati anche su odds/mean_statistics disponibili
    (stesso requisito del dataset diretto): i rating EXP-01 sono quindi
    calcolati sul sottoinsieme con quote disponibili, non sull'intero storico
    grezzo. Miglioria possibile in un task futuro se necessario.
    """
    resolved_direct_expert = direct_expert or DirectMarketExpert.load_production(market=DIRECT_MARKET_NAME)

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

    return run_btts_benchmark(
        matches=matches,
        direct_expert=resolved_direct_expert,
        ensemble_weight=ensemble_weight,
        calibration_method=calibration_method,
        save_model=save_model,
    )


