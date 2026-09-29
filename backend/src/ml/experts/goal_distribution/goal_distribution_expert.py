"""Goal Distribution Expert (EXP-02).

Porta in pipeline ufficiale la logica Poisson gia' presente in
`src/service_ia/training/under_over/consistent/total_goals/OverUnderFromLambda.py`:
- sostituisce lo split casuale (train_test_split, vietato in produzione) con
  validazione temporale (`expanding_window_splits`),
- estende le soglie supportate a 1.5/2.5/3.5/4.5,
- espone la score distribution completa P(home=i, away=j), non solo le
  probabilita' Under/Over,
- confronta Poisson con Binomiale Negativa (metodo dei momenti) per rilevare
  overdispersion nei gol reali, come richiesto dal task.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import PoissonRegressor

from src.ml.validation.temporal_split import expanding_window_splits

EXPERT_NAME = "goal_distribution"
EXPERT_CONFIG_VERSION = 1
DEFAULT_THRESHOLDS: tuple[float, ...] = (1.5, 2.5, 3.5, 4.5)
MAX_GOALS_DEFAULT = 10
LEAGUE_AVG_GOALS_DEFAULT = 1.35


def _threshold_label(threshold: float) -> str:
    return f"over_{str(threshold).replace('.', '_')}"


def poisson_over_probabilities(lam: Any, thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """P(goal_totali > soglia) per ciascuna soglia.

    Monotone per costruzione: essendo la CDF di Poisson non decrescente in k,
    la sopravvivenza 1-CDF e' non crescente al crescere della soglia. Il
    `np.minimum` e' solo una protezione difensiva su micro-errori numerici.
    """
    lam_arr = np.clip(np.asarray(lam, dtype=float).reshape(-1), 1e-6, None)

    sorted_thresholds = sorted(set(float(t) for t in thresholds))
    columns: dict[str, np.ndarray] = {}
    previous: Optional[np.ndarray] = None
    for th in sorted_thresholds:
        k = int(np.floor(th))
        p_over = 1.0 - poisson.cdf(k, lam_arr)
        if previous is not None:
            p_over = np.minimum(p_over, previous)
        previous = p_over
        columns[_threshold_label(th)] = p_over

    return pd.DataFrame(columns)


def score_distribution_matrix(home_lambda: float, away_lambda: float, max_goals: int = MAX_GOALS_DEFAULT) -> pd.DataFrame:
    """Matrice P(home=i, away=j) assumendo Poisson indipendenti (baseline classico)."""
    home_lambda = max(float(home_lambda), 1e-6)
    away_lambda = max(float(away_lambda), 1e-6)

    goal_range = np.arange(0, max_goals + 1)
    home_pmf = poisson.pmf(goal_range, home_lambda)
    away_pmf = poisson.pmf(goal_range, away_lambda)

    matrix = np.outer(home_pmf, away_pmf)
    return pd.DataFrame(matrix, index=goal_range, columns=goal_range)


def total_goals_distribution(score_matrix: pd.DataFrame) -> pd.Series:
    """Distribuzione marginale dei gol totali per convoluzione dalla matrice scoreline."""
    max_total = int(score_matrix.index.max()) + int(score_matrix.columns.max())
    totals = pd.Series(0.0, index=range(0, max_total + 1))
    values = score_matrix.to_numpy()
    for i, home_goals in enumerate(score_matrix.index):
        for j, away_goals in enumerate(score_matrix.columns):
            totals[int(home_goals) + int(away_goals)] += values[i, j]
    return totals


def over_under_from_score_matrix(score_matrix: pd.DataFrame, thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS) -> dict[str, float]:
    totals = total_goals_distribution(score_matrix)
    result: dict[str, float] = {}
    for th in sorted(set(float(t) for t in thresholds)):
        k = int(np.floor(th))
        result[_threshold_label(th)] = float(totals[totals.index > k].sum())
    return result


@dataclass
class NegativeBinomialComparison:
    mean: float
    empirical_variance: float
    dispersion_index: float  # empirical_variance / mean; >1 indica overdispersion vs Poisson
    recommended_model: str  # "poisson" oppure "negative_binomial"
    nb_r: Optional[float]
    nb_p: Optional[float]


def compare_poisson_vs_negative_binomial(
    goals: Any,
    overdispersion_threshold: float = 1.15,
) -> NegativeBinomialComparison:
    """Confronto Poisson vs Binomiale Negativa (metodo dei momenti) su una serie di gol osservati.

    Poisson assume varianza == media. Se la varianza empirica supera
    sensibilmente la media (`dispersion_index > overdispersion_threshold`),
    la Binomiale Negativa e' raccomandata perche' modella l'overdispersion.
    """
    values = np.asarray(goals, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("Serie goal vuota: impossibile confrontare le distribuzioni")

    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=1)) if values.size > 1 else 0.0
    dispersion_index = (variance / mean) if mean > 0 else 1.0

    nb_r: Optional[float] = None
    nb_p: Optional[float] = None
    recommended = "poisson"

    if mean > 0 and dispersion_index > overdispersion_threshold:
        # Metodo dei momenti: mean = r(1-p)/p, var = r(1-p)/p^2  =>  p = mean/var, r = mean*p/(1-p)
        p_param = mean / variance
        if 0.0 < p_param < 1.0:
            r_param = mean * p_param / (1.0 - p_param)
            if r_param > 0:
                nb_r = float(r_param)
                nb_p = float(p_param)
                recommended = "negative_binomial"

    return NegativeBinomialComparison(
        mean=mean,
        empirical_variance=variance,
        dispersion_index=float(dispersion_index),
        recommended_model=recommended,
        nb_r=nb_r,
        nb_p=nb_p,
    )


def negative_binomial_over_probabilities(
    r: float,
    p: float,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, float]:
    """Alternativa a `poisson_over_probabilities` quando e' rilevata overdispersion."""
    result: dict[str, float] = {}
    previous: Optional[float] = None
    for th in sorted(set(float(t) for t in thresholds)):
        k = int(np.floor(th))
        p_over = float(1.0 - nbinom.cdf(k, r, p))
        if previous is not None:
            p_over = min(p_over, previous)
        previous = p_over
        result[_threshold_label(th)] = p_over
    return result


class GoalDistributionExpert:
    """Stima goal attesi (lambda) ed espone probabilita' U/O + score distribution versionate."""

    VERSION: str = ""  # valorizzato sotto la classe

    def __init__(self, thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS, max_goals: int = MAX_GOALS_DEFAULT):
        self.thresholds = tuple(sorted(set(float(t) for t in thresholds)))
        self.max_goals = max_goals
        self._regressor: Optional[PoissonRegressor] = None
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Regressore lambda con validazione temporale (sostituisce train_test_split legacy)
    # ------------------------------------------------------------------
    def fit_lambda_regressor(
        self,
        X: pd.DataFrame,
        y_total_goals: pd.Series,
        time_col_frame: pd.DataFrame,
        time_col: str = "prediction_at",
        n_splits: int = 5,
        min_train_size: int = 60,
        min_valid_size: int = 15,
    ) -> dict[str, Any]:
        splits = expanding_window_splits(
            frame=time_col_frame,
            time_col=time_col,
            n_splits=n_splits,
            min_train_size=min_train_size,
            min_valid_size=min_valid_size,
        )
        if not splits:
            raise ValueError("Dataset insufficiente per validazione temporale (nessuno split valido)")

        fold_reports: list[dict[str, Any]] = []
        for train_idx, valid_idx in splits:
            model = PoissonRegressor(alpha=1.0, max_iter=2000)
            model.fit(X.iloc[train_idx], y_total_goals.iloc[train_idx])
            lam_pred = model.predict(X.iloc[valid_idx])
            y_valid = y_total_goals.iloc[valid_idx].to_numpy(dtype=float)

            mae = float(np.mean(np.abs(lam_pred - y_valid)))
            lam_safe = np.clip(lam_pred, 1e-6, None)
            mean_log_likelihood = float(np.mean(poisson.logpmf(y_valid.astype(int), lam_safe)))
            fold_reports.append({"mae": mae, "mean_log_likelihood": mean_log_likelihood, "valid_size": len(valid_idx)})

        # Fit finale su tutto lo storico disponibile, per l'uso in produzione.
        final_model = PoissonRegressor(alpha=1.0, max_iter=2000)
        final_model.fit(X, y_total_goals)
        self._regressor = final_model
        self._feature_names = X.columns.tolist()

        return {
            "cv_strategy": "expanding_window",
            "folds": fold_reports,
            "mean_mae": float(np.mean([f["mae"] for f in fold_reports])),
            "mean_log_likelihood": float(np.mean([f["mean_log_likelihood"] for f in fold_reports])),
        }

    def predict_lambda(self, X: pd.DataFrame) -> np.ndarray:
        if self._regressor is None:
            raise RuntimeError("Regressore non addestrato: chiamare fit_lambda_regressor prima")
        return self._regressor.predict(X[self._feature_names])

    # ------------------------------------------------------------------
    # Stima deterministica (senza training) da rating EXP-01
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_lambdas_from_ratings(
        home_attack_rating: float,
        away_defense_rating: float,
        away_attack_rating: float,
        home_defense_rating: float,
        league_avg_goals: float = LEAGUE_AVG_GOALS_DEFAULT,
    ) -> tuple[float, float]:
        """Stima home/away lambda dai rating offensivo/difensivo di `TeamStrengthExpert` (EXP-01).

        Formula moltiplicativa classica: lambda_home = attack_home * defense_away / media_lega.
        """
        league_avg_goals = max(float(league_avg_goals), 1e-6)
        home_lambda = float(home_attack_rating) * (float(away_defense_rating) / league_avg_goals)
        away_lambda = float(away_attack_rating) * (float(home_defense_rating) / league_avg_goals)
        return max(home_lambda, 1e-6), max(away_lambda, 1e-6)

    # ------------------------------------------------------------------
    # Output completo, versionato
    # ------------------------------------------------------------------
    def build_expert_output(self, home_lambda: float, away_lambda: float) -> dict[str, Any]:
        score_matrix = score_distribution_matrix(home_lambda=home_lambda, away_lambda=away_lambda, max_goals=self.max_goals)
        combined_lambda = home_lambda + away_lambda

        over_probs = poisson_over_probabilities(lam=[combined_lambda], thresholds=self.thresholds).iloc[0].to_dict()
        over_probs_from_matrix = over_under_from_score_matrix(score_matrix=score_matrix, thresholds=self.thresholds)

        return {
            "expert_version": self.VERSION,
            "home_lambda": round(float(home_lambda), 4),
            "away_lambda": round(float(away_lambda), 4),
            "total_lambda": round(float(combined_lambda), 4),
            "over_under": {k: round(float(v), 6) for k, v in over_probs.items()},
            "over_under_from_score_matrix": {k: round(float(v), 6) for k, v in over_probs_from_matrix.items()},
            "score_matrix": score_matrix.round(6).to_dict(),
        }

    @staticmethod
    def _build_version() -> str:
        config = {
            "expert": EXPERT_NAME,
            "config_version": EXPERT_CONFIG_VERSION,
            "thresholds": DEFAULT_THRESHOLDS,
            "max_goals": MAX_GOALS_DEFAULT,
        }
        serialized = json.dumps(config, sort_keys=True)
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]


GoalDistributionExpert.VERSION = GoalDistributionExpert._build_version()
