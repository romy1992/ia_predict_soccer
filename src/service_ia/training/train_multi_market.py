from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.service_ia.pre_processing.feature_selection import FeatureSelectionService
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.utility_training.save_load import SaveLoad

logging.basicConfig(level=logging.INFO)


@dataclass
class MarketTrainResult:
    market: str
    rows: int
    status: str
    champion: Optional[str]
    best_cv_f1: Optional[float]
    selected_features: list[str]
    details: dict[str, Any]


def _build_cv(y: pd.Series) -> Optional[StratifiedKFold]:
    class_counts = y.value_counts()
    if class_counts.empty or class_counts.min() < 2:
        return None

    splits = int(max(2, min(5, class_counts.min())))
    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _to_serializable_dict(values: dict[str, Any]) -> dict[str, Any]:
    serializable = {}
    for key, value in values.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serializable[key] = value
        else:
            serializable[key] = str(value)
    return serializable


def _model_space() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    return {
        "logistic": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            {
                "model__C": [0.05, 0.1, 0.5, 1.0, 2.0],
                "model__solver": ["lbfgs", "liblinear"],
            },
        ),
        "random_forest": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=500,
                            random_state=42,
                            n_jobs=-1,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            {
                "model__max_depth": [None, 8, 16, 24],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
    }


def train_market(
    market: str,
    seasons: Optional[list[int]] = None,
    selection_method: str = "kbest",
    save_model: bool = True,
) -> MarketTrainResult:
    service = FilterMarketService()
    df = service.build_dataset(market=market, seasons=seasons)

    if df.empty:
        return MarketTrainResult(
            market=market,
            rows=0,
            status="skipped_no_data",
            champion=None,
            best_cv_f1=None,
            selected_features=[],
            details={},
        )

    y = df["y"].astype(int)
    X = df.drop(columns=["y", "market"], errors="ignore")
    X = X.drop(columns=["id_fixture", "season"], errors="ignore")

    cv = _build_cv(y)
    if cv is None:
        return MarketTrainResult(
            market=market,
            rows=len(df),
            status="skipped_single_class_or_few_rows",
            champion=None,
            best_cv_f1=None,
            selected_features=[],
            details={"classes": y.value_counts().to_dict()},
        )

    # 1) Feature selection
    if selection_method == "rfe":
        selected = FeatureSelectionService.select_rfe(X=X, y=y, n_features_to_select=min(20, X.shape[1]))
    else:
        selected = FeatureSelectionService.select_k_best(X=X, y=y, k=min(30, X.shape[1]))

    X_selected = selected.X_selected
    feature_names = selected.selected_features

    scorer = make_scorer(f1_score, average="weighted", zero_division=0)
    model_results: dict[str, dict[str, Any]] = {}
    fitted_estimators: dict[str, Any] = {}

    for model_name, (pipeline, grid) in _model_space().items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_selected, y)
        fitted_estimators[model_name] = search.best_estimator_
        model_results[model_name] = {
            "best_cv_f1": _safe_float(search.best_score_),
            "best_params": _to_serializable_dict(search.best_params_),
        }

    # 2) Ensemble (voting + stacking) costruiti sui 2 migliori modelli base
    ranked = sorted(model_results.items(), key=lambda kv: kv[1]["best_cv_f1"], reverse=True)
    top_names = [name for name, _ in ranked[:2]]

    if len(top_names) >= 2:
        est_a = fitted_estimators[top_names[0]]
        est_b = fitted_estimators[top_names[1]]

        voting = VotingClassifier(
            estimators=[(top_names[0], est_a), (top_names[1], est_b)],
            voting="soft",
            n_jobs=-1,
        )
        voting_score = cross_val_score(voting, X_selected, y, scoring=scorer, cv=cv, n_jobs=-1).mean()
        voting.fit(X_selected, y)

        model_results["voting"] = {
            "best_cv_f1": _safe_float(voting_score),
            "best_params": {"base_models": top_names},
        }
        fitted_estimators["voting"] = voting

        final_estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        stacking = StackingClassifier(
            estimators=[(top_names[0], est_a), (top_names[1], est_b)],
            final_estimator=final_estimator,
            stack_method="predict_proba",
            passthrough=True,
            n_jobs=-1,
            cv=cv,
        )
        stacking_score = cross_val_score(stacking, X_selected, y, scoring=scorer, cv=cv, n_jobs=-1).mean()
        stacking.fit(X_selected, y)

        model_results["stacking"] = {
            "best_cv_f1": _safe_float(stacking_score),
            "best_params": {"base_models": top_names, "passthrough": True},
        }
        fitted_estimators["stacking"] = stacking

    # 3) Champion selection
    champion_name, champion_payload = max(model_results.items(), key=lambda kv: kv[1]["best_cv_f1"])
    champion = fitted_estimators[champion_name]

    if save_model:
        saver = SaveLoad(
            save_pkl=True,
            filename=f"{market}_champion",
            market_name=market,
            feature_names=feature_names,
            metrics={
                "best_cv_f1": champion_payload["best_cv_f1"],
                "model_family": champion_name,
                "rows": len(df),
            },
            registry_enabled=True,
        )
        saver.save_model(
            estimator=champion,
            model_name=champion_name,
            params=champion_payload.get("best_params"),
        )

    return MarketTrainResult(
        market=market,
        rows=len(df),
        status="trained",
        champion=champion_name,
        best_cv_f1=champion_payload["best_cv_f1"],
        selected_features=feature_names,
        details=model_results,
    )


def train_all_markets(
    markets: Optional[list[str]] = None,
    seasons: Optional[list[int]] = None,
    selection_method: str = "kbest",
    save_model: bool = True,
) -> list[MarketTrainResult]:
    markets = markets or [
        "h2h",
        "under_over_2_5",
        "goal_no_goal",
        "corners",
        "cards",
        "dc",
    ]

    results = []
    for market in markets:
        try:
            result = train_market(
                market=market,
                seasons=seasons,
                selection_method=selection_method,
                save_model=save_model,
            )
            logging.info("Market %s -> %s", market, result.status)
            results.append(result)
        except Exception as exc:
            logging.exception("Errore training mercato %s", market)
            results.append(
                MarketTrainResult(
                    market=market,
                    rows=0,
                    status="failed",
                    champion=None,
                    best_cv_f1=None,
                    selected_features=[],
                    details={"error": str(exc)},
                )
            )

    summary_path = os.path.abspath(os.path.join("best_models", "training_summary.json"))
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    serializable_results = [
        {
            "market": r.market,
            "rows": r.rows,
            "status": r.status,
            "champion": r.champion,
            "best_cv_f1": r.best_cv_f1,
            "selected_features": r.selected_features,
            "details": r.details,
        }
        for r in results
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    return results


if __name__ == "__main__":
    train_results = train_all_markets()
    for train_result in train_results:
        print(train_result)

