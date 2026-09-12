from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.evaluation.probability_metrics import (
    champion_probability_score,
    compute_probability_metrics,
    grouped_probability_report,
    temporal_oof_probabilities,
)
from src.ml.calibration.calibration_service import CalibrationService
from src.ml.validation.temporal_split import expanding_window_splits
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


def _build_temporal_cv(df: pd.DataFrame) -> Optional[list[tuple[list[int], list[int]]]]:
    if df.empty or "prediction_at" not in df.columns:
        return None

    min_train = max(30, int(len(df) * 0.45))
    min_valid = max(10, int(len(df) * 0.1))
    splits = expanding_window_splits(
        frame=df,
        time_col="prediction_at",
        n_splits=5,
        min_train_size=min_train,
        min_valid_size=min_valid,
    )
    if not splits:
        return None
    return splits


def _filter_valid_splits(y: pd.Series, splits: list[tuple[list[int], list[int]]]) -> list[tuple[list[int], list[int]]]:
    valid_splits: list[tuple[list[int], list[int]]] = []
    for train_idx, valid_idx in splits:
        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue
        if y.iloc[train_idx].nunique() < 2:
            continue
        if y.iloc[valid_idx].nunique() < 2:
            continue
        valid_splits.append((train_idx, valid_idx))
    return valid_splits


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


def _extract_selected_features(estimator: Any, feature_names: list[str]) -> list[str]:
    if not feature_names:
        return []

    selector = None
    if hasattr(estimator, "named_steps"):
        selector = estimator.named_steps.get("selector")

    if selector is None or not hasattr(selector, "get_support"):
        return feature_names

    try:
        support = selector.get_support()
    except Exception:
        return feature_names

    if len(support) != len(feature_names):
        return feature_names
    return [name for name, keep in zip(feature_names, support) if bool(keep)]


def _evaluate_estimator(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[tuple[list[int], list[int]]],
    market: str,
    season_series: pd.Series,
    league_series: pd.Series,
) -> tuple[dict[str, Any], float]:
    oof_frame = temporal_oof_probabilities(
        estimator=estimator,
        X=X,
        y=y,
        cv_splits=cv_splits,
    )
    if oof_frame.empty:
        raise ValueError("Nessun fold valido per calcolo metriche probabilistiche")

    p1 = oof_frame["probability"].astype(float).to_numpy()
    y_eval = oof_frame["y_true"].astype(int).to_numpy()
    probability_metrics = compute_probability_metrics(y_true=y_eval, probabilities=p1, n_bins=10)

    report_frame = pd.DataFrame(
        {
            "market": [market] * len(oof_frame),
            "season": season_series.iloc[oof_frame["index"].to_numpy()].to_numpy(),
            "league": league_series.iloc[oof_frame["index"].to_numpy()].to_numpy(),
            "y": y_eval,
            "probability": p1,
        }
    )
    grouped_report = grouped_probability_report(
        frame=report_frame,
        probability_col="probability",
        target_col="y",
        group_cols=["market", "season", "league"],
        n_bins=10,
    )

    score = champion_probability_score(
        metrics=probability_metrics,
        f1_weighted=float(f1_score(y_eval, (p1 >= 0.5).astype(int), average="weighted", zero_division=0)),
    )
    return {"probability_metrics": probability_metrics, "grouped_metrics": grouped_report}, float(score)


def _model_space(selection_method: str, feature_count: int) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    selector_logistic = FeatureSelectionService.build_selector(selection_method, feature_count)
    selector_rf = FeatureSelectionService.build_selector(selection_method, feature_count)
    selector_rf_smote = FeatureSelectionService.build_selector(selection_method, feature_count)

    selector_grid_logistic: dict[str, list[Any]] = {}
    selector_grid_rf: dict[str, list[Any]] = {}
    selector_grid_rf_smote: dict[str, list[Any]] = {}

    if isinstance(selector_logistic, SelectKBest):
        selector_grid_logistic = {
            "selector__k": sorted(set([max(1, min(feature_count, value)) for value in [10, 20, 30]]))
        }
        selector_grid_rf = {
            "selector__k": sorted(set([max(1, min(feature_count, value)) for value in [10, 20, 30]]))
        }
        selector_grid_rf_smote = {
            "selector__k": sorted(set([max(1, min(feature_count, value)) for value in [10, 20, 30]]))
        }
    elif isinstance(selector_logistic, RFE):
        selector_grid_logistic = {
            "selector__n_features_to_select": sorted(set([max(1, min(feature_count, value)) for value in [8, 12, 20]]))
        }
        selector_grid_rf = {
            "selector__n_features_to_select": sorted(set([max(1, min(feature_count, value)) for value in [8, 12, 20]]))
        }
        selector_grid_rf_smote = {
            "selector__n_features_to_select": sorted(set([max(1, min(feature_count, value)) for value in [8, 12, 20]]))
        }

    return {
        "logistic": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("selector", selector_logistic),
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
                **selector_grid_logistic,
                "model__C": [0.05, 0.1, 0.5, 1.0, 2.0],
                "model__solver": ["lbfgs", "liblinear"],
            },
        ),
        "random_forest": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("selector", selector_rf),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=200,
                            random_state=42,
                            n_jobs=-1,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            {
                **selector_grid_rf,
                # Griglia alleggerita (era n_estimators=500 fisso x max_depth
                # 4 valori x min_samples_split 3 x min_samples_leaf 3 = 36
                # combinazioni x selector_k x 5 fold): su dataset con 13+
                # stagioni storiche mandava il processo in MemoryError prima
                # di completare (nessun modello mai salvato). Ora
                # n_estimators=200 (fisso, sopra) x 2x2x2=8 combinazioni,
                # ~11x meno lavoro totale, stessa logica di selezione.
                "model__max_depth": [8, 16],
                "model__min_samples_split": [2, 10],
                "model__min_samples_leaf": [1, 4],
            },
        ),
        # Candidato aggiuntivo (richiesto esplicitamente dall'operatore,
        # 2026-09-08): SMOTE al posto di class_weight="balanced" per lo
        # sbilanciamento di classe. SMOTE esisteva gia' come dipendenza
        # (imbalanced-learn) ma solo nel codice legacy pre-v2
        # (src/service_ia/training/under_over/{inconsistent,consistent}/,
        # non piu' importato da nulla), mai confrontato in QUESTA pipeline.
        # Confronto controllato (scripts/analysis/test_smote_vs_class_weight.py,
        # dati reali Under/Over): aiuta in proporzione a quanto il mercato e'
        # sbilanciato - miglioramento netto su classi minoritarie sotto il
        # 15-20% (selection_score +0.016, log_loss -5%, brier -6% su
        # under_over_4_5, minoranza 13.5%), marginale o nullo su mercati
        # gia' quasi bilanciati (under_over_2_5, ~47%/53%). Per questo NON
        # sostituisce "random_forest" (class_weight="balanced"): e' un
        # candidato IN PIU' che compete sullo stesso selection_score - vince
        # solo dove aiuta davvero, invece di essere imposto ovunque.
        # `imblearn.pipeline.Pipeline` (non quella sklearn usata sopra)
        # applica SMOTE SOLO durante il fit del training fold di ciascun
        # fold - MAI al validation fold - stessa garanzia anti-leakage gia'
        # rispettata altrove nel progetto.
        "random_forest_smote": (
            ImbPipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("selector", selector_rf_smote),
                    ("smote", SMOTE(random_state=42)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=200,
                            random_state=42,
                            n_jobs=-1,
                            class_weight=None,
                        ),
                    ),
                ]
            ),
            {
                **selector_grid_rf_smote,
                "model__max_depth": [8, 16],
                "model__min_samples_split": [2, 10],
                "model__min_samples_leaf": [1, 4],
            },
        ),
    }


@dataclass
class ModelSearchResult:
    """Esito della ricerca modello (grid search + ensemble + champion
    selection) su UN (X, y, cv_splits) - indipendente da come X/y sono stati
    costruiti, cosi' riusabile anche fuori da `train_market()` (es. per
    Corners/Cards a linea configurabile, MARKET-05/06, 2026-09-12: stessa
    identica ricerca, un'istanza per linea, sullo stesso frame condiviso)."""

    champion_name: str
    champion_estimator: Any
    model_results: dict[str, dict[str, Any]]
    fitted_estimators: dict[str, Any]


def _select_champion_via_model_search(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[tuple[list[int], list[int]]],
    market: str,
    season_series: pd.Series,
    league_series: pd.Series,
    selection_method: str = "kbest",
) -> ModelSearchResult:
    """Grid search su `_model_space` (logistic/random_forest/
    random_forest_smote) + ensemble (voting/stacking) sui 2 migliori
    candidati + selezione del champion per `selection_score` - ESTRATTO
    (2026-09-12, comportamento INVARIATO) da `train_market()`, che ora la
    richiama qui sotto. Nessuna logica cambiata: stesso `_model_space`,
    stesso `GridSearchCV`, stesso criterio di ranking."""
    scorer = make_scorer(f1_score, average="weighted", zero_division=0)
    model_results: dict[str, dict[str, Any]] = {}
    fitted_estimators: dict[str, Any] = {}

    for model_name, (pipeline, grid) in _model_space(selection_method=selection_method, feature_count=X.shape[1]).items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring=scorer,
            cv=cv_splits,
            n_jobs=-1,
            verbose=0,
        )
        # backend "threading" (non il default "loky" a processi): evita di
        # nidificare due livelli di parallelismo a PROCESSI separati
        # (GridSearchCV + RandomForestClassifier, entrambi n_jobs=-1), che
        # su dataset ampi (13+ stagioni storiche) satura CPU/memoria della
        # macchina fino a far uccidere il processo (OOM/SIGKILL esterno).
        # Stesso fix gia' applicato in src/ml/markets/market_1x2.py.
        with parallel_backend("threading", n_jobs=-1):
            search.fit(X, y)
        best_estimator = search.best_estimator_
        fitted_estimators[model_name] = best_estimator

        prob_report, ranking_score = _evaluate_estimator(
            estimator=best_estimator,
            X=X,
            y=y,
            cv_splits=cv_splits,
            market=market,
            season_series=season_series,
            league_series=league_series,
        )
        model_results[model_name] = {
            "best_cv_f1": _safe_float(search.best_score_),
            "selection_score": ranking_score,
            "best_params": _to_serializable_dict(search.best_params_),
            **prob_report,
        }

    # 2) Ensemble (voting + stacking) costruiti sui 2 migliori modelli base
    ranked = sorted(model_results.items(), key=lambda kv: kv[1].get("selection_score", -1.0), reverse=True)
    top_names = [name for name, _ in ranked[:2]]

    if len(top_names) >= 2:
        est_a = fitted_estimators[top_names[0]]
        est_b = fitted_estimators[top_names[1]]

        voting = VotingClassifier(
            estimators=[(top_names[0], est_a), (top_names[1], est_b)],
            voting="soft",
            n_jobs=-1,
        )
        with parallel_backend("threading", n_jobs=-1):
            voting_score = cross_val_score(voting, X, y, scoring=scorer, cv=cv_splits, n_jobs=-1).mean()
            voting.fit(X, y)
        voting_prob_report, voting_ranking_score = _evaluate_estimator(
            estimator=voting,
            X=X,
            y=y,
            cv_splits=cv_splits,
            market=market,
            season_series=season_series,
            league_series=league_series,
        )

        model_results["voting"] = {
            "best_cv_f1": _safe_float(voting_score),
            "selection_score": voting_ranking_score,
            "best_params": {"base_models": top_names},
            **voting_prob_report,
        }
        fitted_estimators["voting"] = voting

        final_estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        stacking = StackingClassifier(
            estimators=[(top_names[0], est_a), (top_names[1], est_b)],
            final_estimator=final_estimator,
            stack_method="predict_proba",
            passthrough=True,
            n_jobs=-1,
            # cv intero (non `cv_splits`, il walk-forward esterno): l'interno
            # di StackingClassifier rigenera le meta-feature OOF con questo cv
            # su QUALSIASI X gli venga passato in fit() - qui sotto sia
            # l'intero X, sia sottoinsiemi piu' piccoli (cross_val_score con
            # cv=cv_splits chiama stacking.fit() su ogni train_idx, e
            # temporal_oof_probabilities/CalibrationService fanno lo stesso
            # per ciascun fold). `cv_splits` referenzia posizioni del dataset
            # COMPLETO: riusato come cv interno su un sottoinsieme piu'
            # piccolo produce split che non partizionano piu' i dati passati
            # ("cross_val_predict only works for partitions", ValueError
            # sistematico - MAI stato eseguito con successo finora).
            cv=5,
        )
        with parallel_backend("threading", n_jobs=-1):
            stacking_score = cross_val_score(stacking, X, y, scoring=scorer, cv=cv_splits, n_jobs=-1).mean()
            stacking.fit(X, y)
        stacking_prob_report, stacking_ranking_score = _evaluate_estimator(
            estimator=stacking,
            X=X,
            y=y,
            cv_splits=cv_splits,
            market=market,
            season_series=season_series,
            league_series=league_series,
        )

        model_results["stacking"] = {
            "best_cv_f1": _safe_float(stacking_score),
            "selection_score": stacking_ranking_score,
            "best_params": {"base_models": top_names, "passthrough": True},
            **stacking_prob_report,
        }
        fitted_estimators["stacking"] = stacking

    # 3) Champion selection
    champion_name, champion_payload = max(model_results.items(), key=lambda kv: kv[1].get("selection_score", -1.0))
    return ModelSearchResult(
        champion_name=champion_name,
        champion_estimator=fitted_estimators[champion_name],
        model_results=model_results,
        fitted_estimators=fitted_estimators,
    )


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

    if "prediction_at" in df.columns:
        df["prediction_at"] = pd.to_datetime(df["prediction_at"], utc=True, errors="coerce")
        df = df.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)
    else:
        df = df.sort_values(by=["season", "id_fixture"]).reset_index(drop=True)

    y = df["y"].astype(int)
    season_series = df["season"] if "season" in df.columns else pd.Series([None] * len(df))
    league_series = df["league"] if "league" in df.columns else pd.Series([None] * len(df))
    X = df.drop(columns=["y", "market"], errors="ignore")
    X = X.drop(columns=["id_fixture", "season", "league", "prediction_at"], errors="ignore")
    feature_names = X.columns.tolist()

    if X.empty:
        return MarketTrainResult(
            market=market,
            rows=len(df),
            status="skipped_no_feature_columns",
            champion=None,
            best_cv_f1=None,
            selected_features=[],
            details={"reason": "all feature columns removed"},
        )

    raw_splits = _build_temporal_cv(df)
    if raw_splits is None:
        return MarketTrainResult(
            market=market,
            rows=len(df),
            status="skipped_insufficient_rows_for_temporal_cv",
            champion=None,
            best_cv_f1=None,
            selected_features=[],
            details={"classes": y.value_counts().to_dict(), "reason": "no temporal splits"},
        )

    cv_splits = _filter_valid_splits(y=y, splits=raw_splits)
    if len(cv_splits) < 2:
        return MarketTrainResult(
            market=market,
            rows=len(df),
            status="skipped_invalid_temporal_folds",
            champion=None,
            best_cv_f1=None,
            selected_features=[],
            details={"classes": y.value_counts().to_dict(), "reason": "temporal folds with single-class train/valid"},
        )

    search_result = _select_champion_via_model_search(
        X=X,
        y=y,
        cv_splits=cv_splits,
        market=market,
        season_series=season_series,
        league_series=league_series,
        selection_method=selection_method,
    )
    model_results = search_result.model_results
    champion_name = search_result.champion_name
    champion_payload = model_results[champion_name]
    champion = search_result.champion_estimator
    champion_selected_features = _extract_selected_features(estimator=champion, feature_names=feature_names)
    champion_estimator = champion
    calibration_payload: dict[str, Any] = {
        "enabled": False,
        "method": None,
        "pre_metrics": None,
        "post_metrics": None,
        "sample_size": len(df),
        "calibrator_path": None,
    }

    try:
        calibration_result = CalibrationService.calibrate_estimator(
            estimator=champion,
            X=X,
            y=y,
            cv_splits=cv_splits,
        )
        champion_estimator = calibration_result.calibrator
        calibration_payload.update(
            {
                "enabled": True,
                "method": calibration_result.method,
                "sample_size": calibration_result.sample_size,
                "positives": calibration_result.positives,
                "negatives": calibration_result.negatives,
                "pre_metrics": calibration_result.pre_metrics,
                "post_metrics": calibration_result.post_metrics,
            }
        )
    except Exception as calibration_exc:
        calibration_payload.update({"error": str(calibration_exc)})

    if save_model:
        champion_prob_metrics = champion_payload.get("probability_metrics") or {}
        if calibration_payload.get("enabled"):
            calibrator_path = os.path.abspath(os.path.join("best_models", f"{market}_champion_calibrator.pkl"))
            os.makedirs(os.path.dirname(calibrator_path), exist_ok=True)
            joblib.dump(champion_estimator, calibrator_path)
            calibration_payload["calibrator_path"] = calibrator_path

        saver = SaveLoad(
            save_pkl=True,
            filename=f"{market}_champion",
            market_name=market,
            feature_names=feature_names,
            metrics={
                "selection_metric": "composite_probability_score",
                "selection_score": champion_payload.get("selection_score"),
                "best_cv_f1": champion_payload["best_cv_f1"],
                "log_loss": champion_prob_metrics.get("log_loss"),
                "brier": champion_prob_metrics.get("brier"),
                "ece": champion_prob_metrics.get("ece"),
                "auc": champion_prob_metrics.get("auc"),
                "calibration_enabled": calibration_payload.get("enabled"),
                "calibration_method": calibration_payload.get("method"),
                "pre_calibration_log_loss": ((calibration_payload.get("pre_metrics") or {}).get("log_loss") if calibration_payload.get("pre_metrics") else None),
                "post_calibration_log_loss": ((calibration_payload.get("post_metrics") or {}).get("log_loss") if calibration_payload.get("post_metrics") else None),
                "pre_calibration_brier": ((calibration_payload.get("pre_metrics") or {}).get("brier") if calibration_payload.get("pre_metrics") else None),
                "post_calibration_brier": ((calibration_payload.get("post_metrics") or {}).get("brier") if calibration_payload.get("post_metrics") else None),
                "model_family": champion_name,
                "rows": len(df),
                "selected_features_count": len(champion_selected_features),
            },
            registry_enabled=True,
        )
        saver.save_model(
            estimator=champion_estimator,
            model_name=champion_name,
            params=champion_payload.get("best_params"),
            extra={
                "selected_features": champion_selected_features,
                "selection_method": selection_method,
                "selection_in_pipeline": True,
                "grouped_metrics": champion_payload.get("grouped_metrics", []),
                "calibration": calibration_payload,
            },
        )

    return MarketTrainResult(
        market=market,
        rows=len(df),
        status="trained",
        champion=champion_name,
        best_cv_f1=champion_payload["best_cv_f1"],
        selected_features=champion_selected_features,
        details={
            "cv_strategy": "expanding_window",
            "cv_folds": len(cv_splits),
            "selection_method": selection_method,
            "selection_in_pipeline": True,
            "selection_metric": "composite_probability_score",
            "champion_selection_score": champion_payload.get("selection_score"),
            "input_features": len(feature_names),
            "selected_features": len(champion_selected_features),
            "calibration": calibration_payload,
            "models": model_results,
        },
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














