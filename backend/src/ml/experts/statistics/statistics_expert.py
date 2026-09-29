"""Statistics Expert (EXP-03).

Modello dedicato a forma/statistiche pre-match, volutamente SEPARATO dal
market expert: nessuna colonna 'odds' entra mai in questo modulo (garanzia
strutturale verificata anche a runtime, vedi `build_dataset_from_records`).

Fonte dati: `Match.mean_statistics`, che secondo il modello ORM rappresenta
"medie stagionali alla giornata corrente, cioe' PRIMA che iniziasse la
partita corrente" -> gia' point-in-time per costruzione, nessun leakage.

Espone:
- dataset point-in-time costruito SOLO dalle medie storiche pre-match,
- fit/valutazione con validazione temporale (framework metriche
  probabilistiche di ML-05: log_loss/brier/ece/auc),
- "embedding" numerico compatto (probabilita' calibrata) riusabile da altri
  modelli/esperti (Oracle ensemble), senza esporre logica scientifica.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.datasets.point_in_time_builder import FINAL_STATUSES
from src.ml.evaluation.probability_metrics import compute_probability_metrics, temporal_oof_probabilities
from src.ml.validation.temporal_split import expanding_window_splits
from src.repository.match_repository import MatchRepository
from src.service_ia.utility.utils import convert_orm_match_to_dict

EXPERT_NAME = "statistics"
SUPPORTED_OUTCOMES = {"home_win", "draw", "away_win", "btts", "over_2_5"}


def _normalize_feature_name(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (key or "").strip().lower()).strip("_")


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "."))
    except Exception:
        return 0.0


def extract_mean_statistics_features(match: dict[str, Any]) -> dict[str, float]:
    """Feature SOLO da `mean_statistics` (rolling pre-match, gia' point-in-time).

    Nessuna colonna 'odds' e nessuna statistica POST-match (score/shots/corner
    reali della partita corrente) entra qui: solo le medie storiche pre-partita.
    """
    mean_stats = match.get("mean_statistics")
    if not isinstance(mean_stats, list) or len(mean_stats) < 2:
        return {}

    id_home = match.get("id_team_home")
    id_away = match.get("id_team_away")
    home_row = next((row for row in mean_stats if row.get("id_team") == id_home), None)
    away_row = next((row for row in mean_stats if row.get("id_team") == id_away), None)
    if not home_row or not away_row:
        return {}

    keys = set(home_row.keys()).union(away_row.keys())
    features: dict[str, float] = {}
    for key in keys:
        if key == "id_team":
            continue
        normalized = _normalize_feature_name(key)
        if not normalized:
            continue

        home_value = _safe_float(home_row.get(key))
        away_value = _safe_float(away_row.get(key))
        features[f"stat_{normalized}_home"] = home_value
        features[f"stat_{normalized}_away"] = away_value
        features[f"stat_{normalized}_diff"] = home_value - away_value

    return features


def derive_outcome_labels(match: dict[str, Any]) -> Optional[dict[str, int]]:
    """Etichette derivate SOLO dal risultato reale (score_ft), mai dalle quote."""
    stats = match.get("statistics") or []
    id_home = match.get("id_team_home")
    id_away = match.get("id_team_away")
    stat_home = next((s for s in stats if s.get("statistics_team_id") == id_home), None)
    stat_away = next((s for s in stats if s.get("statistics_team_id") == id_away), None)
    if not stat_home or not stat_away:
        return None

    home_ft = stat_home.get("score_ft")
    away_ft = stat_away.get("score_ft")
    if home_ft is None or away_ft is None:
        return None

    home_ft, away_ft = int(home_ft), int(away_ft)
    total = home_ft + away_ft
    return {
        "home_win": int(home_ft > away_ft),
        "draw": int(home_ft == away_ft),
        "away_win": int(home_ft < away_ft),
        "btts": int(home_ft > 0 and away_ft > 0),
        "over_2_5": int(total > 2),
    }


class StatisticsExpert:
    """Modello pre-match basato esclusivamente su statistiche/forma (no odds)."""

    def __init__(self, match_repo: Optional[MatchRepository] = None):
        self.match_repo = match_repo or MatchRepository()
        self._pipeline: Optional[Pipeline] = None
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Dataset (nessuna dipendenza da odds/market expert)
    # ------------------------------------------------------------------
    def build_dataset_from_records(self, matches: list[dict[str, Any]], outcome: str = "home_win") -> pd.DataFrame:
        if outcome not in SUPPORTED_OUTCOMES:
            raise ValueError(f"Outcome non supportato: {outcome}")

        rows: list[dict[str, Any]] = []
        for match in matches:
            status = str(match.get("status") or "").upper()
            if status and status not in FINAL_STATUSES:
                continue

            features = extract_mean_statistics_features(match)
            if not features:
                continue
            labels = derive_outcome_labels(match)
            if labels is None:
                continue

            row: dict[str, Any] = {
                "id_fixture": match.get("id_fixture"),
                "season": match.get("season"),
                "prediction_at": match.get("date_match"),
                "y": labels[outcome],
            }
            row.update(features)
            rows.append(row)

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame

        frame["prediction_at"] = pd.to_datetime(frame["prediction_at"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)

        # Garanzia strutturale (acceptance criteria EXP-03): separazione netta da market expert.
        assert not any("odd" in col.lower() for col in frame.columns), (
            "Statistics Expert non deve contenere feature di mercato/odds"
        )
        return frame

    def build_from_db(self, outcome: str = "home_win", seasons: Optional[list[int]] = None) -> pd.DataFrame:
        filters: dict[str, Any] = {
            "mean_statistics": "not None",
            "statistics": "not None",
            "status": list(FINAL_STATUSES),
        }
        if seasons:
            filters["season"] = seasons

        orm_matches = self.match_repo.search_filter(filters=filters)
        matches = convert_orm_match_to_dict(orm_matches)
        return self.build_dataset_from_records(matches=matches, outcome=outcome)

    # ------------------------------------------------------------------
    # Fit + validazione temporale (metriche temporali disponibili)
    # ------------------------------------------------------------------
    def fit_and_evaluate(
        self,
        frame: pd.DataFrame,
        n_splits: int = 5,
        min_train_size: int = 60,
        min_valid_size: int = 15,
    ) -> dict[str, Any]:
        if frame.empty:
            raise ValueError("Dataset vuoto: impossibile addestrare Statistics Expert")

        feature_cols = [c for c in frame.columns if c not in {"id_fixture", "season", "prediction_at", "y"}]
        X = frame[feature_cols]
        y = frame["y"].astype(int)

        splits = expanding_window_splits(
            frame=frame,
            time_col="prediction_at",
            n_splits=n_splits,
            min_train_size=min_train_size,
            min_valid_size=min_valid_size,
        )
        if not splits:
            raise ValueError("Dataset insufficiente per validazione temporale")

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )

        oof = temporal_oof_probabilities(estimator=pipeline, X=X, y=y, cv_splits=splits)
        if oof.empty:
            raise ValueError("Nessun fold valido per Statistics Expert")

        probability_metrics = compute_probability_metrics(
            y_true=oof["y_true"].astype(int).to_numpy(),
            probabilities=oof["probability"].astype(float).to_numpy(),
            n_bins=10,
        )

        # Fit finale su tutto lo storico per l'uso in produzione (embedding_features).
        pipeline.fit(X, y)
        self._pipeline = pipeline
        self._feature_names = feature_cols

        return {
            "cv_strategy": "expanding_window",
            "folds": len(splits),
            "probability_metrics": probability_metrics,
        }

    # ------------------------------------------------------------------
    # Embedding riusabile da altri esperti/modelli
    # ------------------------------------------------------------------
    def embedding_features(self, frame: pd.DataFrame) -> pd.Series:
        """Output numerico compatto: probabilita' calibrata dal modello pre-match."""
        if self._pipeline is None:
            raise RuntimeError("Modello non addestrato: chiamare fit_and_evaluate prima")

        X = frame[self._feature_names]
        proba = self._pipeline.predict_proba(X)
        p1 = proba[:, -1] if proba.ndim == 2 else proba
        return pd.Series(p1, index=frame.index, name="statistics_expert_embedding")
