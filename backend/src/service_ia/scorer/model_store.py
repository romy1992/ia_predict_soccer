"""Modello "il giocatore segna?": addestramento con iperparametri fissi (scelti dalla ricerca fatta in
cloud, vedi `scripts/analysis/train_scorer_from_export.py`) e salvataggio su bucket S3.

Il bundle contiene anche le mediane del train (per i valori mancanti), le colonne usate e la versione
di scikit-learn: se la versione cambia il bundle viene riaddestrato, mai caricato alla cieca."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.player.scoring_dataset import FEATURE_COLUMNS, TARGET
from src.storage import bucket_store

logger = logging.getLogger(__name__)

BUNDLE_KEY = "models/scorer/anytime/current.joblib"
SEED = 42
MIN_TRAIN_ROWS = 20_000


def _estimator() -> VotingClassifier:
    """Voting soft tra logistica, HistGradientBoosting e random forest (campione della ricerca in cloud)."""
    logreg = make_pipeline(StandardScaler(), LogisticRegression(C=0.0113, max_iter=3000))
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.08, max_depth=2, max_iter=100, min_samples_leaf=50, l2_regularization=1,
        early_stopping=False, random_state=SEED,
    )
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=50, max_features=0.5, random_state=SEED, n_jobs=2,
    )
    return VotingClassifier([("logreg", logreg), ("hgb", hgb), ("rf", rf)], voting="soft")


def _matrix(df: pd.DataFrame, medians: pd.Series) -> np.ndarray:
    return df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(medians).to_numpy(dtype=float)


def train_bundle(dataset: pd.DataFrame) -> dict[str, Any]:
    """Addestra su TUTTO il dataset passato (partite concluse, una riga per giocatore-partita)."""
    if len(dataset) < MIN_TRAIN_ROWS:
        raise ValueError(f"Dataset troppo piccolo per addestrare: {len(dataset)} righe (< {MIN_TRAIN_ROWS})")
    dataset = dataset.sort_values("match_date")
    medians = dataset[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).median()
    model = _estimator().fit(_matrix(dataset, medians), dataset[TARGET].to_numpy())
    return {
        "model": model,
        "medians": medians,
        "feature_columns": list(FEATURE_COLUMNS),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(dataset)),
        "n_fixtures": int(dataset["fixture_id"].nunique()),
        "train_until": str(dataset["match_date"].max()),
        "positive_rate": float(dataset[TARGET].mean()),
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M"),
    }


def predict_proba(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    if bundle["feature_columns"] != list(FEATURE_COLUMNS):
        raise ValueError("Le feature del bundle non coincidono con quelle del codice: riaddestrare.")
    return bundle["model"].predict_proba(_matrix(frame, bundle["medians"]))[:, 1]


def save_bundle(bundle: dict[str, Any]) -> None:
    bucket_store.put_joblib(BUNDLE_KEY, bundle)
    bucket_store.put_joblib(f"models/scorer/anytime/v{bundle['version']}.joblib", bundle)


def load_bundle() -> Optional[dict[str, Any]]:
    """Bundle corrente dal bucket, oppure None se manca / e' stato addestrato con un'altra versione di sklearn."""
    try:
        if not bucket_store.exists(BUNDLE_KEY):
            return None
        bundle = bucket_store.get_joblib(BUNDLE_KEY)
    except Exception:  # noqa: BLE001 - bucket non raggiungibile = nessun modello, non un crash del job
        logger.exception("Scorer: impossibile leggere il modello dal bucket")
        return None
    if bundle.get("sklearn_version") != sklearn.__version__ or bundle.get("feature_columns") != list(FEATURE_COLUMNS):
        logger.warning("Scorer: modello salvato non compatibile (sklearn o feature cambiate), serve riaddestrare")
        return None
    return bundle
