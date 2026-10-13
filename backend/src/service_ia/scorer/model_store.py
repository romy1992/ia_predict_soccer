"""Modello "il giocatore segna?": addestramento con iperparametri fissi (scelti dalla ricerca fatta in
cloud, vedi `scripts/analysis/train_scorer_from_export.py`), valutazione su holdout temporale e
registrazione come CANDIDATE nel registro modelli (stesso `ModelRegistry`, stessa policy di promozione
degli altri mercati: nessun training diventa production da solo).

Il registro e' SEPARATO (`best_models/registry_scorer`) da quello dei mercati di partita: il mercato dei
marcatori ha un'unita' diversa (giocatore, non partita) e nel registro principale finirebbe nei giri di
Dashboard/snapshot/betslip che predicono per partita.

Il bundle contiene anche le mediane del train (per i valori mancanti), le colonne usate e la versione
di scikit-learn: se la versione cambia il bundle non viene caricato alla cieca."""

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
from src.ml.evaluation.probability_metrics import compute_probability_metrics
from src.service_ia.training.model_registry import ModelRegistry
from src.storage import bucket_store

logger = logging.getLogger(__name__)

MARKET = "player_anytime_goalscorer"
MODEL_NAME = "voting_logreg_hgb_rf"
REGISTRY_DIR = "best_models/registry_scorer"
BUNDLE_PREFIX = "best_models/player_anytime_scorer"
HOLDOUT_FRACTION = 0.2
SEED = 42
MIN_TRAIN_ROWS = 20_000


def _logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.0113, max_iter=3000))


def _hgb():
    return HistGradientBoostingClassifier(
        learning_rate=0.08, max_depth=2, max_iter=100, min_samples_leaf=50, l2_regularization=1,
        early_stopping=False, random_state=SEED,
    )


def _rf():
    return RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=50, max_features=0.5, random_state=SEED, n_jobs=2,
    )


# Famiglie in gara a ogni training (iperparametri fissi scelti dalla ricerca in cloud). Vince la log loss
# piu' bassa sull'holdout temporale; il voting soft combina le tre.
FAMILIES = ("logreg", "hgb", "rf", "voting")


def _estimator(family: str = "voting"):
    if family == "logreg":
        return _logreg()
    if family == "hgb":
        return _hgb()
    if family == "rf":
        return _rf()
    return VotingClassifier([("logreg", _logreg()), ("hgb", _hgb()), ("rf", _rf())], voting="soft")


def _matrix(df: pd.DataFrame, medians: pd.Series) -> np.ndarray:
    return df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(medians).to_numpy(dtype=float)


def train_bundle(dataset: pd.DataFrame, family: str = "voting") -> dict[str, Any]:
    """Addestra su TUTTO il dataset passato (partite concluse, una riga per giocatore-partita)."""
    if len(dataset) < MIN_TRAIN_ROWS:
        raise ValueError(f"Dataset troppo piccolo per addestrare: {len(dataset)} righe (< {MIN_TRAIN_ROWS})")
    dataset = dataset.sort_values("match_date")
    medians = dataset[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).median()
    model = _estimator(family).fit(_matrix(dataset, medians), dataset[TARGET].to_numpy())
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
        "family": family,
    }


def predict_proba(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    if bundle["feature_columns"] != list(FEATURE_COLUMNS):
        raise ValueError("Le feature del bundle non coincidono con quelle del codice: riaddestrare.")
    return bundle["model"].predict_proba(_matrix(frame, bundle["medians"]))[:, 1]


def registry() -> ModelRegistry:
    return ModelRegistry(registry_dir=REGISTRY_DIR)


def holdout_metrics(dataset: pd.DataFrame, families: tuple[str, ...] = FAMILIES) -> dict[str, Any]:
    """Addestra ogni famiglia sul passato e la misura sulle partite piu' recenti (ultimo 20% delle DATE,
    mai una partita spezzata tra train e test). Ritorna le metriche della famiglia VINCENTE (log loss piu'
    bassa) piu' il confronto tra tutte in `families`."""
    dataset = dataset.sort_values("match_date").reset_index(drop=True)
    cut = dataset["match_date"].quantile(1.0 - HOLDOUT_FRACTION)
    train, test = dataset[dataset["match_date"] < cut], dataset[dataset["match_date"] >= cut]
    if len(train) < MIN_TRAIN_ROWS or test.empty or test[TARGET].nunique() < 2:
        raise ValueError("Dati insufficienti per la valutazione su holdout temporale")
    medians = train[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).median()
    x_train, y_train, x_test = _matrix(train, medians), train[TARGET].to_numpy(), _matrix(test, medians)
    base = float(train[TARGET].mean())
    baseline = float(-np.mean(test[TARGET] * np.log(base) + (1 - test[TARGET]) * np.log(1 - base)))

    results: dict[str, dict[str, Any]] = {}
    probs: dict[str, np.ndarray] = {}
    for family in families:
        p = _estimator(family).fit(x_train, y_train).predict_proba(x_test)[:, 1]
        probs[family] = p
        results[family] = compute_probability_metrics(test[TARGET].to_numpy(), p)
    best = min(results, key=lambda f: results[f]["log_loss"])
    metrics = dict(results[best])
    metrics.pop("reliability", None)
    metrics["reliability"] = results[best].get("reliability")
    metrics["best_family"] = best
    metrics["baseline_log_loss"] = baseline
    metrics["holdout_from"] = str(cut)
    metrics["families"] = {f: {k: results[f][k] for k in ("log_loss", "brier", "ece", "auc")} for f in results}
    p = probs[best]
    metrics["by_role"] = {
        role: {"n": int(m.sum()), "observed": float(test[TARGET][m].mean()), "predicted": float(p[m.to_numpy()].mean())}
        for role in ("F", "M", "D") if (m := test[f"pos_{role}"] == 1).any()
    }
    return metrics


def train_candidate(dataset: pd.DataFrame, *, reg: Optional[ModelRegistry] = None) -> dict[str, Any]:
    """Sceglie la famiglia migliore sull'holdout, riaddestra quella su tutto, salva il bundle e registra un
    run `candidate`. La promozione e' un passo separato (`auto_promote` o `promote_with_policy`)."""
    metrics = holdout_metrics(dataset)
    family = metrics["best_family"]
    bundle = train_bundle(dataset, family)
    key = f"{BUNDLE_PREFIX}/v{bundle['version']}.joblib"
    bucket_store.put_joblib(key, bundle)
    metrics["rows"] = bundle["n_rows"]
    return (reg or registry()).register(
        model_path=key, market=MARKET, model_name=family, metrics=metrics,
        feature_names=list(FEATURE_COLUMNS),
        params={"family": family, "target": "anytime_goal_regulation"},
        extra={"sklearn_version": bundle["sklearn_version"], "n_fixtures": bundle["n_fixtures"],
               "train_until": bundle["train_until"], "dataset_version": f"player_scoring:{bundle['train_until']}"},
        stage="candidate",
    )


# Controlli AGGIUNTIVI a quelli della policy di promozione (soglie troppo larghe per un evento raro).
MIN_GAIN_OVER_BASELINE = 0.005   # log loss almeno 0,005 sotto la media storica
MAX_ECE = 0.03
MAX_ROLE_GAP = 0.03              # |previsto - osservato| medio per ruolo


def auto_promote(run: dict[str, Any], *, reg: Optional[ModelRegistry] = None) -> dict[str, Any]:
    """Promuove il candidate a production SOLO se supera i controlli qui sotto E la policy standard
    (gate metriche + confronto con l'attuale production). Mai `force`; ogni tentativo e' nell'audit con
    actor 'auto:scorer_training'. Ritorna {promoted, reasons}."""
    reg = reg or registry()
    m = run.get("metrics") or {}
    reasons = []
    if (m.get("baseline_log_loss") or 0) - (m.get("log_loss") or 99) < MIN_GAIN_OVER_BASELINE:
        reasons.append("log_loss non abbastanza sotto la baseline")
    if (m.get("ece") if m.get("ece") is not None else 99) > MAX_ECE:
        reasons.append("calibrazione (ECE) sopra soglia")
    for role, r in (m.get("by_role") or {}).items():
        if abs(r["predicted"] - r["observed"]) > MAX_ROLE_GAP:
            reasons.append(f"ruolo {role}: probabilita' media lontana dalla realta'")
    if reasons:
        return {"promoted": False, "reasons": reasons}
    result = reg.promote_with_policy(run["run_id"], to_stage="production", actor="auto:scorer_training",
                                     reason="auto-promozione: controlli marcatori + policy superati")
    out = {"promoted": bool(result["promoted"]), "reasons": []}
    if not result["promoted"]:
        out["reasons"] = ["policy di promozione: non supera il confronto con production o il gate metriche"]
    return out


def load_production_bundle(*, reg: Optional[ModelRegistry] = None) -> Optional[dict[str, Any]]:
    """Bundle del run in stage `production`, oppure None (nessuna promozione fatta / versione incompatibile)."""
    try:
        run = (reg or registry()).get_production(market=MARKET)
        if run is None:
            return None
        bundle = bucket_store.get_joblib(run["model_path"])
    except Exception:  # noqa: BLE001 - bucket non raggiungibile = nessun modello, non un crash del job
        logger.exception("Scorer: impossibile leggere il modello production")
        return None
    if bundle.get("sklearn_version") != sklearn.__version__ or bundle.get("feature_columns") != list(FEATURE_COLUMNS):
        logger.warning("Scorer: modello production non compatibile (sklearn o feature cambiate): riaddestrare e ripromuovere")
        return None
    bundle["run_id"] = run["run_id"]
    return bundle
