"""Modello "il giocatore prende un giallo?": stessa struttura di `service_ia/scorer/model_store.py`
(famiglie in gara su holdout temporale, registrazione come CANDIDATE, promozione solo con controlli propri +
policy standard), con registro SEPARATO (`best_models/registry_player_cards`).

Differenze rispetto ai marcatori:
- si addestra e si valuta SOLO sui titolari: sono gli unici su cui si predice (la formazione ufficiale), e
  includere i subentrati sporcherebbe calibrazione e baseline;
- iperparametri scelti sul dataset v1 (eventi + formazioni + arbitro, senza statistiche individuali);
- il segnale e' piu' debole (AUC ~0,58 sui titolari): il modello vale per la calibrazione e per ruolo/arbitro/
  forma, non per "indovinare" l'ammonito. Le soglie di promozione restano quelle dei marcatori."""

from __future__ import annotations

import dataclasses
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

from src.data.player.cards_dataset import FEATURE_COLUMNS, TARGET
from src.ml.evaluation.classification_report import compute_compact_classification_summary
from src.ml.evaluation.probability_metrics import compute_probability_metrics
from src.ml.registry.promotion_policy import DEFAULT_PROMOTION_POLICY
from src.service_ia.scorer.model_store import (
    EURO_CUPS, HOLDOUT_FRACTION, MAX_ECE, MAX_ROLE_GAP, MIN_GAIN_OVER_BASELINE, MIN_GROUP_ROWS,
    MIN_TRAIN_ROWS, REFRESH_TOLERANCE, SEED, league_group,
)
from src.service_ia.training.model_registry import ModelRegistry
from src.storage import bucket_store

logger = logging.getLogger(__name__)

MARKET = "player_yellow_card"
REGISTRY_DIR = "best_models/registry_player_cards"
BUNDLE_PREFIX = "best_models/player_yellow_card"
FAMILIES = ("logreg", "hgb", "rf", "voting")
__all__ = ["EURO_CUPS", "MARKET", "REGISTRY_DIR", "FAMILIES"]


def _logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))


def _hgb():
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=3, max_iter=150, min_samples_leaf=100, l2_regularization=1,
        early_stopping=False, random_state=SEED,
    )


def _rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=100, max_features=0.5, random_state=SEED, n_jobs=2,
    )


def _estimator(family: str = "voting"):
    if family == "logreg":
        return _logreg()
    if family == "hgb":
        return _hgb()
    if family == "rf":
        return _rf()
    return VotingClassifier([("logreg", _logreg()), ("hgb", _hgb()), ("rf", _rf())], voting="soft")


def starters_only(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset[dataset["is_starter"] == 1]


def _matrix(df: pd.DataFrame, medians: pd.Series) -> np.ndarray:
    return df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(medians).to_numpy(dtype=float)


def train_bundle(dataset: pd.DataFrame, family: str = "voting") -> dict[str, Any]:
    """Addestra su TUTTI i titolari del dataset passato (partite concluse)."""
    dataset = starters_only(dataset).sort_values("match_date")
    if len(dataset) < MIN_TRAIN_ROWS:
        raise ValueError(f"Dataset troppo piccolo per addestrare: {len(dataset)} righe (< {MIN_TRAIN_ROWS})")
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


def _group_metrics(test: pd.DataFrame, p: np.ndarray, base: float) -> dict[str, Any]:
    groups = test["league"].map(league_group).to_numpy()
    y_all = test[TARGET].to_numpy()
    out: dict[str, Any] = {}
    for name in ("top5", "cup", "other"):
        mask = groups == name
        if mask.sum() < MIN_GROUP_ROWS or y_all[mask].min() == y_all[mask].max():
            continue
        y = y_all[mask]
        m = compute_probability_metrics(y, p[mask])
        out[name] = {
            "n": int(mask.sum()), "log_loss": m["log_loss"], "ece": m["ece"], "auc": m["auc"],
            "baseline_log_loss": float(-np.mean(y * np.log(base) + (1 - y) * np.log(1 - base))),
        }
    return out


def holdout_metrics(dataset: pd.DataFrame, families: tuple[str, ...] = FAMILIES) -> dict[str, Any]:
    """Addestra ogni famiglia sul passato e la misura sulle partite piu' recenti (ultimo 20% delle DATE),
    sui soli titolari. Ritorna le metriche della famiglia VINCENTE (log loss piu' bassa) e il confronto."""
    dataset = starters_only(dataset).sort_values("match_date").reset_index(drop=True)
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
    metrics["best_family"] = best
    metrics["baseline_log_loss"] = baseline
    metrics["holdout_from"] = str(cut)
    metrics["families"] = {f: {k: results[f][k] for k in ("log_loss", "brier", "ece", "auc")} for f in results}
    p = probs[best]
    metrics["classification"] = compute_compact_classification_summary(test[TARGET].to_numpy(), p)
    metrics["groups"] = _group_metrics(test, p, base)
    metrics["by_role"] = {
        role: {"n": int(m.sum()), "observed": float(test[TARGET][m].mean()), "predicted": float(p[m.to_numpy()].mean())}
        for role in ("F", "M", "D") if (m := test[f"pos_{role}"] == 1).any()
    }
    return metrics


def train_candidate(dataset: pd.DataFrame, *, reg: Optional[ModelRegistry] = None) -> dict[str, Any]:
    """Sceglie la famiglia migliore sull'holdout, riaddestra quella su tutto, salva il bundle e registra un
    run `candidate`. La promozione e' un passo separato (`auto_promote`)."""
    metrics = holdout_metrics(dataset)
    family = metrics["best_family"]
    bundle = train_bundle(dataset, family)
    key = f"{BUNDLE_PREFIX}/v{bundle['version']}.joblib"
    bucket_store.put_joblib(key, bundle)
    metrics["rows"] = bundle["n_rows"]
    return (reg or registry()).register(
        model_path=key, market=MARKET, model_name=family, metrics=metrics,
        feature_names=list(FEATURE_COLUMNS),
        params={"family": family, "target": "yellow_card_regulation", "population": "starters"},
        extra={"sklearn_version": bundle["sklearn_version"], "n_fixtures": bundle["n_fixtures"],
               "train_until": bundle["train_until"], "dataset_version": f"player_cards:{bundle['train_until']}"},
        stage="candidate",
    )


def auto_promote(run: dict[str, Any], *, reg: Optional[ModelRegistry] = None) -> dict[str, Any]:
    """Promuove il candidate a production SOLO se supera i controlli qui sotto E la policy standard. Mai
    `force`; ogni tentativo e' nell'audit con actor 'auto:player_cards_training'."""
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
    for name, g in (m.get("groups") or {}).items():
        if g["baseline_log_loss"] - g["log_loss"] < MIN_GAIN_OVER_BASELINE:
            reasons.append(f"gruppo {name}: log_loss non abbastanza sotto la baseline")
        if g["ece"] > MAX_ECE:
            reasons.append(f"gruppo {name}: calibrazione (ECE) sopra soglia")
    if reasons:
        return {"promoted": False, "reasons": reasons}
    policy = dataclasses.replace(DEFAULT_PROMOTION_POLICY, min_improvement_over_production=-REFRESH_TOLERANCE)
    result = reg.promote_with_policy(run["run_id"], to_stage="production", policy=policy,
                                     actor="auto:player_cards_training",
                                     reason="auto-promozione: controlli gialli + policy superati")
    out = {"promoted": bool(result["promoted"]), "reasons": []}
    if not result["promoted"]:
        out["reasons"] = ["policy di promozione: non supera il confronto con production o il gate metriche"]
    return out


def load_production_bundle(*, reg: Optional[ModelRegistry] = None) -> Optional[dict[str, Any]]:
    """Bundle del run in stage `production`, oppure None (nessuna promozione / versione incompatibile)."""
    try:
        run = (reg or registry()).get_production(market=MARKET)
        if run is None:
            return None
        bundle = bucket_store.get_joblib(run["model_path"])
    except Exception:  # noqa: BLE001 - bucket non raggiungibile = nessun modello, non un crash del job
        logger.exception("Gialli: impossibile leggere il modello production")
        return None
    if bundle.get("sklearn_version") != sklearn.__version__ or bundle.get("feature_columns") != list(FEATURE_COLUMNS):
        logger.warning("Gialli: modello production non compatibile (sklearn o feature cambiate): riaddestrare")
        return None
    bundle["run_id"] = run["run_id"]
    return bundle
