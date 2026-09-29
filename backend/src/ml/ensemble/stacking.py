"""Meta Model / Stacker per mercato (ORACLE-02, Fase ENSEMBLE).

Combina le probabilita' gia' prodotte da piu' Oracle Expert per lo STESSO
mercato (le "meta feature", una colonna per esperto — tipicamente ottenute
da `combine_expert_outputs`/`as_feature_row()` dello schema ORACLE-01,
`src/ml/ensemble/expert_output.py`) confrontando due strategie:

- ``weighted_blend``: combinazione lineare con pesi non negativi che sommano
  a 1 (`WeightedBlendClassifier`), ottimizzati per minimizzare la log loss.
  Nessun modello "black box": e' l'equivalente "imparato" di una media
  pesata (vedi `ensemble_probability` in `btts_market.py`, li' a peso
  fisso 0.5).
- ``learned_stacker``: un classificatore (default `LogisticRegression`, per
  coerenza con lo stile gia' usato per la calibrazione Platt altrove nel
  progetto) che impara a combinare le stesse meta-feature.

Principio "niente leakage stacking" (acceptance criteria): ENTRAMBI gli
approcci sono validati con lo STESSO meccanismo di OOF walk-forward gia'
usato nel resto del progetto (`temporal_oof_probabilities`, ML-05): per
ogni fold temporale (`expanding_window_splits`, ML-05) il modello di
secondo livello viene rifittato SOLO sul training set del fold e valutato
sul validation set — mai il contrario. Le colonne di input (le probabilita'
dei singoli esperti) devono essere gia' "pulite" rispetto al leakage: se un
esperto e' esso stesso un modello allenato, le sue probabilita' devono
provenire da un proprio processo out-of-fold/produzione a monte
(responsabilita' del chiamante — stesso principio gia' adottato da
`BttsBenchmarkReport`/`TotalsBenchmarkReport`, che assumono probabilita' di
input gia' calcolate).

Questo modulo NON re-implementa il training degli esperti esistenti: si
limita a orchestrarne la combinazione, coerentemente con la dipendenza
dichiarata ORACLE-01.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss

from src.ml.ensemble.expert_output import ExpertOutput, combine_expert_outputs
from src.ml.evaluation.probability_metrics import (
    champion_probability_score,
    compute_probability_metrics,
    temporal_oof_probabilities,
)
from src.ml.validation.temporal_split import expanding_window_splits
from src.service_ia.training.model_registry import ModelRegistry

WEIGHTED_BLEND = "weighted_blend"
LEARNED_STACKER = "learned_stacker"
STACKING_APPROACHES: tuple[str, str] = (WEIGHTED_BLEND, LEARNED_STACKER)


def _softmax(raw: np.ndarray) -> np.ndarray:
    """Softmax numericamente stabile: garantisce pesi >=0 che sommano a 1
    per COSTRUZIONE, qualunque sia `raw` (nessun vincolo esplicito da
    imporre all'ottimizzatore)."""
    shifted = raw - np.max(raw)
    exp = np.exp(shifted)
    total = exp.sum()
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(raw, 1.0 / max(1, raw.size))
    return exp / total


class WeightedBlendClassifier(BaseEstimator, ClassifierMixin):
    """Combinazione lineare (pesi >=0, somma=1) di probabilita' esperto gia'
    calcolate. I pesi sono parametrizzati via softmax (sempre un simplex
    valido) e ottimizzati per minimizzare la log loss sul training set
    fornito a `fit` — l'equivalente "imparato" di una media pesata fissa.

    Compatibile con `sklearn.base.clone` (richiesto da
    `temporal_oof_probabilities` per il walk-forward): `__init__` si limita
    a salvare i parametri cosi' come ricevuti, nessuno stato mutabile.
    """

    def __init__(self, l2: float = 0.0):
        self.l2 = l2

    def fit(self, X, y) -> "WeightedBlendClassifier":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int).reshape(-1)
        self.classes_ = np.array([0, 1])
        n_features = X_arr.shape[1] if X_arr.ndim == 2 else 0

        if n_features == 0:
            self.weights_ = np.zeros(0)
            self._fallback_probability_ = float(np.mean(y_arr)) if y_arr.size else 0.5
            return self

        if np.unique(y_arr).size < 2:
            # Log loss non informativa con una sola classe nel fold di
            # training: pesi uniformi come fallback sicuro (nessuna
            # ottimizzazione ha significato statistico in questo caso).
            self.weights_ = _softmax(np.zeros(n_features))
            return self

        def objective(raw_weights: np.ndarray) -> float:
            weights = _softmax(raw_weights)
            p1 = np.clip(X_arr @ weights, 1e-9, 1.0 - 1e-9)
            loss = log_loss(y_arr, p1, labels=[0, 1])
            return float(loss + self.l2 * np.sum(raw_weights ** 2))

        x0 = np.zeros(n_features)
        result = minimize(objective, x0, method="BFGS")
        self.weights_ = _softmax(result.x) if result.success else _softmax(x0)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        weights = getattr(self, "weights_", None)
        if weights is None or weights.size == 0:
            p1 = np.full(X_arr.shape[0], getattr(self, "_fallback_probability_", 0.5))
        else:
            p1 = np.clip(X_arr @ weights, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def default_stacker_estimator() -> LogisticRegression:
    """Learned stacker di default: interpretabile e coerente con lo stile
    gia' usato per la calibrazione Platt altrove nel progetto."""
    return LogisticRegression(max_iter=1000)


def build_meta_features_from_expert_outputs(
    rows_of_expert_outputs: Sequence[Sequence[ExpertOutput]],
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Costruisce la matrice di meta-feature (una riga per fixture, una
    colonna per 'expert_name__outcome'/'expert_name__confidence') a partire
    da liste di `ExpertOutput` (schema ORACLE-01) — una lista per fixture,
    riusando `combine_expert_outputs` senza duplicarne la logica (acceptance
    criteria "Meta features dagli expert").

    Se un esperto non produce output per alcune righe (es. fallisce su
    alcune fixture), le colonne mancanti sono riempite con `fill_value`
    (default 0.0: "nessun segnale", scelta esplicita — NON una stima
    neutra di probabilita' 0.5) cosi' che il risultato sia sempre
    direttamente utilizzabile da uno stimatore sklearn (che non accetta NaN).
    """
    combined_rows = [combine_expert_outputs(outputs) for outputs in rows_of_expert_outputs]
    frame = pd.DataFrame(combined_rows)
    if frame.empty:
        return frame
    frame = frame.reindex(sorted(frame.columns), axis=1)
    return frame.fillna(fill_value)


def generate_stacking_oof(
    estimator: Any,
    meta_features: pd.DataFrame,
    y_true: Sequence[int],
    cv_splits: list[tuple[list[int], list[int]]],
) -> pd.DataFrame:
    """OOF walk-forward (`temporal_oof_probabilities`, ML-05) per uno
    stimatore di secondo livello (blend o stacker): unico punto in cui
    avviene fit/predict, cosi' i due approcci sono confrontati con
    ESATTAMENTE lo stesso meccanismo anti-leakage."""
    if meta_features.empty:
        raise ValueError("meta_features vuoto: impossibile generare OOF di stacking")
    if not cv_splits:
        raise ValueError("cv_splits vuoto: impossibile generare OOF di stacking senza fold temporali")

    X = meta_features.reset_index(drop=True)
    y = pd.Series(np.asarray(y_true, dtype=int)).reset_index(drop=True)
    oof = temporal_oof_probabilities(estimator=estimator, X=X, y=y, cv_splits=cv_splits)
    if oof.empty:
        raise ValueError("Nessun fold valido per generare OOF di stacking (verifica cv_splits/y)")
    return oof


@dataclass
class StackingBenchmarkReport:
    """Report benchmark (acceptance criteria "Confronto weighted blend vs
    learned stacker"): metriche OOF per ciascun approccio di secondo
    livello, diagnostica sui singoli esperti in input, il migliore
    selezionato e il meta-model finale (fittato sull'intero storico,
    pronto per la registrazione come 'candidate')."""

    expert_names: list[str]
    single_expert_metrics: dict[str, dict[str, Any]]
    approach_metrics: dict[str, dict[str, Any]]
    best_approach: str
    meta_model: Any


def benchmark_stacking_approaches(
    meta_features: pd.DataFrame,
    y_true: Sequence[int],
    cv_splits: list[tuple[list[int], list[int]]],
    stacker_estimator: Optional[Any] = None,
) -> StackingBenchmarkReport:
    """Confronta weighted_blend vs learned_stacker sulle STESSE meta-feature
    (OOF walk-forward per entrambi) e sceglie il migliore per
    `champion_probability_score` (ML-05, stesso criterio di tutti gli altri
    mercati). Include anche le metriche "as-is" di ciascun singolo esperto
    in input come riferimento diagnostico: NON competono per
    `best_approach` (l'obiettivo del task e' il meta-model, non ri-scegliere
    il singolo miglior esperto)."""
    if meta_features.empty:
        raise ValueError("meta_features vuoto: impossibile eseguire il benchmark di stacking")

    expert_names = list(meta_features.columns)
    y = np.asarray(y_true, dtype=int)

    single_expert_metrics: dict[str, dict[str, Any]] = {}
    for column in expert_names:
        values = meta_features[column].to_numpy(dtype=float)
        metrics = compute_probability_metrics(y_true=y, probabilities=values, n_bins=10)
        predicted = (values >= 0.5).astype(int)
        f1_weighted = float(f1_score(y, predicted, average="weighted", zero_division=0))
        single_expert_metrics[column] = {**metrics, "f1_weighted": f1_weighted}

    resolved_stacker = stacker_estimator or default_stacker_estimator()
    estimators: dict[str, Any] = {
        WEIGHTED_BLEND: WeightedBlendClassifier(),
        LEARNED_STACKER: resolved_stacker,
    }

    approach_metrics: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    fitted_meta_models: dict[str, Any] = {}
    for name, estimator in estimators.items():
        oof = generate_stacking_oof(estimator=estimator, meta_features=meta_features, y_true=y, cv_splits=cv_splits)
        metrics = compute_probability_metrics(
            y_true=oof["y_true"].to_numpy(), probabilities=oof["probability"].to_numpy(), n_bins=10
        )
        predicted = (oof["probability"].to_numpy() >= 0.5).astype(int)
        f1_weighted = float(f1_score(oof["y_true"].to_numpy(), predicted, average="weighted", zero_division=0))
        selection_score = champion_probability_score(metrics=metrics, f1_weighted=f1_weighted)
        approach_metrics[name] = {
            **metrics,
            "f1_weighted": f1_weighted,
            "selection_score": selection_score,
            "oof_rows": int(len(oof)),
        }
        scores[name] = selection_score

        # Meta-model "di produzione": rifittato sull'INTERO storico (stesso
        # principio del calibratore finale in btts_market.py/totals_market.py).
        final_model = clone(estimator)
        final_model.fit(meta_features.reset_index(drop=True), pd.Series(y).reset_index(drop=True))
        fitted_meta_models[name] = final_model

    best_approach = max(scores.items(), key=lambda kv: kv[1])[0]

    return StackingBenchmarkReport(
        expert_names=expert_names,
        single_expert_metrics=single_expert_metrics,
        approach_metrics=approach_metrics,
        best_approach=best_approach,
        meta_model=fitted_meta_models[best_approach],
    )


@dataclass
class StackingRunResult:
    """Esito dell'orchestrazione end-to-end (meta-feature reali -> report
    benchmark -> eventuale registrazione), stesso stile di
    `BttsBenchmarkRunResult`/`TotalsBenchmarkRunResult`."""

    market: str
    rows: int
    status: str
    best_approach: Optional[str]
    details: dict[str, Any]


def run_stacking_benchmark(
    market: str,
    meta_features: pd.DataFrame,
    y_true: Sequence[int],
    prediction_at: Sequence[Any],
    stacker_estimator: Optional[Any] = None,
    n_splits: int = 5,
    save_model: bool = True,
) -> StackingRunResult:
    """Costruisce i fold walk-forward (`expanding_window_splits`, stesso
    meccanismo di tutti gli altri mercati) su `prediction_at`, esegue
    `benchmark_stacking_approaches` e, se richiesto, registra il meta-model
    vincente come 'candidate' (MAI 'production' automaticamente: vincolo
    generale del progetto, promozione manuale via OPS-02)."""
    if meta_features.empty:
        return StackingRunResult(market=market, rows=0, status="skipped_no_data", best_approach=None, details={})

    frame = pd.DataFrame({"prediction_at": pd.to_datetime(list(prediction_at), utc=True, errors="coerce")})
    if frame["prediction_at"].isna().any():
        raise ValueError("prediction_at contiene valori data non validi")

    min_train = max(30, int(len(meta_features) * 0.45))
    min_valid = max(10, int(len(meta_features) * 0.1))
    cv_splits = expanding_window_splits(
        frame=frame, time_col="prediction_at", n_splits=n_splits, min_train_size=min_train, min_valid_size=min_valid
    )
    if not cv_splits:
        return StackingRunResult(
            market=market,
            rows=len(meta_features),
            status="skipped_insufficient_rows_for_temporal_cv",
            best_approach=None,
            details={},
        )

    report = benchmark_stacking_approaches(
        meta_features=meta_features, y_true=y_true, cv_splits=cv_splits, stacker_estimator=stacker_estimator
    )

    run_metadata = None
    if save_model:
        model_path = os.path.abspath(os.path.join("best_models", f"{market}_meta_{report.best_approach}.pkl"))
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(report.meta_model, model_path)

        registry = ModelRegistry()
        run_metadata = registry.register(
            model_path=model_path,
            market=market,
            model_name=f"meta_{report.best_approach}",
            feature_names=report.expert_names,
            metrics={
                "selection_score": report.approach_metrics[report.best_approach]["selection_score"],
                "log_loss": report.approach_metrics[report.best_approach]["log_loss"],
                "brier": report.approach_metrics[report.best_approach]["brier"],
                "ece": report.approach_metrics[report.best_approach]["ece"],
            },
            extra={
                "stacking_approach": report.best_approach,
                "approach_metrics": report.approach_metrics,
                "single_expert_metrics": report.single_expert_metrics,
                "rows": len(meta_features),
            },
            stage="candidate",
        )

    return StackingRunResult(
        market=market,
        rows=len(meta_features),
        status="benchmarked",
        best_approach=report.best_approach,
        details={
            "approach_metrics": report.approach_metrics,
            "single_expert_metrics": report.single_expert_metrics,
            "expert_names": report.expert_names,
            "run": run_metadata,
        },
    )
