"""Calibrazione finale dell'Oracle Ensemble (ORACLE-03, Fase ENSEMBLE).

Applica la calibrazione (Platt/isotonic, `CalibrationService` di ML-06, gia'
riusata da corners/cards) al meta-model vincitore prodotto dallo stacking per
mercato (ORACLE-02, `benchmark_stacking_approaches` in
`src/ml/ensemble/stacking.py`), producendo le probabilita' FINALI
dell'Oracle Ensemble per quel mercato/outcome (acceptance criteria "Final
probabilities calibrate e versionate").

Niente leakage aggiuntivo: `CalibrationService.calibrate_estimator` valuta
pre/post metrics sugli STESSI fold walk-forward (`expanding_window_splits`,
ML-05) gia' usati per scegliere fra weighted_blend/learned_stacker — nessun
nuovo meccanismo di split introdotto qui, e `sklearn.base.clone` (usato sia
da `temporal_oof_probabilities` sia da `CalibratedClassifierCV`) garantisce
che il meta-model gia' fittato sull'intero storico venga sempre ri-allenato
da zero per ogni fold di validazione.

Fallback esplicito (acceptance criteria "Fallback quando campione
insufficiente"): se il numero di righe e' sotto `min_calibration_samples`
oppure la calibrazione fallisce per qualunque motivo (es. troppo pochi fold
validi/OOF vuoto per lo specifico meta-model), il meta-model NON calibrato
(quello gia' selezionato da ORACLE-02) viene usato cosi' com'e', con il
motivo del fallback riportato esplicitamente nel report — mai un'eccezione
che blocca l'intera pipeline.

Questo modulo NON modifica `stacking.py` (ORACLE-02, gia' testato/completo):
lo consuma tramite `benchmark_stacking_approaches`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from src.ml.calibration.calibration_service import CalibrationService
from src.ml.ensemble.stacking import StackingBenchmarkReport, benchmark_stacking_approaches
from src.ml.validation.temporal_split import expanding_window_splits
from src.service_ia.training.model_registry import ModelRegistry

# Soglia minima esplicita di righe sotto la quale la calibrazione (che al suo
# interno rifitta un `CalibratedClassifierCV` con cv=3 per ciascun fold) non
# viene nemmeno tentata: con troppo pochi dati il calibratore rischierebbe di
# introdurre rumore invece di correggere la probabilita' (stessa filosofia
# della soglia isotonic/sigmoid gia' in `CalibrationService.select_method`,
# qui applicata all'intero step di calibrazione).
DEFAULT_MIN_CALIBRATION_SAMPLES = 60


@dataclass
class OracleEnsembleCalibrationReport:
    """Report pre/post (acceptance criteria "Report pre/post"): espone se la
    calibrazione e' stata applicata, il metodo scelto, le metriche prima e
    dopo, e il motivo dell'eventuale fallback. `final_model` e' sempre
    utilizzabile (calibrato se `calibration_applied`, altrimenti il
    meta-model raw di ORACLE-02)."""

    market: str
    calibration_applied: bool
    method: Optional[str]
    sample_size: int
    fallback_reason: Optional[str]
    pre_metrics: Optional[dict[str, Any]]
    post_metrics: Optional[dict[str, Any]]
    stacking_report: StackingBenchmarkReport
    final_model: Any


def calibrate_oracle_ensemble(
    meta_features: pd.DataFrame,
    y_true: Sequence[int],
    cv_splits: list[tuple[list[int], list[int]]],
    market: str,
    stacker_estimator: Optional[Any] = None,
    min_calibration_samples: int = DEFAULT_MIN_CALIBRATION_SAMPLES,
) -> OracleEnsembleCalibrationReport:
    """Esegue il benchmark di stacking (ORACLE-02) e applica la calibrazione
    finale (ML-06) al meta-model vincitore. Ritorna sempre un
    `final_model` utilizzabile, calibrato o (in fallback) raw."""
    stacking_report = benchmark_stacking_approaches(
        meta_features=meta_features, y_true=y_true, cv_splits=cv_splits, stacker_estimator=stacker_estimator
    )

    sample_size = int(len(meta_features))

    def _fallback(reason: str) -> OracleEnsembleCalibrationReport:
        return OracleEnsembleCalibrationReport(
            market=market,
            calibration_applied=False,
            method=None,
            sample_size=sample_size,
            fallback_reason=reason,
            pre_metrics=None,
            post_metrics=None,
            stacking_report=stacking_report,
            final_model=stacking_report.meta_model,
        )

    if sample_size < min_calibration_samples:
        return _fallback(f"sample_size={sample_size} < min_calibration_samples={min_calibration_samples}")

    X = meta_features.reset_index(drop=True)
    y = pd.Series(np.asarray(y_true, dtype=int)).reset_index(drop=True)

    try:
        calibration_result = CalibrationService.calibrate_estimator(
            estimator=stacking_report.meta_model, X=X, y=y, cv_splits=cv_splits
        )
    except Exception as exc:  # fallback deliberatamente ampio: mai bloccare la pipeline
        return _fallback(f"calibration_failed: {exc}")

    return OracleEnsembleCalibrationReport(
        market=market,
        calibration_applied=True,
        method=calibration_result.method,
        sample_size=calibration_result.sample_size,
        fallback_reason=None,
        pre_metrics=calibration_result.pre_metrics,
        post_metrics=calibration_result.post_metrics,
        stacking_report=stacking_report,
        final_model=calibration_result.calibrator,
    )


@dataclass
class OracleEnsembleCalibrationRunResult:
    """Esito dell'orchestrazione end-to-end, stesso stile di
    `StackingRunResult`/`BttsBenchmarkRunResult`."""

    market: str
    rows: int
    status: str
    calibration_applied: Optional[bool]
    best_approach: Optional[str]
    details: dict[str, Any]


def run_oracle_ensemble_calibration(
    market: str,
    meta_features: pd.DataFrame,
    y_true: Sequence[int],
    prediction_at: Sequence[Any],
    stacker_estimator: Optional[Any] = None,
    min_calibration_samples: int = DEFAULT_MIN_CALIBRATION_SAMPLES,
    n_splits: int = 5,
    save_model: bool = True,
) -> OracleEnsembleCalibrationRunResult:
    """Costruisce i fold walk-forward (`expanding_window_splits`, stesso
    meccanismo di ORACLE-02 e di tutti gli altri mercati) su `prediction_at`,
    esegue `calibrate_oracle_ensemble` e, se richiesto, registra il modello
    finale (calibrato, o raw in fallback) come 'candidate' (MAI 'production'
    automaticamente: vincolo generale del progetto, promozione manuale)."""
    if meta_features.empty:
        return OracleEnsembleCalibrationRunResult(
            market=market, rows=0, status="skipped_no_data", calibration_applied=None,
            best_approach=None, details={},
        )

    frame = pd.DataFrame({"prediction_at": pd.to_datetime(list(prediction_at), utc=True, errors="coerce")})
    if frame["prediction_at"].isna().any():
        raise ValueError("prediction_at contiene valori data non validi")

    min_train = max(30, int(len(meta_features) * 0.45))
    min_valid = max(10, int(len(meta_features) * 0.1))
    cv_splits = expanding_window_splits(
        frame=frame, time_col="prediction_at", n_splits=n_splits, min_train_size=min_train, min_valid_size=min_valid
    )
    if not cv_splits:
        return OracleEnsembleCalibrationRunResult(
            market=market,
            rows=len(meta_features),
            status="skipped_insufficient_rows_for_temporal_cv",
            calibration_applied=None,
            best_approach=None,
            details={},
        )

    report = calibrate_oracle_ensemble(
        meta_features=meta_features,
        y_true=y_true,
        cv_splits=cv_splits,
        market=market,
        stacker_estimator=stacker_estimator,
        min_calibration_samples=min_calibration_samples,
    )

    run_metadata = None
    if save_model:
        suffix = "calibrated" if report.calibration_applied else "raw_fallback"
        model_path = os.path.abspath(os.path.join("best_models", f"{market}_oracle_ensemble_{suffix}.pkl"))
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(report.final_model, model_path)

        best_approach = report.stacking_report.best_approach
        registry = ModelRegistry()
        run_metadata = registry.register(
            model_path=model_path,
            market=market,
            model_name=f"oracle_ensemble_{best_approach}_{suffix}",
            feature_names=report.stacking_report.expert_names,
            metrics={
                "selection_score": report.stacking_report.approach_metrics[best_approach]["selection_score"],
                "pre_log_loss": (report.pre_metrics or {}).get("log_loss"),
                "post_log_loss": (report.post_metrics or {}).get("log_loss"),
                "pre_brier": (report.pre_metrics or {}).get("brier"),
                "post_brier": (report.post_metrics or {}).get("brier"),
            },
            extra={
                "stacking_approach": best_approach,
                "calibration_applied": report.calibration_applied,
                "calibration_method": report.method,
                "fallback_reason": report.fallback_reason,
                "pre_metrics": report.pre_metrics,
                "post_metrics": report.post_metrics,
                "approach_metrics": report.stacking_report.approach_metrics,
                "rows": len(meta_features),
            },
            stage="candidate",
        )

    return OracleEnsembleCalibrationRunResult(
        market=market,
        rows=len(meta_features),
        status="calibrated" if report.calibration_applied else "calibration_fallback",
        calibration_applied=report.calibration_applied,
        best_approach=report.stacking_report.best_approach,
        details={
            "pre_metrics": report.pre_metrics,
            "post_metrics": report.post_metrics,
            "fallback_reason": report.fallback_reason,
            "approach_metrics": report.stacking_report.approach_metrics,
            "run": run_metadata,
        },
    )
