"""Calibrazione MULTICLASSE (MARKET-01).

`CalibrationService` (ML-06) e' scritta per classificazione binaria
(P(classe positiva)) ed e' gia' usata in produzione dai mercati "diretti"
esistenti: non viene toccata qui, per non introdurre regressioni.
`CalibratedClassifierCV` di sklearn supporta nativamente il multiclasse
(calibrazione one-vs-rest interna con rinormalizzazione finale), quindi
questo modulo la riusa con lo stesso principio del framework binario
esistente (pre/post metrics su OOF temporali, calibratore versionato) ma
con le metriche multiclasse dedicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from src.ml.evaluation.multiclass_probability_metrics import (
    compute_multiclass_probability_metrics,
    reorder_probabilities_to_labels,
    temporal_oof_multiclass_probabilities,
)


@dataclass
class MulticlassCalibrationResult:
    method: str
    sample_size: int
    class_counts: dict[str, int]
    pre_metrics: dict[str, Any]
    post_metrics: dict[str, Any]
    calibrator: Any


class MulticlassCalibrationService:
    @staticmethod
    def select_method(
        y: pd.Series,
        class_labels: Sequence[str],
        min_isotonic_samples: int = 600,
        min_class_samples: int = 150,
    ) -> str:
        counts = y.value_counts()
        min_count = min((int(counts.get(label, 0)) for label in class_labels), default=0)
        if len(y) >= min_isotonic_samples and min_count >= min_class_samples:
            return "isotonic"
        return "sigmoid"

    @classmethod
    def _temporal_oof_calibrated_probabilities(
        cls,
        estimator,
        X: pd.DataFrame,
        y: pd.Series,
        cv_splits: list[tuple[list[int], list[int]]],
        class_labels: Sequence[str],
        method: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        prob_cols = [f"prob_{label}" for label in class_labels]

        for train_idx, valid_idx in cv_splits:
            if not train_idx or not valid_idx:
                continue

            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
            if y_train.nunique() < 2:
                continue

            current_method = method
            try:
                calibrator = CalibratedClassifierCV(estimator=clone(estimator), method=current_method, cv=3)
                calibrator.fit(X_train, y_train)
            except Exception:
                current_method = "sigmoid"
                calibrator = CalibratedClassifierCV(estimator=clone(estimator), method=current_method, cv=3)
                calibrator.fit(X_train, y_train)

            proba = calibrator.predict_proba(X_valid)
            ordered = reorder_probabilities_to_labels(calibrator.classes_, proba, class_labels)
            for row_pos, idx in enumerate(valid_idx):
                row: dict[str, Any] = {"index": int(idx), "y_true": str(y_valid.iloc[row_pos])}
                for col_idx, col_name in enumerate(prob_cols):
                    row[col_name] = float(ordered[row_pos, col_idx])
                rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["index", "y_true", *prob_cols])

        frame = pd.DataFrame(rows)
        agg = {col: "mean" for col in prob_cols}
        agg["y_true"] = "first"
        return (
            frame.groupby("index", as_index=False)
            .agg(agg)
            .sort_values(by=["index"])
            .reset_index(drop=True)
        )

    @classmethod
    def calibrate_estimator(
        cls,
        estimator,
        X: pd.DataFrame,
        y: pd.Series,
        cv_splits: list[tuple[list[int], list[int]]],
        class_labels: Sequence[str],
    ) -> MulticlassCalibrationResult:
        if X.empty or y.empty:
            raise ValueError("Dataset vuoto: impossibile calibrare")

        prob_cols = [f"prob_{label}" for label in class_labels]

        pre_frame = temporal_oof_multiclass_probabilities(
            estimator=estimator, X=X, y=y, cv_splits=cv_splits, class_labels=class_labels
        )
        if pre_frame.empty:
            raise ValueError("Nessun fold valido per calibration pre-metrics")
        pre_metrics = compute_multiclass_probability_metrics(
            y_true=pre_frame["y_true"].tolist(),
            probabilities=pre_frame[prob_cols].to_numpy(),
            class_labels=class_labels,
        )

        method = cls.select_method(y=y, class_labels=class_labels)
        post_frame = cls._temporal_oof_calibrated_probabilities(
            estimator=estimator, X=X, y=y, cv_splits=cv_splits, class_labels=class_labels, method=method
        )
        if post_frame.empty:
            method = "sigmoid"
            post_frame = cls._temporal_oof_calibrated_probabilities(
                estimator=estimator, X=X, y=y, cv_splits=cv_splits, class_labels=class_labels, method=method
            )
        if post_frame.empty:
            raise ValueError("Nessun fold valido per calibration post-metrics")

        post_metrics = compute_multiclass_probability_metrics(
            y_true=post_frame["y_true"].tolist(),
            probabilities=post_frame[prob_cols].to_numpy(),
            class_labels=class_labels,
        )

        calibrator = CalibratedClassifierCV(estimator=clone(estimator), method=method, cv=3)
        calibrator.fit(X, y)

        counts = y.value_counts()
        class_counts = {str(label): int(counts.get(label, 0)) for label in class_labels}

        return MulticlassCalibrationResult(
            method=method,
            sample_size=int(len(y)),
            class_counts=class_counts,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
            calibrator=calibrator,
        )

