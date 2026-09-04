from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from src.ml.evaluation.probability_metrics import compute_probability_metrics, temporal_oof_probabilities


@dataclass
class CalibrationResult:
    method: str
    sample_size: int
    positives: int
    negatives: int
    pre_metrics: dict[str, Any]
    post_metrics: dict[str, Any]
    calibrator: Any


class CalibrationService:
    @staticmethod
    def select_method(y: pd.Series, min_isotonic_samples: int = 400, min_class_samples: int = 120) -> str:
        positives = int((y == 1).sum())
        negatives = int((y == 0).sum())
        if len(y) >= min_isotonic_samples and positives >= min_class_samples and negatives >= min_class_samples:
            return "isotonic"
        return "sigmoid"

    @staticmethod
    def _class1_probability(values) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            return arr
        if arr.shape[1] == 1:
            return arr[:, 0]
        return arr[:, -1]

    @classmethod
    def _temporal_oof_calibrated_probabilities(
        cls,
        estimator,
        X: pd.DataFrame,
        y: pd.Series,
        cv_splits: list[tuple[list[int], list[int]]],
        method: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for train_idx, valid_idx in cv_splits:
            if not train_idx or not valid_idx:
                continue

            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_valid = y.iloc[valid_idx]

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

            prob = calibrator.predict_proba(X_valid)
            p1 = cls._class1_probability(prob)
            for idx, p, target in zip(valid_idx, p1, y_valid):
                rows.append({"index": int(idx), "probability": float(p), "y_true": int(target)})

        if not rows:
            return pd.DataFrame(columns=["index", "probability", "y_true"])

        frame = pd.DataFrame(rows)
        return (
            frame.groupby("index", as_index=False)
            .agg({"probability": "mean", "y_true": "first"})
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
    ) -> CalibrationResult:
        if X.empty or y.empty:
            raise ValueError("Dataset vuoto: impossibile calibrare")

        pre_frame = temporal_oof_probabilities(
            estimator=estimator,
            X=X,
            y=y,
            cv_splits=cv_splits,
        )
        if pre_frame.empty:
            raise ValueError("Nessun fold valido per calibration pre-metrics")
        pre_p1 = pre_frame["probability"].astype(float).to_numpy()
        pre_y = pre_frame["y_true"].astype(int).to_numpy()
        pre_metrics = compute_probability_metrics(y_true=pre_y, probabilities=pre_p1, n_bins=10)

        method = cls.select_method(y=y)

        post_frame = cls._temporal_oof_calibrated_probabilities(
            estimator=estimator,
            X=X,
            y=y,
            cv_splits=cv_splits,
            method=method,
        )
        if post_frame.empty:
            method = "sigmoid"
            post_frame = cls._temporal_oof_calibrated_probabilities(
                estimator=estimator,
                X=X,
                y=y,
                cv_splits=cv_splits,
                method=method,
            )
        if post_frame.empty:
            raise ValueError("Nessun fold valido per calibration post-metrics")

        post_p1 = post_frame["probability"].astype(float).to_numpy()
        post_y = post_frame["y_true"].astype(int).to_numpy()
        post_metrics = compute_probability_metrics(y_true=post_y, probabilities=post_p1, n_bins=10)

        calibrator = CalibratedClassifierCV(estimator=clone(estimator), method=method, cv=3)
        calibrator.fit(X, y)

        positives = int((y == 1).sum())
        negatives = int((y == 0).sum())
        return CalibrationResult(
            method=method,
            sample_size=int(len(y)),
            positives=positives,
            negatives=negatives,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
            calibrator=calibrator,
        )

