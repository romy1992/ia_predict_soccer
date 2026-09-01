from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression


@dataclass
class FeatureSelectionResult:
    X_selected: pd.DataFrame
    selected_features: list[str]


class FeatureSelectionService:
    """Feature selection helpers used by the multi-market training pipeline."""

    @staticmethod
    def select_k_best(X: pd.DataFrame, y: pd.Series, k: int = 20) -> FeatureSelectionResult:
        if X.empty:
            return FeatureSelectionResult(X_selected=X, selected_features=[])

        k = max(1, min(k, X.shape[1]))
        selector = SelectKBest(score_func=f_classif, k=k)
        transformed = selector.fit_transform(X, y)
        selected = X.columns[selector.get_support()].tolist()
        return FeatureSelectionResult(
            X_selected=pd.DataFrame(transformed, columns=selected, index=X.index),
            selected_features=selected,
        )

    @staticmethod
    def select_rfe(X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 15) -> FeatureSelectionResult:
        if X.empty:
            return FeatureSelectionResult(X_selected=X, selected_features=[])

        n_features_to_select = max(1, min(n_features_to_select, X.shape[1]))
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced")
        selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
        transformed = selector.fit_transform(X, y)
        selected = X.columns[selector.get_support()].tolist()
        return FeatureSelectionResult(
            X_selected=pd.DataFrame(transformed, columns=selected, index=X.index),
            selected_features=selected,
        )

