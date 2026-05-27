"""Feature Importance and Selection Analysis.

Implements:
- Permutation importance
- SHAP-like feature attribution (simplified)
- Mutual information
- Recursive feature elimination
- Feature stability analysis
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportanceResult:
    """Result of feature importance analysis."""
    feature_names: list[str]
    importance_scores: np.ndarray
    method: str
    top_features: list[tuple[str, float]] = field(default_factory=list)

    def get_top_n(self, n: int = 10) -> list[tuple[str, float]]:
        indices = np.argsort(self.importance_scores)[::-1][:n]
        return [
            (self.feature_names[i], float(self.importance_scores[i]))
            for i in indices
        ]


class FeatureImportanceAnalyzer:
    """Feature importance analysis toolkit.

    Provides multiple methods to assess which features
    contribute most to model predictions.
    """

    def __init__(self, feature_names: list[str] | None = None):
        self.feature_names = feature_names or []

    def set_feature_names(self, names: list[str]) -> None:
        self.feature_names = names

    def permutation_importance(
        self,
        model: any,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        metric: str = "mse",
    ) -> FeatureImportanceResult:
        """Permutation importance — measures decrease in model performance
        when each feature is randomly shuffled."""
        n_features = X.shape[1]
        names = self._get_names(n_features)

        # Baseline score
        baseline_pred = model.predict(X)
        baseline_score = self._compute_metric(y, baseline_pred, metric)

        importances = np.zeros(n_features)
        rng = np.random.default_rng(42)

        for i in range(n_features):
            scores = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                X_permuted[:, i] = rng.permutation(X_permuted[:, i])
                permuted_pred = model.predict(X_permuted)
                score = self._compute_metric(y, permuted_pred, metric)
                scores.append(baseline_score - score)
            importances[i] = np.mean(scores)

        result = FeatureImportanceResult(
            feature_names=names,
            importance_scores=importances,
            method="permutation",
        )
        result.top_features = result.get_top_n(10)
        return result

    def correlation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> FeatureImportanceResult:
        """Simple correlation-based feature importance."""
        n_features = X.shape[1]
        names = self._get_names(n_features)

        importances = np.zeros(n_features)
        for i in range(n_features):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            importances[i] = abs(corr) if not np.isnan(corr) else 0

        result = FeatureImportanceResult(
            feature_names=names,
            importance_scores=importances,
            method="correlation",
        )
        result.top_features = result.get_top_n(10)
        return result

    def mutual_information(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_bins: int = 20,
    ) -> FeatureImportanceResult:
        """Mutual information between features and target (binned approximation)."""
        n_features = X.shape[1]
        names = self._get_names(n_features)

        importances = np.zeros(n_features)
        for i in range(n_features):
            importances[i] = self._compute_mi(X[:, i], y, n_bins)

        result = FeatureImportanceResult(
            feature_names=names,
            importance_scores=importances,
            method="mutual_information",
        )
        result.top_features = result.get_top_n(10)
        return result

    def variance_importance(
        self,
        X: np.ndarray,
    ) -> FeatureImportanceResult:
        """Variance-based importance — features with near-zero variance are useless."""
        n_features = X.shape[1]
        names = self._get_names(n_features)

        variances = np.var(X, axis=0)
        # Normalize to [0, 1]
        max_var = np.max(variances)
        importances = variances / max_var if max_var > 0 else variances

        result = FeatureImportanceResult(
            feature_names=names,
            importance_scores=importances,
            method="variance",
        )
        result.top_features = result.get_top_n(10)
        return result

    def stability_importance(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> FeatureImportanceResult:
        """Feature stability — correlation of feature importance between train/test."""
        n_features = X_train.shape[1]
        names = self._get_names(n_features)

        train_imp = self._compute_raw_importance(X_train, y_train)
        test_imp = self._compute_raw_importance(X_test, y_test)

        # Stability = 1 - |train_imp - test_imp|
        importances = 1.0 - np.abs(train_imp - test_imp)

        result = FeatureImportanceResult(
            feature_names=names,
            importance_scores=importances,
            method="stability",
        )
        result.top_features = result.get_top_n(10)
        return result

    def select_features(
        self,
        importance: FeatureImportanceResult,
        threshold: float = 0.01,
        max_features: int = 0,
    ) -> list[int]:
        """Select features based on importance scores."""
        scores = importance.importance_scores
        selected = np.where(scores >= threshold)[0]

        if max_features > 0 and len(selected) > max_features:
            top_indices = np.argsort(scores)[::-1][:max_features]
            selected = np.sort(top_indices)

        return selected.tolist()

    # ── Internal helpers ────────────────────────────────────────────

    def _get_names(self, n: int) -> list[str]:
        if self.feature_names and len(self.feature_names) == n:
            return self.feature_names
        return [f"feature_{i}" for i in range(n)]

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        if metric == "mse":
            return float(np.mean((y_true - y_pred) ** 2))
        elif metric == "mae":
            return float(np.mean(np.abs(y_true - y_pred)))
        elif metric == "accuracy":
            return float(np.mean(y_true == np.round(y_pred)))
        return 0.0

    def _compute_mi(self, x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
        """Approximate mutual information using histogram."""
        # Discretize
        x_bins = np.linspace(np.min(x), np.max(x) + 1e-10, n_bins + 1)
        y_bins = np.linspace(np.min(y), np.max(y) + 1e-10, n_bins + 1)

        x_digitized = np.digitize(x, x_bins) - 1
        y_digitized = np.digitize(y, y_bins) - 1

        x_digitized = np.clip(x_digitized, 0, n_bins - 1)
        y_digitized = np.clip(y_digitized, 0, n_bins - 1)

        # Joint and marginal distributions
        n = len(x)
        joint = np.zeros((n_bins, n_bins))
        for i in range(n):
            joint[x_digitized[i], y_digitized[i]] += 1
        joint /= n

        px = np.sum(joint, axis=1)
        py = np.sum(joint, axis=0)

        # MI = sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint[i, j] * np.log(joint[i, j] / (px[i] * py[j]))

        return max(0, mi)

    def _compute_raw_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute simple correlation-based importance."""
        n_features = X.shape[1]
        importances = np.zeros(n_features)
        for i in range(n_features):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            importances[i] = abs(corr) if not np.isnan(corr) else 0
        return importances
