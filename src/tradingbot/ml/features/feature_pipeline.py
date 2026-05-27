"""ML Feature Pipeline — Build, transform, and manage features for ML models.

Handles feature computation, normalization, selection, and caching
for training and inference pipelines.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.types import OHLCVBar
from ...features.technical import TechnicalIndicators, compute_features

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """End-to-end feature engineering for ML models.

    Computes technical indicators, adds derived features,
    handles normalization, and produces train-ready matrices.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.lookback = cfg.get("lookback", 50)
        self.forward_bars = cfg.get("forward_bars", 5)
        self.label_type = cfg.get("label_type", "direction")  # direction, magnitude, return
        self.normalize = cfg.get("normalize", True)
        self._feature_names: list[str] = []
        self._means: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None

    def build_dataset(
        self,
        bars: list[OHLCVBar],
        fit_scaler: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build feature matrix X and label vector y from OHLCV bars.

        Returns:
            X: (n_samples, n_features) array
            y: (n_samples,) array of labels
            feature_names: list of feature names
        """
        if len(bars) < self.lookback + self.forward_bars + 50:
            return np.array([]), np.array([]), []

        # Compute all features
        features = compute_features(bars)
        closes = np.array([b.close for b in bars])

        # Build feature matrix
        feature_names = sorted(features.keys())
        self._feature_names = feature_names
        X = np.column_stack([features[name] for name in feature_names])

        # Build labels
        y = self._build_labels(closes)

        # Trim to valid range (skip lookback warmup and forward prediction window)
        start = self.lookback
        end = len(bars) - self.forward_bars
        X = X[start:end]
        y = y[start:end]

        # Remove rows with any NaN
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid]
        y = y[valid]

        # Normalize
        if self.normalize and len(X) > 0:
            if fit_scaler:
                self._means = np.mean(X, axis=0)
                self._stds = np.std(X, axis=0)
                self._stds[self._stds == 0] = 1.0
            if self._means is not None:
                X = (X - self._means) / self._stds

        return X, y, feature_names

    def transform(self, bars: list[OHLCVBar]) -> np.ndarray:
        """Transform new bars into features using fitted scaler."""
        features = compute_features(bars)
        feature_names = sorted(features.keys())
        X = np.column_stack([features[name] for name in feature_names])

        if self.normalize and self._means is not None:
            X = (X - self._means) / self._stds

        return X

    def _build_labels(self, closes: np.ndarray) -> np.ndarray:
        """Build labels from close prices."""
        n = len(closes)
        y = np.full(n, np.nan)

        if self.label_type == "direction":
            # Binary: 1 if price goes up, 0 if down
            for i in range(n - self.forward_bars):
                fwd_ret = (closes[i + self.forward_bars] - closes[i]) / closes[i]
                y[i] = 1.0 if fwd_ret > 0 else 0.0

        elif self.label_type == "magnitude":
            # Continuous: forward return magnitude
            for i in range(n - self.forward_bars):
                y[i] = (closes[i + self.forward_bars] - closes[i]) / closes[i]

        elif self.label_type == "return":
            # Continuous: log return
            for i in range(n - self.forward_bars):
                y[i] = np.log(closes[i + self.forward_bars] / closes[i])

        return y

    def get_feature_importance(self, model, feature_names: list[str]) -> dict[str, float]:
        """Extract feature importance from a trained model."""
        importance = {}
        if hasattr(model, "feature_importances_"):
            for name, imp in zip(feature_names, model.feature_importances_):
                importance[name] = float(imp)
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                coef = coef[0]
            for name, c in zip(feature_names, coef):
                importance[name] = float(abs(c))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


class FeatureSelector:
    """Select the most informative features for ML models."""

    def __init__(self, max_features: int = 20):
        self.max_features = max_features

    def select_by_variance(self, X: np.ndarray, names: list[str], threshold: float = 0.01) -> tuple[np.ndarray, list[str]]:
        """Remove low-variance features."""
        variances = np.var(X, axis=0)
        mask = variances > threshold
        # Keep at least some features
        if mask.sum() < 5:
            top_idx = np.argsort(variances)[-self.max_features:]
            mask = np.zeros(len(names), dtype=bool)
            mask[top_idx] = True
        return X[:, mask], [names[i] for i in range(len(names)) if mask[i]]

    def select_by_correlation(self, X: np.ndarray, names: list[str], threshold: float = 0.95) -> tuple[np.ndarray, list[str]]:
        """Remove highly correlated features."""
        if X.shape[1] < 2:
            return X, names

        corr = np.corrcoef(X.T)
        to_remove = set()

        for i in range(corr.shape[0]):
            if i in to_remove:
                continue
            for j in range(i + 1, corr.shape[1]):
                if j in to_remove:
                    continue
                if abs(corr[i, j]) > threshold:
                    # Remove the one with lower variance
                    if np.var(X[:, i]) < np.var(X[:, j]):
                        to_remove.add(i)
                    else:
                        to_remove.add(j)

        mask = [i not in to_remove for i in range(len(names))]
        return X[:, mask], [names[i] for i in range(len(names)) if mask[i]]
