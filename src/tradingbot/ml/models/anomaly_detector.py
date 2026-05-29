"""Anomaly Detection — autoencoder-based market anomaly detection.

Implements:
- Autoencoder reconstruction error
- Z-score anomaly detection
- Isolation forest (simplified)
- Anomaly scoring
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Anomaly detection result."""
    score: float = 0.0
    is_anomaly: bool = False
    feature_contributions: dict = None
    method: str = ""

    def __post_init__(self):
        if self.feature_contributions is None:
            self.feature_contributions = {}


class AutoencoderDetector:
    """Simple autoencoder for anomaly detection."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.n_features = config.get("n_features", 10)
        self.encoding_dim = config.get("encoding_dim", 3)
        self.threshold = config.get("threshold", 2.0)

        # Simple linear autoencoder weights
        self.encoder = np.random.randn(self.n_features, self.encoding_dim) * 0.1
        self.decoder = np.random.randn(self.encoding_dim, self.n_features) * 0.1
        self._history: list[np.ndarray] = []
        self._reconstruction_errors: list[float] = []

    def encode(self, x: np.ndarray) -> np.ndarray:
        return x @ self.encoder

    def decode(self, z: np.ndarray) -> np.ndarray:
        return z @ self.decoder

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(x))

    def reconstruction_error(self, x: np.ndarray) -> float:
        x_hat = self.reconstruct(x)
        return float(np.mean((x - x_hat) ** 2))

    def fit(self, X: np.ndarray, epochs: int = 100, lr: float = 0.001) -> None:
        """Train autoencoder."""
        for _ in range(epochs):
            for x in X:
                z = self.encode(x)
                x_hat = self.decode(z)
                error = x - x_hat

                # Gradient descent
                self.decoder -= lr * z[:, np.newaxis] * error
                self.encoder -= lr * x[:, np.newaxis] * (error @ self.decoder.T)

    def detect(self, x: np.ndarray) -> AnomalyResult:
        """Detect if input is anomalous."""
        error = self.reconstruction_error(x)
        self._reconstruction_errors.append(error)

        if len(self._reconstruction_errors) > 30:
            mean_err = np.mean(self._reconstruction_errors[-30:])
            std_err = np.std(self._reconstruction_errors[-30:])
            z_score = (error - mean_err) / std_err if std_err > 0 else 0
        else:
            z_score = 0

        return AnomalyResult(
            score=error,
            is_anomaly=abs(z_score) > self.threshold,
            method="autoencoder",
        )


class IsolationDetector:
    """Simplified isolation forest for anomaly detection."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.n_trees = config.get("n_trees", 100)
        self.threshold = config.get("threshold", 0.6)
        self._trees: list[dict] = []

    def fit(self, X: np.ndarray) -> None:
        """Build isolation trees."""
        n_samples = len(X)
        for _ in range(self.n_trees):
            # Random subsample
            idx = np.random.choice(n_samples, min(256, n_samples), replace=False)
            subsample = X[idx]

            # Random split
            feature = np.random.randint(X.shape[1])
            threshold = np.random.uniform(subsample[:, feature].min(), subsample[:, feature].max())

            self._trees.append({"feature": feature, "threshold": threshold})

    def score(self, x: np.ndarray) -> float:
        """Anomaly score (0-1, higher = more anomalous)."""
        if not self._trees:
            return 0.0

        depths = []
        for tree in self._trees:
            if x[tree["feature"]] < tree["threshold"]:
                depths.append(0)  # Short path = anomalous
            else:
                depths.append(1)

        return 1 - np.mean(depths)

    def detect(self, x: np.ndarray) -> AnomalyResult:
        score = self.score(x)
        return AnomalyResult(
            score=score,
            is_anomaly=score > self.threshold,
            method="isolation_forest",
        )
