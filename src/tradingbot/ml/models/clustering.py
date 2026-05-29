"""Market Regime Clustering — Identify market states via unsupervised learning.

Implements:
- K-means clustering (numpy only)
- Regime classification from feature vectors
- Cluster transition tracking over time
- Feature construction: returns, volatility, volume
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegimeState:
    """Current regime classification result."""
    cluster_id: int = -1
    distance: float = 0.0
    confidence: float = 0.0
    features: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamp: float = 0.0


@dataclass
class TransitionRecord:
    """A single regime transition event."""
    from_cluster: int = -1
    to_cluster: int = -1
    timestamp: float = 0.0
    duration_bars: int = 0


class KMeansClusterer:
    """K-means clustering implemented in pure numpy.

    Supports configurable k, initialization via k-means++ style seeding,
    and convergence tracking.
    """

    def __init__(self, k: int = 4, max_iter: int = 100, tol: float = 1e-4, seed: int | None = None):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self._rng = np.random.RandomState(seed)
        self.centroids: np.ndarray = np.array([])
        self._fitted = False

    def fit(self, X: np.ndarray) -> KMeansClusterer:
        """Fit k-means to the data matrix X (n_samples, n_features)."""
        if len(X) < self.k:
            logger.warning("Fewer samples than clusters; reducing k to %d", len(X))
            self.k = max(1, len(X))

        # K-means++ style initialization
        self.centroids = self._init_centroids(X)

        for iteration in range(self.max_iter):
            # Assign each sample to the nearest centroid
            labels = self._assign(X)
            # Update centroids
            new_centroids = np.empty_like(self.centroids)
            for c in range(self.k):
                members = X[labels == c]
                if len(members) > 0:
                    new_centroids[c] = members.mean(axis=0)
                else:
                    # Re-seed empty cluster to a random data point
                    new_centroids[c] = X[self._rng.randint(len(X))]

            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids

            if shift < self.tol:
                logger.debug("K-means converged at iteration %d (shift=%.6f)", iteration, shift)
                break

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each row of X to the nearest cluster."""
        if not self._fitted:
            raise RuntimeError("KMeansClusterer not fitted; call fit() first")
        return self._assign(X)

    def _assign(self, X: np.ndarray) -> np.ndarray:
        """Compute nearest centroid for every row."""
        # Squared distances: (n, k)
        diffs = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        sq_dist = np.sum(diffs ** 2, axis=2)
        return np.argmin(sq_dist, axis=1)

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """K-means++ style centroid initialization."""
        n = len(X)
        centroids = np.empty((self.k, X.shape[1]))
        # First centroid: random
        centroids[0] = X[self._rng.randint(n)]

        for c in range(1, self.k):
            # Distance to nearest existing centroid
            diffs = X[:, np.newaxis, :] - centroids[:c][np.newaxis, :, :]
            sq_dist = np.min(np.sum(diffs ** 2, axis=2), axis=1)
            # Probability proportional to distance squared
            probs = sq_dist / (sq_dist.sum() + 1e-12)
            idx = self._rng.choice(n, p=probs)
            centroids[c] = X[idx]

        return centroids

    def inertia(self, X: np.ndarray) -> float:
        """Sum of squared distances to nearest centroid."""
        labels = self._assign(X)
        total = 0.0
        for c in range(self.k):
            members = X[labels == c]
            if len(members) > 0:
                total += np.sum((members - self.centroids[c]) ** 2)
        return float(total)


class RegimeClassifier:
    """Classify market regimes from price/volume features.

    Builds feature vectors of returns, volatility, and volume change,
    clusters them, and tracks transitions over time.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.n_clusters = config.get("n_clusters", 4)
        self.volatility_window = config.get("volatility_window", 20)
        self.return_window = config.get("return_window", 5)
        self.seed = config.get("seed", 42)

        self._clusterer = KMeansClusterer(
            k=self.n_clusters, seed=self.seed,
        )
        self._current_regime: RegimeState = RegimeState()
        self._regime_history: list[RegimeState] = []
        self._transitions: list[TransitionRecord] = []
        self._bars_in_regime: int = 0
        self._fitted = False

    def build_features(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> np.ndarray:
        """Build regime feature matrix from close prices and volumes.

        Features per time step:
        - Log return over return_window
        - Rolling volatility (std of log returns) over volatility_window
        - Volume change ratio (current / SMA)
        """
        n = len(closes)
        window = max(self.volatility_window, self.return_window) + 1
        if n < window:
            return np.array([]).reshape(0, 3)

        log_returns = np.diff(np.log(np.maximum(closes, 1e-10)))
        features = []

        for i in range(window - 1, n):
            ret = log_returns[i - self.return_window + 1 : i + 1].sum()
            vol = np.std(log_returns[i - self.volatility_window + 1 : i + 1])
            vol_sma = np.mean(volumes[max(0, i - self.volatility_window + 1) : i + 1])
            vol_ratio = volumes[i] / vol_sma if vol_sma > 0 else 1.0
            features.append([ret, vol, vol_ratio])

        return np.array(features)

    def fit(self, closes: np.ndarray, volumes: np.ndarray) -> RegimeClassifier:
        """Fit the clusterer on historical price/volume data."""
        X = self.build_features(closes, volumes)
        if len(X) < self.n_clusters:
            logger.warning("Not enough data to fit regime classifier")
            return self

        self._clusterer.fit(X)
        self._fitted = True
        logger.info("Regime classifier fitted on %d samples with %d clusters", len(X), self.n_clusters)
        return self

    def classify(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
        timestamp: float = 0.0,
    ) -> RegimeState:
        """Classify the current market state and track transitions."""
        if not self._fitted:
            raise RuntimeError("RegimeClassifier not fitted; call fit() first")

        X = self.build_features(closes, volumes)
        if len(X) == 0:
            return RegimeState()

        latest = X[-1:]
        cluster_id = int(self._clusterer.predict(latest)[0])

        # Distance to assigned centroid
        dist = float(np.linalg.norm(latest[0] - self._clusterer.centroids[cluster_id]))

        # Confidence: inverse of distance relative to other clusters
        all_dists = np.linalg.norm(
            self._clusterer.centroids - latest[0], axis=1,
        )
        min_dist = all_dists.min()
        max_dist = all_dists.max()
        confidence = 1.0 - (min_dist / (max_dist + 1e-10))

        state = RegimeState(
            cluster_id=cluster_id,
            distance=dist,
            confidence=confidence,
            features=latest[0],
            timestamp=timestamp,
        )

        # Track transitions
        prev_id = self._current_regime.cluster_id
        if prev_id >= 0 and cluster_id != prev_id:
            self._transitions.append(TransitionRecord(
                from_cluster=prev_id,
                to_cluster=cluster_id,
                timestamp=timestamp,
                duration_bars=self._bars_in_regime,
            ))
            self._bars_in_regime = 0
            logger.info(
                "Regime transition: %d -> %d at %.2f",
                prev_id, cluster_id, timestamp,
            )

        self._bars_in_regime += 1
        self._current_regime = state
        self._regime_history.append(state)
        return state

    @property
    def current_regime(self) -> RegimeState:
        return self._current_regime

    @property
    def transitions(self) -> list[TransitionRecord]:
        return list(self._transitions)

    @property
    def regime_history(self) -> list[RegimeState]:
        return list(self._regime_history)

    def get_transition_matrix(self) -> np.ndarray:
        """Compute empirical transition probability matrix."""
        mat = np.zeros((self.n_clusters, self.n_clusters))
        for t in self._transitions:
            if 0 <= t.from_cluster < self.n_clusters and 0 <= t.to_cluster < self.n_clusters:
                mat[t.from_cluster, t.to_cluster] += 1

        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return mat / row_sums

    def get_regime_stats(self) -> dict:
        """Return per-cluster statistics."""
        if not self._regime_history:
            return {}

        stats: dict[int, dict] = {}
        for state in self._regime_history:
            cid = state.cluster_id
            if cid not in stats:
                stats[cid] = {"count": 0, "avg_distance": 0.0, "avg_confidence": 0.0}
            stats[cid]["count"] += 1
            stats[cid]["avg_distance"] += state.distance
            stats[cid]["avg_confidence"] += state.confidence

        for cid in stats:
            n = stats[cid]["count"]
            stats[cid]["avg_distance"] /= n
            stats[cid]["avg_confidence"] /= n

        return stats

    def get_status(self) -> dict:
        """Return classifier status summary."""
        return {
            "fitted": self._fitted,
            "n_clusters": self.n_clusters,
            "current_regime": self._current_regime.cluster_id,
            "n_transitions": len(self._transitions),
            "bars_in_current_regime": self._bars_in_regime,
        }
