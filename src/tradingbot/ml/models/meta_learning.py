"""Meta-Learning for Strategy Selection — Match strategies to market contexts.

Implements:
- Strategy-context matching via feature similarity
- Performance prediction for strategy-context pairs
- Online adaptation of strategy weights
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyContext:
    """Snapshot of market context features."""
    features: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamp: float = 0.0
    regime_id: int = -1


@dataclass
class StrategyPerformance:
    """Tracked performance for a single strategy."""
    strategy_id: str = ""
    returns: list[float] = field(default_factory=list)
    sharpe: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0
    n_trades: int = 0


@dataclass
class MetaPrediction:
    """Meta-learner's strategy recommendation."""
    strategy_id: str = ""
    predicted_return: float = 0.0
    confidence: float = 0.0
    context_similarity: float = 0.0


class StrategyContextMatcher:
    """Match strategies to market contexts using nearest-neighbour similarity.

    Stores historical (context, performance) pairs and finds the strategies
    that performed best in the most similar past contexts.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.similarity_window = config.get("similarity_window", 100)
        self.min_history = config.get("min_history", 5)

        # Stored context-performance history: list of (features, strategy_id, return)
        self._history: list[tuple[np.ndarray, str, float]] = []

    def add_observation(self, features: np.ndarray, strategy_id: str, realized_return: float) -> None:
        """Record a (context, strategy, outcome) observation."""
        self._history.append((features.copy(), strategy_id, realized_return))
        if len(self._history) > self.similarity_window * 20:
            self._history = self._history[-self.similarity_window * 20:]

    def predict(self, context_features: np.ndarray, strategy_ids: list[str]) -> list[MetaPrediction]:
        """Predict expected return for each strategy given the current context."""
        if len(self._history) < self.min_history:
            return [
                MetaPrediction(strategy_id=sid, predicted_return=0.0, confidence=0.0)
                for sid in strategy_ids
            ]

        # Compute similarity to every stored context
        stored_features = np.array([h[0] for h in self._history])
        dists = np.linalg.norm(stored_features - context_features, axis=1)

        # Use k nearest observations
        k = min(self.similarity_window, len(self._history))
        nearest_idx = np.argsort(dists)[:k]
        nearest_dists = dists[nearest_idx]

        # Gaussian kernel weights
        bandwidth = np.median(nearest_dists) + 1e-10
        weights = np.exp(-0.5 * (nearest_dists / bandwidth) ** 2)

        predictions: list[MetaPrediction] = []
        for sid in strategy_ids:
            weighted_return = 0.0
            total_weight = 0.0
            n_matches = 0

            for rank, idx in enumerate(nearest_idx):
                _, obs_sid, obs_ret = self._history[idx]
                if obs_sid == sid:
                    weighted_return += weights[rank] * obs_ret
                    total_weight += weights[rank]
                    n_matches += 1

            if total_weight > 0:
                pred_return = weighted_return / total_weight
                confidence = min(1.0, n_matches / self.min_history)
            else:
                pred_return = 0.0
                confidence = 0.0

            predictions.append(MetaPrediction(
                strategy_id=sid,
                predicted_return=float(pred_return),
                confidence=float(confidence),
                context_similarity=float(np.mean(weights)),
            ))

        return predictions

    def best_strategy(self, context_features: np.ndarray, strategy_ids: list[str]) -> MetaPrediction:
        """Return the strategy with the highest predicted return."""
        preds = self.predict(context_features, strategy_ids)
        if not preds:
            return MetaPrediction()
        return max(preds, key=lambda p: p.predicted_return)

    def get_status(self) -> dict:
        return {
            "n_observations": len(self._history),
            "n_strategies": len(set(h[1] for h in self._history)),
        }


class PerformancePredictor:
    """Predict strategy performance from context features using linear regression.

    Maintains a per-strategy linear model that is updated incrementally.
    """

    def __init__(self, n_features: int, lr: float = 0.01):
        self.n_features = n_features
        self.lr = lr
        # Per-strategy weights: strategy_id -> (weights, bias)
        self._models: dict[str, tuple[np.ndarray, float]] = {}

    def _get_model(self, strategy_id: str) -> tuple[np.ndarray, float]:
        if strategy_id not in self._models:
            self._models[strategy_id] = (np.zeros(self.n_features), 0.0)
        return self._models[strategy_id]

    def update(self, features: np.ndarray, strategy_id: str, realized_return: float) -> None:
        """Online gradient descent update for one observation."""
        w, b = self._get_model(strategy_id)
        pred = float(features @ w + b)
        error = pred - realized_return
        # Gradient step
        w -= self.lr * error * features
        b -= self.lr * error
        self._models[strategy_id] = (w, b)

    def predict(self, features: np.ndarray, strategy_id: str) -> float:
        """Predict return for a strategy given context features."""
        w, b = self._get_model(strategy_id)
        return float(features @ w + b)

    def predict_all(self, features: np.ndarray, strategy_ids: list[str]) -> dict[str, float]:
        """Predict return for multiple strategies."""
        return {sid: self.predict(features, sid) for sid in strategy_ids}


class MetaLearner:
    """Combined meta-learning system for strategy selection.

    Integrates context matching and performance prediction with
    online adaptation of strategy weights based on realized outcomes.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.n_features = config.get("n_features", 5)
        self.adaptation_rate = config.get("adaptation_rate", 0.05)
        self.min_trades = config.get("min_trades", 3)
        self.decay = config.get("decay", 0.99)

        self._matcher = StrategyContextMatcher(config)
        self._predictor = PerformancePredictor(self.n_features, lr=config.get("lr", 0.01))
        self._strategy_weights: dict[str, float] = {}
        self._performance: dict[str, StrategyPerformance] = {}

    def register_strategy(self, strategy_id: str, initial_weight: float = 1.0) -> None:
        """Register a strategy for meta-learning."""
        self._strategy_weights[strategy_id] = initial_weight
        self._performance[strategy_id] = StrategyPerformance(strategy_id=strategy_id)
        logger.info("Registered strategy: %s (weight=%.2f)", strategy_id, initial_weight)

    def observe(
        self,
        context_features: np.ndarray,
        strategy_id: str,
        realized_return: float,
    ) -> None:
        """Record an observation: context + strategy + outcome."""
        self._matcher.add_observation(context_features, strategy_id, realized_return)
        self._predictor.update(context_features, strategy_id, realized_return)

        perf = self._performance.get(strategy_id)
        if perf is not None:
            perf.returns.append(realized_return)
            perf.n_trades = len(perf.returns)
            perf.avg_return = float(np.mean(perf.returns))
            if len(perf.returns) > 1:
                std = float(np.std(perf.returns))
                perf.sharpe = perf.avg_return / std if std > 0 else 0.0
            perf.win_rate = sum(1 for r in perf.returns if r > 0) / len(perf.returns)

        # Adapt weight: increase if positive return, decrease if negative
        w = self._strategy_weights.get(strategy_id, 1.0)
        w = w * self.decay + self.adaptation_rate * realized_return
        self._strategy_weights[strategy_id] = max(0.01, w)

    def select_strategy(self, context_features: np.ndarray) -> str:
        """Select the best strategy for the given context."""
        strategy_ids = list(self._strategy_weights.keys())
        if not strategy_ids:
            return ""

        # Blend context-match prediction with linear predictor
        match_preds = self._matcher.predict(context_features, strategy_ids)
        linear_preds = self._predictor.predict_all(context_features, strategy_ids)

        best_id = ""
        best_score = -np.inf

        for pred in match_preds:
            linear_score = linear_preds.get(pred.strategy_id, 0.0)
            # Weighted blend: context similarity + linear model + strategy weight
            weight = self._strategy_weights.get(pred.strategy_id, 1.0)
            score = (
                0.4 * pred.predicted_return * pred.confidence
                + 0.4 * linear_score
                + 0.2 * weight
            )
            if score > best_score:
                best_score = score
                best_id = pred.strategy_id

        logger.debug("Selected strategy: %s (score=%.4f)", best_id, best_score)
        return best_id

    def get_strategy_weights(self) -> dict[str, float]:
        """Return current strategy weights (normalised)."""
        total = sum(self._strategy_weights.values())
        if total <= 0:
            return dict(self._strategy_weights)
        return {sid: w / total for sid, w in self._strategy_weights.items()}

    def get_performance_summary(self) -> dict[str, dict]:
        """Return performance stats for each registered strategy."""
        summary: dict[str, dict] = {}
        for sid, perf in self._performance.items():
            summary[sid] = {
                "n_trades": perf.n_trades,
                "avg_return": perf.avg_return,
                "sharpe": perf.sharpe,
                "win_rate": perf.win_rate,
                "weight": self._strategy_weights.get(sid, 0.0),
            }
        return summary

    def get_status(self) -> dict:
        return {
            "n_strategies": len(self._strategy_weights),
            "weights": self.get_strategy_weights(),
            "matcher_observations": self._matcher.get_status()["n_observations"],
        }
