"""Meta-Learner — learns which strategies work in which conditions.

Implements:
- Strategy-regime mapping
- Feature importance for strategy selection
- Online learning of strategy weights
- Automatic strategy retirement/promotion
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyScore:
    """Score for a strategy in current conditions."""
    strategy_id: str = ""
    confidence: float = 0.0
    expected_sharpe: float = 0.0
    recommended_weight: float = 0.0
    regime_fit: float = 0.0


class MetaLearner:
    """Learns which strategies to deploy in which conditions."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.min_history = config.get("min_history", 10)
        self.decay_factor = config.get("decay_factor", 0.95)
        self._strategy_history: dict[str, list[float]] = {}
        self._regime_history: dict[str, list[str]] = {}
        self._strategy_regime_map: dict[str, dict[str, float]] = {}

    def record_outcome(
        self,
        strategy_id: str,
        pnl: float,
        regime: str = "unknown",
    ) -> None:
        """Record a strategy outcome."""
        if strategy_id not in self._strategy_history:
            self._strategy_history[strategy_id] = []
            self._regime_history[strategy_id] = []
            self._strategy_regime_map[strategy_id] = {}

        self._strategy_history[strategy_id].append(pnl)
        self._regime_history[strategy_id].append(regime)

        # Update regime mapping
        if regime not in self._strategy_regime_map[strategy_id]:
            self._strategy_regime_map[strategy_id][regime] = 0
        # Exponential moving average
        prev = self._strategy_regime_map[strategy_id][regime]
        self._strategy_regime_map[strategy_id][regime] = (
            prev * self.decay_factor + pnl * (1 - self.decay_factor)
        )

    def score_strategy(
        self,
        strategy_id: str,
        current_regime: str = "unknown",
    ) -> StrategyScore:
        """Score a strategy for current conditions."""
        history = self._strategy_history.get(strategy_id, [])
        if len(history) < self.min_history:
            return StrategyScore(strategy_id=strategy_id, confidence=0.1)

        # Recent performance (exponentially weighted)
        weights = np.array([self.decay_factor ** i for i in range(len(history))][::-1])
        pnls = np.array(history)
        weighted_pnl = np.average(pnls, weights=weights)

        # Regime fit
        regime_map = self._strategy_regime_map.get(strategy_id, {})
        regime_fit = regime_map.get(current_regime, 0)

        # Sharpe estimate
        if np.std(pnls) > 0:
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(365)
        else:
            sharpe = 0

        # Confidence based on history length and consistency
        consistency = 1 - min(1, np.std(pnls) / (abs(np.mean(pnls)) + 1e-10))
        confidence = min(1, len(history) / 100) * consistency

        # Weight recommendation
        score = weighted_pnl * 0.4 + regime_fit * 0.3 + sharpe * 0.3
        weight = max(0, min(1, score / 10))

        return StrategyScore(
            strategy_id=strategy_id,
            confidence=confidence,
            expected_sharpe=sharpe,
            recommended_weight=weight,
            regime_fit=regime_fit,
        )

    def rank_strategies(self, current_regime: str = "unknown") -> list[StrategyScore]:
        """Rank all strategies for current conditions."""
        scores = []
        for sid in self._strategy_history:
            scores.append(self.score_strategy(sid, current_regime))
        scores.sort(key=lambda s: s.expected_sharpe * s.confidence, reverse=True)
        return scores

    def should_retire(self, strategy_id: str, threshold: float = -0.5) -> bool:
        """Check if a strategy should be retired."""
        history = self._strategy_history.get(strategy_id, [])
        if len(history) < 20:
            return False
        recent = history[-20:]
        return np.mean(recent) < threshold

    def should_promote(self, strategy_id: str, threshold: float = 1.0) -> bool:
        """Check if a strategy should get more capital."""
        history = self._strategy_history.get(strategy_id, [])
        if len(history) < 10:
            return False
        recent = history[-10:]
        return np.mean(recent) > threshold

    def get_strategy_stats(self, strategy_id: str) -> dict:
        history = self._strategy_history.get(strategy_id, [])
        if not history:
            return {"n_trades": 0}

        pnls = np.array(history)
        return {
            "n_trades": len(pnls),
            "total_pnl": float(np.sum(pnls)),
            "avg_pnl": float(np.mean(pnls)),
            "std_pnl": float(np.std(pnls)),
            "sharpe": float(np.mean(pnls) / np.std(pnls) * np.sqrt(365)) if np.std(pnls) > 0 else 0,
            "win_rate": float(np.sum(pnls > 0) / len(pnls)),
            "regime_map": dict(self._strategy_regime_map.get(strategy_id, {})),
        }
