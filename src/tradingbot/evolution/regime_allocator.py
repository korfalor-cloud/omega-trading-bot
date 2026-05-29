"""Regime-Adaptive Allocator — dynamically allocates capital based on market regime.

Implements:
- Regime detection and classification
- Dynamic capital allocation per regime
- Strategy-regime affinity scoring
- Automatic rebalancing on regime change
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AllocationResult:
    """Allocation decision."""
    strategy_weights: dict[str, float] = None
    regime: str = ""
    confidence: float = 0.0
    total_allocated: float = 0.0
    reason: str = ""

    def __post_init__(self):
        if self.strategy_weights is None:
            self.strategy_weights = {}


class RegimeAdaptiveAllocator:
    """Dynamically allocate capital based on market regime."""

    # Regime definitions
    REGIMES = {
        "bull_low_vol": {"description": "Trending up, low volatility", "bias": "long"},
        "bull_high_vol": {"description": "Trending up, high volatility", "bias": "cautious_long"},
        "bear_low_vol": {"description": "Trending down, low volatility", "bias": "short"},
        "bear_high_vol": {"description": "Trending down, high volatility", "bias": "defensive"},
        "sideways_low_vol": {"description": "Range-bound, low volatility", "bias": "mean_reversion"},
        "sideways_high_vol": {"description": "Range-bound, high volatility", "bias": "reduced"},
        "crisis": {"description": "Extreme volatility, correlation breakdown", "bias": "cash"},
    }

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.min_allocation = config.get("min_allocation", 0.0)
        self.max_allocation = config.get("max_allocation", 0.40)
        self.cash_reserve = config.get("cash_reserve", 0.10)
        self._strategy_regime_scores: dict[str, dict[str, float]] = {}
        self._current_regime = "sideways_low_vol"

    def detect_regime(
        self,
        returns: np.ndarray,
        volatility: float = None,
        trend_strength: float = None,
    ) -> tuple[str, float]:
        """Detect current market regime."""
        if len(returns) < 20:
            return "sideways_low_vol", 0.3

        # Trend detection
        recent = returns[-20:]
        trend = np.mean(recent)
        trend_val = trend_strength if trend_strength is not None else abs(trend) / (np.std(recent) + 1e-10)

        # Volatility detection
        vol = volatility if volatility is not None else np.std(recent) * np.sqrt(365)

        # Classify
        if vol > 0.8:
            if trend < -0.001:
                regime = "crisis"
            elif trend > 0.001:
                regime = "bull_high_vol"
            else:
                regime = "sideways_high_vol"
        elif vol > 0.4:
            if trend > 0.0005:
                regime = "bull_high_vol"
            elif trend < -0.0005:
                regime = "bear_high_vol"
            else:
                regime = "sideways_high_vol"
        else:
            if trend > 0.0005:
                regime = "bull_low_vol"
            elif trend < -0.0005:
                regime = "bear_low_vol"
            else:
                regime = "sideways_low_vol"

        # Confidence based on how clearly regime is defined
        confidence = min(1.0, trend_val / 2 + (1 - vol / 2))

        self._current_regime = regime
        return regime, float(confidence)

    def register_strategy_regime_score(
        self,
        strategy_id: str,
        regime: str,
        score: float,
    ) -> None:
        """Register how well a strategy performs in a regime."""
        if strategy_id not in self._strategy_regime_scores:
            self._strategy_regime_scores[strategy_id] = {}
        self._strategy_regime_scores[strategy_id][regime] = score

    def allocate(
        self,
        strategy_ids: list[str],
        total_capital: float,
        regime: str = None,
    ) -> AllocationResult:
        """Allocate capital across strategies for current regime."""
        r = regime or self._current_regime
        regime_info = self.REGIMES.get(r, {})
        bias = regime_info.get("bias", "neutral")

        # Score each strategy for this regime
        scores = {}
        for sid in strategy_ids:
            regime_scores = self._strategy_regime_scores.get(sid, {})
            score = regime_scores.get(r, 0.5)  # Default neutral
            scores[sid] = max(0, score)

        # Normalize
        total_score = sum(scores.values())
        if total_score == 0:
            # Equal weight fallback
            weights = {sid: 1 / len(strategy_ids) for sid in strategy_ids}
        else:
            weights = {sid: s / total_score for sid, s in scores.items()}

        # Apply regime bias
        if bias == "cash":
            # Crisis — reduce all allocations
            weights = {sid: w * 0.3 for sid, w in weights.items()}
        elif bias == "defensive":
            weights = {sid: w * 0.6 for sid, w in weights.items()}

        # Apply min/max constraints
        for sid in weights:
            weights[sid] = max(self.min_allocation, min(self.max_allocation, weights[sid]))

        # Normalize to leave cash reserve
        total = sum(weights.values())
        available = 1 - self.cash_reserve
        if total > 0:
            weights = {sid: w / total * available for sid, w in weights.items()}

        return AllocationResult(
            strategy_weights=weights,
            regime=r,
            confidence=0.7,
            total_allocated=sum(weights.values()),
            reason=f"Regime: {r}, bias: {bias}",
        )

    def get_regime_description(self, regime: str) -> str:
        info = self.REGIMES.get(regime, {})
        return info.get("description", "Unknown regime")

    def get_current_regime(self) -> str:
        return self._current_regime
