"""Market Regime Detection.

Implements:
- Hidden Markov Model regime detection
- Volatility regime classification (low/medium/high)
- Trend regime detection (trending/ranging)
- Regime transition probability estimation
- Regime-conditional statistics
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegimeState:
    """Current regime information."""
    regime_id: int = 0
    regime_name: str = "unknown"
    confidence: float = 0.0
    volatility_regime: str = "medium"  # low, medium, high
    trend_regime: str = "ranging"  # trending_up, trending_down, ranging
    transition_probs: dict[int, float] = field(default_factory=dict)
    duration: int = 0  # Bars in current regime


class RegimeDetector:
    """Market regime detection using multiple methods.

    Combines volatility-based and trend-based regime classification
    with HMM-inspired state tracking.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.n_regimes = config.get("n_regimes", 3)
        self.vol_lookback = config.get("vol_lookback", 20)
        self.trend_lookback = config.get("trend_lookback", 50)
        self.vol_thresholds = config.get("vol_thresholds", [0.3, 0.7])  # Quantiles
        self._regime_history: list[int] = []
        self._vol_history: list[float] = []
        self._current_regime = 0
        self._regime_duration = 0

    def detect(self, returns: np.ndarray) -> RegimeState:
        """Detect current market regime from returns series."""
        if len(returns) < self.vol_lookback:
            return RegimeState()

        vol_regime = self._volatility_regime(returns)
        trend_regime = self._trend_regime(returns)

        # Combine into unified regime
        regime_id = self._compute_regime_id(vol_regime, trend_regime)

        # Track regime duration
        if regime_id == self._current_regime:
            self._regime_duration += 1
        else:
            self._current_regime = regime_id
            self._regime_duration = 1

        self._regime_history.append(regime_id)

        # Estimate transition probabilities
        trans_probs = self._transition_probabilities()

        confidence = self._regime_confidence(returns, regime_id)

        regime_names = {
            0: "low_vol_ranging",
            1: "low_vol_trending",
            2: "med_vol_ranging",
            3: "med_vol_trending",
            4: "high_vol_ranging",
            5: "high_vol_trending",
        }

        return RegimeState(
            regime_id=regime_id,
            regime_name=regime_names.get(regime_id, "unknown"),
            confidence=confidence,
            volatility_regime=vol_regime,
            trend_regime=trend_regime,
            transition_probs=trans_probs,
            duration=self._regime_duration,
        )

    def _volatility_regime(self, returns: np.ndarray) -> str:
        """Classify volatility regime."""
        recent = returns[-self.vol_lookback:]
        current_vol = np.std(recent)

        self._vol_history.append(current_vol)

        if len(self._vol_history) < 20:
            return "medium"

        vol_array = np.array(self._vol_history[-100:])
        low_thresh = np.quantile(vol_array, self.vol_thresholds[0])
        high_thresh = np.quantile(vol_array, self.vol_thresholds[1])

        if current_vol <= low_thresh:
            return "low"
        elif current_vol >= high_thresh:
            return "high"
        return "medium"

    def _trend_regime(self, returns: np.ndarray) -> str:
        """Classify trend regime using directional consistency."""
        recent = returns[-self.trend_lookback:]
        if len(recent) < 10:
            return "ranging"

        # Cumulative return direction
        cum_ret = np.sum(recent)

        # Fraction of positive returns
        pos_frac = np.mean(recent > 0)

        # Strong trend: consistent direction
        if cum_ret > 0 and pos_frac > 0.55:
            return "trending_up"
        elif cum_ret < 0 and pos_frac < 0.45:
            return "trending_down"
        return "ranging"

    def _compute_regime_id(self, vol_regime: str, trend_regime: str) -> int:
        """Map (vol, trend) to regime ID."""
        vol_map = {"low": 0, "medium": 1, "high": 2}
        trend_map = {"ranging": 0, "trending_up": 1, "trending_down": 1}

        v = vol_map.get(vol_regime, 1)
        t = trend_map.get(trend_regime, 0)
        return v * 2 + t

    def _transition_probabilities(self) -> dict[int, float]:
        """Estimate regime transition probabilities from history."""
        if len(self._regime_history) < 10:
            return {}

        current = self._regime_history[-1]
        transitions = {}
        total = 0

        for i in range(1, len(self._regime_history)):
            if self._regime_history[i - 1] == current:
                next_regime = self._regime_history[i]
                transitions[next_regime] = transitions.get(next_regime, 0) + 1
                total += 1

        if total == 0:
            return {}

        return {k: v / total for k, v in transitions.items()}

    def _regime_confidence(self, returns: np.ndarray, regime_id: int) -> float:
        """Estimate confidence in regime classification."""
        recent = returns[-self.vol_lookback:]
        vol = np.std(recent)
        mean_ret = np.mean(recent)

        # Higher confidence for extreme regimes
        vol_z = abs(vol - np.mean(self._vol_history[-50:])) / (np.std(self._vol_history[-50:]) + 1e-10)
        confidence = min(1.0, 0.5 + vol_z * 0.2)
        return confidence

    def get_regime_stats(
        self, returns: np.ndarray, regime_id: int
    ) -> dict[str, float]:
        """Get statistics conditioned on a specific regime."""
        if len(self._regime_history) != len(returns):
            return {}

        mask = np.array(self._regime_history[-len(returns):]) == regime_id
        if np.sum(mask) < 5:
            return {}

        regime_returns = returns[mask]
        return {
            "mean_return": float(np.mean(regime_returns)),
            "volatility": float(np.std(regime_returns)),
            "sharpe": float(np.mean(regime_returns) / (np.std(regime_returns) + 1e-10)),
            "count": int(np.sum(mask)),
        }

    def dominant_regime(self) -> Optional[int]:
        """Return the most frequently observed regime."""
        if not self._regime_history:
            return None
        counts = np.bincount(self._regime_history)
        return int(np.argmax(counts))
