"""Multi-Timeframe Analysis.

Implements:
- Multi-timeframe trend alignment
- Higher-timeframe bias detection
- Cross-timeframe confirmation
- Timeframe momentum scoring
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...core.enums import Timeframe

logger = logging.getLogger(__name__)


@dataclass
class TimeframeState:
    """State for a single timeframe."""
    timeframe: str = ""
    trend: str = "neutral"  # bullish, bearish, neutral
    momentum: float = 0.0
    volatility: float = 0.0
    strength: float = 0.0


@dataclass
class MTFAnalysis:
    """Multi-timeframe analysis result."""
    bias: str = "neutral"  # Overall bias
    alignment: float = 0.0  # -1 to 1, how aligned timeframes are
    timeframe_states: list[TimeframeState] = None
    confidence: float = 0.0
    recommended_direction: str = ""

    def __post_init__(self):
        if self.timeframe_states is None:
            self.timeframe_states = []


class MultiTimeframeAnalyzer:
    """Multi-timeframe analysis engine.

    Analyzes trends across multiple timeframes to determine
    the overall market bias and find high-probability setups.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.lookback = config.get("lookback", 50)
        self.timeframe_weights = config.get("timeframe_weights", {
            "1h": 0.2, "4h": 0.3, "1d": 0.5,
        })

    def analyze(
        self,
        prices_by_timeframe: dict[str, np.ndarray],
    ) -> MTFAnalysis:
        """Analyze multiple timeframes.

        Args:
            prices_by_timeframe: {timeframe: price_array}
        """
        states = []
        trend_scores = []

        for tf, prices in prices_by_timeframe.items():
            if len(prices) < 10:
                continue

            state = self._analyze_timeframe(tf, prices)
            states.append(state)

            weight = self.timeframe_weights.get(tf, 0.1)
            trend_val = {"bullish": 1, "bearish": -1, "neutral": 0}.get(state.trend, 0)
            trend_scores.append(trend_val * weight * state.strength)

        if not trend_scores:
            return MTFAnalysis(timeframe_states=states)

        total_score = sum(trend_scores)
        alignment = np.clip(total_score, -1, 1)

        if total_score > 0.2:
            bias = "bullish"
            direction = "buy"
        elif total_score < -0.2:
            bias = "bearish"
            direction = "sell"
        else:
            bias = "neutral"
            direction = ""

        # Confidence based on alignment
        confidence = min(1.0, abs(total_score))

        return MTFAnalysis(
            bias=bias,
            alignment=float(alignment),
            timeframe_states=states,
            confidence=confidence,
            recommended_direction=direction,
        )

    def _analyze_timeframe(
        self,
        timeframe: str,
        prices: np.ndarray,
    ) -> TimeframeState:
        """Analyze a single timeframe."""
        if len(prices) < 20:
            return TimeframeState(timeframe=timeframe)

        # Trend via EMA crossover
        fast = self._ema(prices, 10)
        slow = self._ema(prices, 30)

        if fast[-1] > slow[-1]:
            trend = "bullish"
        elif fast[-1] < slow[-1]:
            trend = "bearish"
        else:
            trend = "neutral"

        # Momentum via ROC
        roc = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] != 0 else 0

        # Volatility
        returns = np.diff(np.log(prices))
        vol = np.std(returns) * np.sqrt(252)

        # Strength via ADX-like measure
        strength = min(1.0, abs(roc) * 10)

        return TimeframeState(
            timeframe=timeframe,
            trend=trend,
            momentum=float(roc),
            volatility=float(vol),
            strength=strength,
        )

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average."""
        alpha = 2 / (period + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    def check_alignment(
        self,
        prices_by_timeframe: dict[str, np.ndarray],
        required_agreement: int = 2,
    ) -> bool:
        """Check if enough timeframes agree on direction."""
        analysis = self.analyze(prices_by_timeframe)
        if analysis.bias == "neutral":
            return False

        agreeing = sum(
            1 for s in analysis.timeframe_states
            if s.trend == analysis.bias
        )
        return agreeing >= required_agreement

    def get_bias_strength(
        self,
        prices_by_timeframe: dict[str, np.ndarray],
    ) -> float:
        """Get bias strength from -1 (strong bearish) to 1 (strong bullish)."""
        analysis = self.analyze(prices_by_timeframe)
        return analysis.alignment
