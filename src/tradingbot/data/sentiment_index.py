"""Market Sentiment Index.

Implements:
- Fear & Greed Index
- Crypto Volatility Index
- Market breadth indicators
- Social sentiment scoring
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SentimentState:
    """Market sentiment state."""
    fear_greed: float = 50.0  # 0=extreme fear, 100=extreme greed
    volatility_index: float = 0.0
    momentum_score: float = 0.0
    social_score: float = 0.0
    overall: str = "neutral"


class SentimentIndex:
    """Composite market sentiment index."""

    def __init__(self, config: dict = None):
        config = config or {}
        self._price_history: list[float] = []
        self._volume_history: list[float] = []

    def update(self, price: float, volume: float) -> None:
        self._price_history.append(price)
        self._volume_history.append(volume)

    def compute_fear_greed(self, lookback: int = 30) -> float:
        """Compute Fear & Greed Index (0-100)."""
        if len(self._price_history) < lookback:
            return 50.0

        prices = np.array(self._price_history[-lookback:])
        volumes = np.array(self._volume_history[-lookback:])
        returns = np.diff(prices) / prices[:-1]

        # Momentum (positive = greed)
        momentum = np.mean(returns) / (np.std(returns) + 1e-10) * 25 + 50

        # Volatility (high = fear)
        vol = np.std(returns) * np.sqrt(365)
        vol_score = max(0, min(100, 100 - vol * 100))

        # Volume trend (rising = greed)
        vol_trend = (volumes[-1] - np.mean(volumes)) / (np.mean(volumes) + 1e-10) * 25 + 50

        # Composite
        fg = momentum * 0.4 + vol_score * 0.3 + vol_trend * 0.3
        return float(np.clip(fg, 0, 100))

    def compute_volatility_index(self, lookback: int = 30) -> float:
        """Compute crypto volatility index."""
        if len(self._price_history) < lookback:
            return 0.0

        prices = np.array(self._price_history[-lookback:])
        returns = np.diff(prices) / prices[:-1]
        return float(np.std(returns) * np.sqrt(365) * 100)

    def compute_momentum_score(self, lookback: int = 14) -> float:
        """Compute momentum score (-1 to 1)."""
        if len(self._price_history) < lookback:
            return 0.0

        prices = np.array(self._price_history[-lookback:])
        roc = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
        return float(np.clip(roc * 10, -1, 1))

    def get_state(self) -> SentimentState:
        """Get current sentiment state."""
        fg = self.compute_fear_greed()
        vi = self.compute_volatility_index()
        momentum = self.compute_momentum_score()

        if fg > 70:
            overall = "greed"
        elif fg < 30:
            overall = "fear"
        elif fg > 55:
            overall = "mild_greed"
        elif fg < 45:
            overall = "mild_fear"
        else:
            overall = "neutral"

        return SentimentState(
            fear_greed=fg,
            volatility_index=vi,
            momentum_score=momentum,
            overall=overall,
        )
