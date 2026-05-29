"""MVRV Ratio — Market Value to Realized Value calculator.

Implements:
- MVRV computation from price history
- Z-score calculation for statistical normalization
- Signal generation (overvalued/undervalued/neutral)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MVRVResult:
    """MVRV analysis result."""
    mvrv_ratio: float = 1.0
    z_score: float = 0.0
    market_value: float = 0.0
    realized_value: float = 0.0
    signal: str = "neutral"
    confidence: float = 0.0


class MVRVCalculator:
    """Market Value to Realized Value ratio calculator."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.lookback_days = config.get("lookback_days", 365)
        self.z_score_upper = config.get("z_score_upper", 2.0)
        self.z_score_lower = config.get("z_score_lower", -1.0)
        self.supply = config.get("supply", 19_500_000)  # BTC circulating supply
        self._price_history: list[float] = []
        self._mvrv_history: list[float] = []

    def update(self, price: float) -> None:
        """Update with new price data."""
        self._price_history.append(price)
        mvrv = self.compute_mvrv()
        if mvrv > 0:
            self._mvrv_history.append(mvrv)

    def compute_mvrv(self) -> float:
        """Compute current MVRV ratio.

        MVRV = Market Cap / Realized Cap
        Market Cap = current_price * circulating_supply
        Realized Cap ~ mean_price * circulating_supply (simplified)
        """
        if len(self._price_history) < self.lookback_days:
            return 1.0

        current_price = self._price_history[-1]
        lookback = self._price_history[-self.lookback_days:]
        realized_price = np.mean(lookback)

        if realized_price <= 0:
            return 1.0

        market_value = current_price * self.supply
        realized_value = realized_price * self.supply
        ratio = market_value / realized_value

        logger.debug("MVRV=%.3f (market=%.0f, realized=%.0f)", ratio, market_value, realized_value)
        return ratio

    def compute_z_score(self, lookback: int = None) -> float:
        """Compute Z-score of MVRV ratio.

        Z = (current_mvrv - mean_mvrv) / std_mvrv
        """
        lookback = lookback or self.lookback_days
        if len(self._mvrv_history) < lookback:
            return 0.0

        window = np.array(self._mvrv_history[-lookback:])
        mean = np.mean(window)
        std = np.std(window)

        if std == 0:
            return 0.0

        current = self._mvrv_history[-1]
        z_score = (current - mean) / std

        logger.debug("MVRV Z-score=%.2f (current=%.3f, mean=%.3f, std=%.3f)", z_score, current, mean, std)
        return float(z_score)

    def get_signal(self) -> str:
        """Generate trading signal from MVRV Z-score.

        Z > upper  -> overvalued (bearish)
        Z < lower  -> undervalued (bullish)
        otherwise  -> neutral
        """
        z = self.compute_z_score()

        if z >= self.z_score_upper:
            return "overvalued"
        elif z <= self.z_score_lower:
            return "undervalued"
        return "neutral"

    def analyze(self) -> MVRVResult:
        """Run full MVRV analysis."""
        mvrv = self.compute_mvrv()
        z = self.compute_z_score()
        signal = self.get_signal()

        # Confidence scales with distance from neutral zone
        abs_z = abs(z)
        if abs_z >= self.z_score_upper or abs_z >= abs(self.z_score_lower):
            confidence = min(1.0, abs_z / 3.0)
        else:
            confidence = abs_z / max(abs(self.z_score_upper), abs(self.z_score_lower))

        current_price = self._price_history[-1] if self._price_history else 0.0

        return MVRVResult(
            mvrv_ratio=mvrv,
            z_score=z,
            market_value=current_price * self.supply,
            realized_value=np.mean(self._price_history[-self.lookback_days:]) * self.supply if len(self._price_history) >= self.lookback_days else 0.0,
            signal=signal,
            confidence=round(confidence, 3),
        )
