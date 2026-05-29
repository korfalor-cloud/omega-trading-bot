"""NVT Ratio — Network Value to Transactions.

Implements:
- NVT computation from market cap and on-chain transaction volume
- Signal generation based on NVT thresholds
- Historical tracking with rolling statistics
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NVTResult:
    """NVT analysis result."""
    nvt_ratio: float = 0.0
    nvt_signal: float = 0.0  # smoothed NVT (moving average)
    z_score: float = 0.0
    signal: str = "neutral"
    transaction_volume_usd: float = 0.0
    market_cap: float = 0.0


class NVTCalculator:
    """Network Value to Transactions ratio calculator."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.smoothing_window = config.get("smoothing_window", 14)
        self.high_threshold = config.get("high_threshold", 150.0)
        self.low_threshold = config.get("low_threshold", 50.0)
        self.supply = config.get("supply", 19_500_000)
        self._price_history: list[float] = []
        self._tx_volume_history: list[float] = []
        self._nvt_history: list[float] = []

    def update(self, price: float, tx_volume_usd: float) -> None:
        """Update with new price and on-chain transaction volume."""
        self._price_history.append(price)
        self._tx_volume_history.append(tx_volume_usd)

        nvt = self.compute_nvt()
        if nvt > 0:
            self._nvt_history.append(nvt)

    def compute_nvt(self) -> float:
        """Compute raw NVT ratio.

        NVT = Market Cap / Daily Transaction Volume (USD)
        """
        if not self._price_history or not self._tx_volume_history:
            return 0.0

        current_price = self._price_history[-1]
        market_cap = current_price * self.supply
        tx_vol = self._tx_volume_history[-1]

        if tx_vol <= 0:
            return 0.0

        nvt = market_cap / tx_vol
        logger.debug("NVT=%.2f (mcap=%.0f, tx_vol=%.0f)", nvt, market_cap, tx_vol)
        return nvt

    def compute_nvt_signal(self) -> float:
        """Compute smoothed NVT signal (moving average of NVT)."""
        if len(self._nvt_history) < self.smoothing_window:
            return self._nvt_history[-1] if self._nvt_history else 0.0
        return float(np.mean(self._nvt_history[-self.smoothing_window:]))

    def compute_z_score(self, lookback: int = 90) -> float:
        """Compute Z-score of current NVT against historical distribution."""
        if len(self._nvt_history) < lookback:
            return 0.0

        window = np.array(self._nvt_history[-lookback:])
        mean = np.mean(window)
        std = np.std(window)

        if std == 0:
            return 0.0

        current = self._nvt_history[-1]
        return float((current - mean) / std)

    def get_signal(self) -> str:
        """Generate signal from NVT ratio.

        High NVT  -> network overvalued relative to usage (bearish)
        Low NVT   -> network undervalued relative to usage (bullish)
        """
        nvt_signal = self.compute_nvt_signal()

        if nvt_signal >= self.high_threshold:
            return "bearish"
        elif nvt_signal <= self.low_threshold:
            return "bullish"
        return "neutral"

    def get_history_stats(self, lookback: int = 30) -> dict:
        """Get historical NVT statistics."""
        if len(self._nvt_history) < lookback:
            lookback = len(self._nvt_history)

        if lookback == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "current": 0.0}

        window = np.array(self._nvt_history[-lookback:])
        return {
            "mean": float(np.mean(window)),
            "std": float(np.std(window)),
            "min": float(np.min(window)),
            "max": float(np.max(window)),
            "current": float(window[-1]),
        }

    def analyze(self) -> NVTResult:
        """Run full NVT analysis."""
        nvt = self.compute_nvt()
        nvt_signal = self.compute_nvt_signal()
        z = self.compute_z_score()
        signal = self.get_signal()

        return NVTResult(
            nvt_ratio=round(nvt, 2),
            nvt_signal=round(nvt_signal, 2),
            z_score=round(z, 3),
            signal=signal,
            transaction_volume_usd=self._tx_volume_history[-1] if self._tx_volume_history else 0.0,
            market_cap=self._price_history[-1] * self.supply if self._price_history else 0.0,
        )
