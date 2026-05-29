"""Puell Multiple — mining revenue indicator.

Implements:
- Daily issuance value calculation
- Puell Multiple (daily issuance / 365d MA issuance)
- Signal generation for market cycle analysis
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PuellResult:
    """Puell Multiple analysis result."""
    puell_multiple: float = 1.0
    daily_issuance_usd: float = 0.0
    avg_issuance_usd: float = 0.0
    signal: str = "neutral"
    zone: str = "neutral"  # extreme_high, high, neutral, low, extreme_low


class PuellMultipleCalculator:
    """Puell Multiple mining indicator calculator."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.block_reward = config.get("block_reward", 3.125)  # post-April 2024 halving
        self.blocks_per_day = config.get("blocks_per_day", 144)
        self.ma_window = config.get("ma_window", 365)
        self.extreme_high = config.get("extreme_high", 4.0)
        self.high = config.get("high", 2.0)
        self.low = config.get("low", 0.5)
        self.extreme_low = config.get("extreme_low", 0.3)
        self._price_history: list[float] = []
        self._issuance_history: list[float] = []

    def update(self, price: float) -> None:
        """Update with new price data."""
        self._price_history.append(price)
        issuance = self.compute_daily_issuance(price)
        self._issuance_history.append(issuance)

    def compute_daily_issuance(self, price: float = None) -> float:
        """Compute daily mining issuance value in USD.

        Daily issuance = block_reward * blocks_per_day * price
        """
        if price is None:
            if not self._price_history:
                return 0.0
            price = self._price_history[-1]

        daily_btc = self.block_reward * self.blocks_per_day
        return daily_btc * price

    def compute_puell_multiple(self) -> float:
        """Compute Puell Multiple.

        Puell = Daily Issuance USD / 365-day MA of Daily Issuance USD
        """
        if len(self._issuance_history) == 0:
            return 1.0

        current = self._issuance_history[-1]

        if len(self._issuance_history) < self.ma_window:
            avg = np.mean(self._issuance_history)
        else:
            avg = np.mean(self._issuance_history[-self.ma_window:])

        if avg <= 0:
            return 1.0

        puell = current / avg
        logger.debug("Puell=%.3f (daily=%.0f, avg=%.0f)", puell, current, avg)
        return puell

    def get_zone(self) -> str:
        """Classify current Puell Multiple into a zone."""
        puell = self.compute_puell_multiple()

        if puell >= self.extreme_high:
            return "extreme_high"
        elif puell >= self.high:
            return "high"
        elif puell <= self.extreme_low:
            return "extreme_low"
        elif puell <= self.low:
            return "low"
        return "neutral"

    def get_signal(self) -> str:
        """Generate signal from Puell Multiple.

        Extreme high -> miners are earning far above average (sell pressure likely)
        Extreme low  -> miners are under duress (potential accumulation zone)
        """
        zone = self.get_zone()

        if zone in ("extreme_high", "high"):
            return "bearish"
        elif zone in ("extreme_low", "low"):
            return "bullish"
        return "neutral"

    def analyze(self) -> PuellResult:
        """Run full Puell Multiple analysis."""
        puell = self.compute_puell_multiple()
        daily = self.compute_daily_issuance()
        zone = self.get_zone()
        signal = self.get_signal()

        if len(self._issuance_history) < self.ma_window:
            avg = np.mean(self._issuance_history) if self._issuance_history else 0.0
        else:
            avg = np.mean(self._issuance_history[-self.ma_window:])

        return PuellResult(
            puell_multiple=round(puell, 4),
            daily_issuance_usd=round(daily, 2),
            avg_issuance_usd=round(avg, 2),
            signal=signal,
            zone=zone,
        )
