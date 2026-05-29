"""Stock-to-Flow Model — scarcity-based price prediction.

Implements:
- S2F ratio calculation
- Price prediction model based on PlanB's S2F
- Halving schedule awareness with dynamic block reward
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

# Bitcoin halving schedule: block height -> block reward
HALVING_SCHEDULE: list[tuple[int, float, int]] = [
    (0, 50.0, 2009),
    (210_000, 25.0, 2012),
    (420_000, 12.5, 2016),
    (630_000, 6.25, 2020),
    (840_000, 3.125, 2024),
    (1_050_000, 1.5625, 2028),
    (1_260_000, 0.78125, 2032),
]

BLOCKS_PER_DAY = 144
BLOCKS_PER_YEAR = BLOCKS_PER_DAY * 365
BTC_SUPPLY_CAP = 21_000_000


@dataclass
class S2FResult:
    """Stock-to-Flow analysis result."""
    s2f_ratio: float = 0.0
    current_supply: float = 0.0
    annual_production: float = 0.0
    predicted_price: float = 0.0
    current_price: float = 0.0
    next_halving_height: int = 0
    blocks_until_halving: int = 0
    days_until_halving: float = 0.0
    signal: str = "neutral"


class StockToFlowCalculator:
    """Stock-to-Flow price prediction model."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.current_supply = config.get("current_supply", 19_800_000)
        self.current_block_height = config.get("current_block_height", 840_000)
        self._price_history: list[float] = []
        self._s2f_history: list[float] = []
        # PlanB S2F model: ln(price) = 3.21 * ln(S2F) - 1.6
        self.model_intercept = config.get("model_intercept", -1.6)
        self.model_slope = config.get("model_slope", 3.21)

    def update(self, price: float, block_height: int = None) -> None:
        """Update with new price data."""
        self._price_history.append(price)
        if block_height is not None:
            self.current_block_height = block_height
        s2f = self.compute_s2f()
        if s2f > 0:
            self._s2f_history.append(s2f)

    def get_current_block_reward(self) -> float:
        """Get the current block reward based on halving schedule."""
        reward = 50.0
        for height, r, _ in HALVING_SCHEDULE:
            if self.current_block_height >= height:
                reward = r
            else:
                break
        return reward

    def compute_annual_production(self) -> float:
        """Compute annual BTC production."""
        reward = self.get_current_block_reward()
        return reward * BLOCKS_PER_YEAR

    def compute_s2f(self) -> float:
        """Compute Stock-to-Flow ratio.

        S2F = Stock / Flow
        Stock = current circulating supply
        Flow = annual production
        """
        annual = self.compute_annual_production()
        if annual <= 0:
            return 0.0

        s2f = self.current_supply / annual
        logger.debug("S2F=%.2f (supply=%.0f, annual=%.0f)", s2f, self.current_supply, annual)
        return s2f

    def predict_price(self) -> float:
        """Predict BTC price using the S2F model.

        Model: ln(price) = slope * ln(S2F) + intercept
        Based on PlanB's original cross-asset model.
        """
        s2f = self.compute_s2f()
        if s2f <= 0:
            return 0.0

        log_price = self.model_slope * np.log(s2f) + self.model_intercept
        predicted = np.exp(log_price)

        logger.debug("S2F predicted price=%.0f (S2F=%.2f)", predicted, s2f)
        return float(predicted)

    def get_next_halving(self) -> tuple[int, int, float]:
        """Get next halving info.

        Returns: (next_halving_height, blocks_remaining, days_remaining)
        """
        next_height = None
        for height, _, _ in HALVING_SCHEDULE:
            if height > self.current_block_height:
                next_height = height
                break

        if next_height is None:
            # Beyond known schedule — estimate next halving
            last_height = HALVING_SCHEDULE[-1][0]
            next_height = last_height + 210_000

        blocks_remaining = next_height - self.current_block_height
        days_remaining = blocks_remaining / BLOCKS_PER_DAY

        return next_height, blocks_remaining, days_remaining

    def get_signal(self) -> str:
        """Generate signal based on price vs S2F model prediction.

        If current price << predicted -> bullish (undervalued vs model)
        If current price >> predicted -> bearish (overvalued vs model)
        """
        if not self._price_history:
            return "neutral"

        current = self._price_history[-1]
        predicted = self.predict_price()

        if predicted <= 0:
            return "neutral"

        ratio = current / predicted

        if ratio < 0.5:
            return "strongly_bullish"
        elif ratio < 0.8:
            return "bullish"
        elif ratio > 2.0:
            return "strongly_bearish"
        elif ratio > 1.25:
            return "bearish"
        return "neutral"

    def analyze(self) -> S2FResult:
        """Run full Stock-to-Flow analysis."""
        s2f = self.compute_s2f()
        predicted = self.predict_price()
        signal = self.get_signal()
        next_halving_h, blocks_rem, days_rem = self.get_next_halving()

        return S2FResult(
            s2f_ratio=round(s2f, 2),
            current_supply=self.current_supply,
            annual_production=round(self.compute_annual_production(), 2),
            predicted_price=round(predicted, 2),
            current_price=self._price_history[-1] if self._price_history else 0.0,
            next_halving_height=next_halving_h,
            blocks_until_halving=blocks_rem,
            days_until_halving=round(days_rem, 1),
            signal=signal,
        )
