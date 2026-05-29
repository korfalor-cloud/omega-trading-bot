"""Volatility Targeting — dynamic position sizing based on vol.

Implements:
- Rolling volatility estimation
- Target volatility scaling
- Vol regime detection
- Position size adjustment
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VolTargetResult:
    """Volatility targeting result."""
    current_vol: float = 0.0
    target_vol: float = 0.0
    vol_scalar: float = 1.0
    regime: str = "normal"
    position_adjustment: float = 1.0


class VolatilityTargeter:
    """Dynamic volatility targeting."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.target_vol = config.get("target_vol", 0.15)
        self.lookback = config.get("lookback", 30)
        self.max_scalar = config.get("max_scalar", 2.0)
        self.min_scalar = config.get("min_scalar", 0.25)
        self._return_history: list[float] = []

    def update(self, return_val: float) -> None:
        self._return_history.append(return_val)
        if len(self._return_history) > self.lookback * 3:
            self._return_history = self._return_history[-self.lookback * 2:]

    def compute(self) -> VolTargetResult:
        """Compute vol targeting adjustment."""
        if len(self._return_history) < self.lookback:
            return VolTargetResult(target_vol=self.target_vol)

        returns = np.array(self._return_history[-self.lookback:])
        current_vol = float(np.std(returns) * np.sqrt(365))

        if current_vol > 0:
            vol_scalar = self.target_vol / current_vol
            vol_scalar = max(self.min_scalar, min(self.max_scalar, vol_scalar))
        else:
            vol_scalar = 1.0

        # Regime classification
        if current_vol > self.target_vol * 2:
            regime = "high_vol"
        elif current_vol < self.target_vol * 0.5:
            regime = "low_vol"
        else:
            regime = "normal"

        return VolTargetResult(
            current_vol=current_vol,
            target_vol=self.target_vol,
            vol_scalar=vol_scalar,
            regime=regime,
            position_adjustment=vol_scalar,
        )

    def adjust_position(self, base_size: float) -> float:
        """Adjust position size based on vol target."""
        result = self.compute()
        return base_size * result.position_adjustment
