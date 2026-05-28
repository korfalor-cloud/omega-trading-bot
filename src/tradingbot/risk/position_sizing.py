"""Position Sizing — Kelly, volatility-targeting, fixed-fractional.

Implements:
- Kelly criterion (half-Kelly, fractional Kelly)
- Volatility-targeting position sizing
- Fixed fractional sizing
- ATR-based sizing
- Risk-per-trade sizing
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    """Result of position sizing calculation."""
    position_size: float = 0.0
    risk_amount: float = 0.0
    method: str = ""
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class PositionSizer:
    """Position sizing toolkit."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.max_position_pct = config.get("max_position_pct", 0.10)
        self.risk_per_trade = config.get("risk_per_trade", 0.01)
        self.kelly_fraction = config.get("kelly_fraction", 0.5)

    def kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        portfolio_value: float,
        price: float,
    ) -> SizingResult:
        """Kelly criterion position sizing.

        f* = (p * b - q) / b
        where p = win rate, q = 1-p, b = avg_win/avg_loss
        """
        if avg_loss == 0 or price == 0:
            return SizingResult(method="kelly")

        b = abs(avg_win / avg_loss)
        q = 1 - win_rate
        kelly = (win_rate * b - q) / b

        # Apply fraction (half-Kelly is standard)
        kelly = max(0, kelly * self.kelly_fraction)
        kelly = min(kelly, self.max_position_pct)

        position_value = portfolio_value * kelly
        position_size = position_value / price

        return SizingResult(
            position_size=position_size,
            risk_amount=position_value,
            method="kelly",
            details={
                "kelly_raw": kelly / self.kelly_fraction,
                "kelly_fraction": self.kelly_fraction,
                "kelly_adjusted": kelly,
                "position_pct": kelly,
            },
        )

    def volatility_target(
        self,
        portfolio_value: float,
        price: float,
        current_vol: float,
        target_vol: float = 0.15,
    ) -> SizingResult:
        """Volatility-targeting position sizing.

        Scale position so portfolio vol = target vol.
        """
        if current_vol == 0 or price == 0:
            return SizingResult(method="vol_target")

        vol_scalar = target_vol / current_vol
        vol_scalar = min(vol_scalar, 2.0)  # Cap at 2x

        position_value = portfolio_value * vol_scalar * self.risk_per_trade * 10
        position_value = min(position_value, portfolio_value * self.max_position_pct)
        position_size = position_value / price

        return SizingResult(
            position_size=position_size,
            risk_amount=position_value,
            method="vol_target",
            details={
                "current_vol": current_vol,
                "target_vol": target_vol,
                "vol_scalar": vol_scalar,
            },
        )

    def risk_per_trade_size(
        self,
        portfolio_value: float,
        price: float,
        stop_distance: float,
    ) -> SizingResult:
        """Size position so max loss = risk_per_trade * portfolio.

        position_size = (portfolio * risk_pct) / stop_distance
        """
        if stop_distance == 0 or price == 0:
            return SizingResult(method="risk_per_trade")

        risk_amount = portfolio_value * self.risk_per_trade
        position_size = risk_amount / stop_distance
        position_value = position_size * price

        # Cap at max position
        max_value = portfolio_value * self.max_position_pct
        if position_value > max_value:
            position_size = max_value / price
            position_value = max_value

        return SizingResult(
            position_size=position_size,
            risk_amount=risk_amount,
            method="risk_per_trade",
            details={
                "stop_distance": stop_distance,
                "risk_pct": self.risk_per_trade,
                "position_pct": position_value / portfolio_value if portfolio_value > 0 else 0,
            },
        )

    def atr_size(
        self,
        portfolio_value: float,
        price: float,
        atr: float,
        atr_multiplier: float = 2.0,
    ) -> SizingResult:
        """ATR-based position sizing.

        Uses ATR as the stop distance.
        """
        stop_distance = atr * atr_multiplier
        return self.risk_per_trade_size(portfolio_value, price, stop_distance)

    def fixed_fractional(
        self,
        portfolio_value: float,
        price: float,
        fraction: float = 0.02,
    ) -> SizingResult:
        """Fixed fractional sizing — allocate fixed % of portfolio."""
        if price == 0:
            return SizingResult(method="fixed_fractional")

        position_value = portfolio_value * fraction
        position_size = position_value / price

        return SizingResult(
            position_size=position_size,
            risk_amount=position_value,
            method="fixed_fractional",
            details={"fraction": fraction},
        )
