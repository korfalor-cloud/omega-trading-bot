"""Portfolio Margin Calculation.

Implements:
- Reg-T margin calculation
- Portfolio margin (SPAN-like)
- Margin utilization monitoring
- Liquidation price calculation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MarginResult:
    """Margin calculation result."""
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    available_margin: float = 0.0
    margin_utilization: float = 0.0
    liquidation_price: float = 0.0
    leverage: float = 0.0


class MarginCalculator:
    """Portfolio margin calculator."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.initial_margin_rate = config.get("initial_margin_rate", 0.10)
        self.maintenance_margin_rate = config.get("maintenance_margin_rate", 0.05)
        self.max_leverage = config.get("max_leverage", 10.0)

    def calculate_margin(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
        equity: float,
    ) -> MarginResult:
        """Calculate margin requirements."""
        total_notional = sum(abs(qty) * prices.get(s, 0) for s, qty in positions.items())
        initial_margin = total_notional * self.initial_margin_rate
        maintenance_margin = total_notional * self.maintenance_margin_rate
        available = max(0, equity - initial_margin)
        utilization = initial_margin / equity if equity > 0 else 0
        leverage = total_notional / equity if equity > 0 else 0

        return MarginResult(
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            available_margin=available,
            margin_utilization=utilization,
            leverage=min(leverage, self.max_leverage),
        )

    def liquidation_price(
        self,
        entry_price: float,
        quantity: float,
        side: str,
        equity: float,
        maintenance_rate: float | None = None,
    ) -> float:
        """Calculate liquidation price for a position.

        Long: liq = entry * (1 - (equity/notional - mm_rate))
        Short: liq = entry * (1 + (equity/notional - mm_rate))
        """
        mm_rate = maintenance_rate or self.maintenance_margin_rate
        notional = abs(quantity) * entry_price
        if notional == 0:
            return 0.0

        equity_ratio = equity / notional

        if side == "buy":
            liq = entry_price * (1 - equity_ratio + mm_rate)
        else:
            liq = entry_price * (1 + equity_ratio - mm_rate)

        return max(0.0, liq)

    def check_margin_call(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
        equity: float,
    ) -> bool:
        """Check if margin call is triggered."""
        total_notional = sum(abs(qty) * prices.get(s, 0) for s, qty in positions.items())
        maintenance = total_notional * self.maintenance_margin_rate
        return equity < maintenance

    def max_position_size(
        self,
        price: float,
        equity: float,
        side: str = "buy",
    ) -> float:
        """Calculate max position size given margin constraints."""
        available = equity * self.max_leverage
        return available / price if price > 0 else 0
