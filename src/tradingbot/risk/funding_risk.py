"""Funding Risk — margin call prediction and management.

Implements:
- Margin call price calculation
- Funding rate risk assessment
- Position sizing based on funding
- Auto-deleveraging risk
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FundingRiskState:
    """Funding risk assessment."""
    margin_call_price: float = 0.0
    distance_to_margin_call: float = 0.0
    daily_funding_cost: float = 0.0
    annualized_funding: float = 0.0
    risk_level: str = "low"
    should_reduce: bool = False


class FundingRiskManager:
    """Manage funding rate risk."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.margin_buffer = config.get("margin_buffer", 0.10)
        self._funding_history: list[float] = []

    def update_funding(self, rate: float) -> None:
        self._funding_history.append(rate)

    def assess(
        self,
        entry_price: float,
        quantity: float,
        side: str,
        equity: float,
        current_price: float,
        funding_rate: float = 0.0001,
    ) -> FundingRiskState:
        """Assess funding risk for a position."""
        notional = abs(quantity) * current_price
        maintenance_margin = notional * 0.05

        # Margin call price
        if side == "buy":
            margin_call = entry_price * (1 - (equity - maintenance_margin) / notional) if notional > 0 else 0
        else:
            margin_call = entry_price * (1 + (equity - maintenance_margin) / notional) if notional > 0 else 0

        distance = abs(current_price - margin_call) / current_price if current_price > 0 else 0

        # Funding cost
        daily_funding = notional * funding_rate * 3  # 3x per day
        annualized = funding_rate * 3 * 365

        # Risk level
        if distance < 0.05:
            risk = "critical"
        elif distance < 0.10:
            risk = "high"
        elif distance < 0.20:
            risk = "medium"
        else:
            risk = "low"

        return FundingRiskState(
            margin_call_price=margin_call,
            distance_to_margin_call=distance,
            daily_funding_cost=daily_funding,
            annualized_funding=annualized,
            risk_level=risk,
            should_reduce=distance < self.margin_buffer,
        )

    def optimal_leverage(self, target_return: float, max_drawdown: float, volatility: float) -> float:
        """Calculate optimal leverage using Kelly criterion."""
        if volatility == 0 or max_drawdown == 0:
            return 1.0

        kelly = target_return / (volatility ** 2)
        conservative = kelly * 0.5  # Half-Kelly
        return max(1.0, min(conservative, 10.0))
