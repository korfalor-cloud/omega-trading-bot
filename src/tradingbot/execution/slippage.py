"""Slippage and Market Impact Models.

Implements:
- Linear slippage model
- Square-root market impact (Almgren)
- Volume-based impact
- Spread-based slippage
- Historical slippage estimation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SlippageEstimate:
    """Estimated slippage for a trade."""
    slippage_pct: float = 0.0
    slippage_abs: float = 0.0
    fill_price: float = 0.0
    market_impact: float = 0.0
    spread_cost: float = 0.0
    method: str = ""


class SlippageModel:
    """Market impact and slippage estimation."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.impact_coeff = config.get("impact_coeff", 0.1)
        self.spread_bps = config.get("spread_bps", 2.0)
        self.participation_rate = config.get("participation_rate", 0.10)

    def linear_slippage(
        self,
        price: float,
        quantity: float,
        avg_volume: float,
        side: str = "buy",
    ) -> SlippageEstimate:
        """Linear slippage model.

        slippage = impact_coeff * (quantity / avg_volume)
        """
        if avg_volume <= 0:
            return SlippageEstimate(method="linear", fill_price=price)

        participation = quantity / avg_volume
        impact = self.impact_coeff * participation
        spread = self.spread_bps / 10000 / 2

        total_slip = impact + spread
        sign = 1 if side == "buy" else -1
        fill_price = price * (1 + total_slip * sign)

        return SlippageEstimate(
            slippage_pct=total_slip,
            slippage_abs=abs(fill_price - price),
            fill_price=fill_price,
            market_impact=impact,
            spread_cost=spread,
            method="linear",
        )

    def sqrt_impact(
        self,
        price: float,
        quantity: float,
        avg_volume: float,
        volatility: float = 0.02,
        side: str = "buy",
    ) -> SlippageEstimate:
        """Square-root market impact model (Almgren).

        impact = sigma * sqrt(Q / V) * eta
        """
        if avg_volume <= 0:
            return SlippageEstimate(method="sqrt", fill_price=price)

        participation = quantity / avg_volume
        impact = volatility * np.sqrt(participation) * self.impact_coeff
        spread = self.spread_bps / 10000 / 2

        total_slip = impact + spread
        sign = 1 if side == "buy" else -1
        fill_price = price * (1 + total_slip * sign)

        return SlippageEstimate(
            slippage_pct=total_slip,
            slippage_abs=abs(fill_price - price),
            fill_price=fill_price,
            market_impact=impact,
            spread_cost=spread,
            method="sqrt",
        )

    def historical_slippage(
        self,
        expected_prices: np.ndarray,
        actual_prices: np.ndarray,
        sides: np.ndarray,
    ) -> dict:
        """Estimate slippage from historical execution data."""
        n = min(len(expected_prices), len(actual_prices), len(sides))
        if n == 0:
            return {"avg_slippage": 0, "median_slippage": 0, "n_samples": 0}

        slippages = []
        for i in range(n):
            if sides[i] == "buy":
                slip = (actual_prices[i] - expected_prices[i]) / expected_prices[i]
            else:
                slip = (expected_prices[i] - actual_prices[i]) / expected_prices[i]
            slippages.append(slip)

        slippages = np.array(slippages)
        return {
            "avg_slippage": float(np.mean(slippages)),
            "median_slippage": float(np.median(slippages)),
            "std_slippage": float(np.std(slippages)),
            "max_slippage": float(np.max(slippages)),
            "n_samples": n,
        }

    def estimate_cost(
        self,
        price: float,
        quantity: float,
        avg_volume: float,
        volatility: float = 0.02,
        side: str = "buy",
        fee_rate: float = 0.001,
    ) -> dict:
        """Estimate total execution cost."""
        slip = self.sqrt_impact(price, quantity, avg_volume, volatility, side)
        fee = price * quantity * fee_rate
        notional = price * quantity

        return {
            "notional": notional,
            "slippage_cost": slip.slippage_abs * quantity,
            "fee_cost": fee,
            "total_cost": slip.slippage_abs * quantity + fee,
            "cost_pct": (slip.slippage_abs * quantity + fee) / notional if notional > 0 else 0,
            "fill_price": slip.fill_price,
        }
