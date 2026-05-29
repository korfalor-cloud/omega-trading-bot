"""Transaction Cost Analysis (TCA).

Implements:
- Pre-trade cost estimation
- Post-trade cost measurement
- Implementation shortfall decomposition
- VWAP/TWAP benchmark comparison
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TCAResult:
    """Transaction cost analysis result."""
    total_cost: float = 0.0
    market_impact: float = 0.0
    timing_cost: float = 0.0
    spread_cost: float = 0.0
    opportunity_cost: float = 0.0
    commission: float = 0.0
    implementation_shortfall: float = 0.0
    vs_vwap: float = 0.0
    vs_arrival: float = 0.0
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class TransactionCostAnalyzer:
    """Pre- and post-trade transaction cost analysis."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.default_fee_rate = config.get("default_fee_rate", 0.001)

    def pre_trade_estimate(
        self,
        price: float,
        quantity: float,
        avg_volume: float,
        volatility: float = 0.02,
        spread_bps: float = 2.0,
        side: str = "buy",
    ) -> TCAResult:
        """Estimate costs before execution."""
        notional = price * quantity

        # Spread cost (half spread for one side)
        spread_cost = spread_bps / 10000 / 2 * notional

        # Market impact (square-root model)
        participation = quantity / avg_volume if avg_volume > 0 else 0
        impact_pct = volatility * np.sqrt(participation) * 0.1
        market_impact = impact_pct * notional

        # Commission
        commission = notional * self.default_fee_rate

        total = spread_cost + market_impact + commission

        return TCAResult(
            total_cost=total,
            market_impact=market_impact,
            spread_cost=spread_cost,
            commission=commission,
            implementation_shortfall=total / notional if notional > 0 else 0,
        )

    def post_trade_analysis(
        self,
        decision_price: float,
        avg_exec_price: float,
        quantity: float,
        vwap: float,
        arrival_price: float,
        side: str = "buy",
        commission: float = 0.0,
    ) -> TCAResult:
        """Analyze costs after execution."""
        sign = 1 if side == "buy" else -1
        notional = decision_price * quantity

        # Implementation shortfall
        is_cost = (avg_exec_price - decision_price) * quantity * sign

        # vs VWAP benchmark
        vs_vwap = (avg_exec_price - vwap) * quantity * sign

        # vs arrival price
        vs_arrival = (avg_exec_price - arrival_price) * quantity * sign

        # Timing cost (decision to arrival)
        timing = (arrival_price - decision_price) * quantity * sign

        # Market impact (arrival to execution)
        impact = (avg_exec_price - arrival_price) * quantity * sign

        return TCAResult(
            total_cost=is_cost + commission,
            market_impact=impact,
            timing_cost=timing,
            commission=commission,
            implementation_shortfall=is_cost / notional if notional > 0 else 0,
            vs_vwap=vs_vwap,
            vs_arrival=vs_arrival,
        )

    def analyze_fills(
        self,
        fills: list[dict],
        decision_price: float,
        side: str = "buy",
        vwap: float = 0.0,
    ) -> TCAResult:
        """Analyze a list of fills.

        Each fill: {"price": float, "quantity": float, "fee": float}
        """
        if not fills:
            return TCAResult()

        total_qty = sum(f["quantity"] for f in fills)
        total_cost = sum(f.get("fee", 0) for f in fills)
        avg_price = sum(f["price"] * f["quantity"] for f in fills) / total_qty if total_qty > 0 else 0

        if vwap == 0:
            vwap = avg_price

        return self.post_trade_analysis(
            decision_price=decision_price,
            avg_exec_price=avg_price,
            quantity=total_qty,
            vwap=vwap,
            arrival_price=decision_price,
            side=side,
            commission=total_cost,
        )
