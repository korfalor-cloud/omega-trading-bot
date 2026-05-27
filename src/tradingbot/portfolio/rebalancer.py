"""Portfolio Rebalancing.

Implements:
- Calendar-based rebalancing (daily, weekly, monthly)
- Threshold-based rebalancing (drift triggers)
- Cost-aware rebalancing (minimize trades)
- Tax-lot optimization
- Rebalancing impact estimation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RebalanceTrade:
    """A proposed rebalancing trade."""
    symbol: str = ""
    side: str = ""  # buy or sell
    quantity: float = 0.0
    current_weight: float = 0.0
    target_weight: float = 0.0
    drift: float = 0.0


@dataclass
class RebalanceResult:
    """Result of a rebalancing analysis."""
    trades: list[RebalanceTrade] = field(default_factory=list)
    total_turnover: float = 0.0
    estimated_cost: float = 0.0
    max_drift: float = 0.0
    should_rebalance: bool = False
    reason: str = ""


class PortfolioRebalancer:
    """Portfolio rebalancing engine.

    Supports calendar and threshold-based rebalancing
    with cost-aware optimization.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.drift_threshold = config.get("drift_threshold", 0.05)  # 5%
        self.cost_per_trade = config.get("cost_per_trade", 0.001)  # 10 bps
        self.min_trade_value = config.get("min_trade_value", 10.0)
        self.rebalance_frequency = config.get("rebalance_frequency", "weekly")
        self.last_rebalance: Optional[datetime] = None

    def check_drift(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> RebalanceResult:
        """Check if rebalancing is needed based on drift threshold."""
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        max_drift = 0.0
        trades = []

        for sym in all_symbols:
            current = current_weights.get(sym, 0.0)
            target = target_weights.get(sym, 0.0)
            drift = abs(current - target)
            max_drift = max(max_drift, drift)

            if drift > self.drift_threshold:
                side = "sell" if current > target else "buy"
                trades.append(RebalanceTrade(
                    symbol=sym,
                    side=side,
                    current_weight=current,
                    target_weight=target,
                    drift=drift,
                ))

        should = max_drift > self.drift_threshold
        reason = f"Max drift {max_drift:.1%} {'exceeds' if should else 'within'} threshold {self.drift_threshold:.1%}"

        return RebalanceResult(
            trades=trades,
            max_drift=max_drift,
            should_rebalance=should,
            reason=reason,
        )

    def check_calendar(
        self,
        current_time: datetime,
    ) -> bool:
        """Check if calendar-based rebalancing is due."""
        if self.last_rebalance is None:
            return True

        delta = current_time - self.last_rebalance

        if self.rebalance_frequency == "daily":
            return delta >= timedelta(days=1)
        elif self.rebalance_frequency == "weekly":
            return delta >= timedelta(days=7)
        elif self.rebalance_frequency == "monthly":
            return delta >= timedelta(days=30)
        elif self.rebalance_frequency == "quarterly":
            return delta >= timedelta(days=90)
        return False

    def compute_trades(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        portfolio_value: float,
        prices: dict[str, float],
    ) -> RebalanceResult:
        """Compute rebalancing trades to reach target weights."""
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        trades = []
        total_turnover = 0.0

        for sym in all_symbols:
            current = current_weights.get(sym, 0.0)
            target = target_weights.get(sym, 0.0)
            drift = target - current

            if abs(drift) < 0.001:
                continue

            price = prices.get(sym, 0.0)
            if price <= 0:
                continue

            trade_value = drift * portfolio_value
            quantity = abs(trade_value) / price

            if quantity * price < self.min_trade_value:
                continue

            side = "buy" if drift > 0 else "sell"
            trades.append(RebalanceTrade(
                symbol=sym,
                side=side,
                quantity=quantity,
                current_weight=current,
                target_weight=target,
                drift=drift,
            ))
            total_turnover += abs(trade_value)

        estimated_cost = total_turnover * self.cost_per_trade

        return RebalanceResult(
            trades=trades,
            total_turnover=total_turnover,
            estimated_cost=estimated_cost,
            max_drift=max(abs(t.drift) for t in trades) if trades else 0.0,
            should_rebalance=len(trades) > 0,
            reason=f"{len(trades)} trades, turnover={total_turnover:.0f}, cost={estimated_cost:.0f}",
        )

    def minimize_trades(
        self,
        trades: list[RebalanceTrade],
        max_trades: int = 10,
    ) -> list[RebalanceTrade]:
        """Reduce number of trades by prioritizing largest drifts."""
        sorted_trades = sorted(trades, key=lambda t: abs(t.drift), reverse=True)
        return sorted_trades[:max_trades]

    def tax_lot_optimize(
        self,
        symbol: str,
        quantity_to_sell: float,
        lots: list[dict],
    ) -> list[dict]:
        """Optimize tax-lot selection for selling.

        Uses highest-cost-first to minimize capital gains.
        """
        # Sort lots by cost basis (highest first)
        sorted_lots = sorted(lots, key=lambda l: l.get("cost_basis", 0), reverse=True)

        remaining = quantity_to_sell
        selected = []

        for lot in sorted_lots:
            if remaining <= 0:
                break
            lot_qty = lot.get("quantity", 0)
            take = min(remaining, lot_qty)
            if take > 0:
                selected.append({
                    "lot_id": lot.get("lot_id", ""),
                    "quantity": take,
                    "cost_basis": lot.get("cost_basis", 0),
                    "acquired_date": lot.get("acquired_date"),
                })
                remaining -= take

        return selected

    def estimate_rebalance_impact(
        self,
        trades: list[RebalanceTrade],
        prices: dict[str, float],
        avg_daily_volume: dict[str, float],
    ) -> dict[str, float]:
        """Estimate market impact of rebalancing trades."""
        impacts = {}
        for trade in trades:
            price = prices.get(trade.symbol, 0)
            adv = avg_daily_volume.get(trade.symbol, 1)
            trade_value = trade.quantity * price
            participation = trade_value / (adv * price) if adv > 0 else 0

            # Simple impact model: 10bps per 1% of ADV
            impact = participation * 0.10
            impacts[trade.symbol] = impact

        return impacts
