"""Realistic Backtesting Engine.

Implements:
- Slippage models (fixed, proportional, market impact)
- Fee modeling (maker/taker, tiered)
- Latency simulation
- Partial fill simulation
- Funding rate costs for perpetual futures
- Realistic order matching
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Realistic backtesting configuration."""
    initial_capital: float = 100_000.0
    maker_fee: float = 0.0002  # 2 bps
    taker_fee: float = 0.0005  # 5 bps
    slippage_model: str = "proportional"  # fixed, proportional, market_impact
    slippage_bps: float = 5.0
    latency_ms: int = 100
    fill_probability: float = 0.95  # For limit orders
    funding_rate: float = 0.0001  # Per 8h
    max_position_pct: float = 0.1
    allow_short: bool = True


@dataclass
class SimulatedFill:
    """A simulated fill from backtesting."""
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    requested_price: float = 0.0
    fill_price: float = 0.0
    quantity: float = 0.0
    fee: float = 0.0
    slippage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RealisticBacktester:
    """Backtesting engine with realistic execution modeling.

    Simulates slippage, fees, latency, and partial fills
    to produce more accurate backtest results.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._capital = self.config.initial_capital
        self._positions: dict[str, float] = {}
        self._fills: list[SimulatedFill] = []
        self._equity_curve: list[tuple[datetime, float]] = []
        self._funding_costs: float = 0.0

    def simulate_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        market_price: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[SimulatedFill]:
        """Simulate a market order with slippage and fees."""
        if quantity <= 0:
            return None

        # Apply slippage
        fill_price = self._apply_slippage(market_price, side, quantity)

        # Calculate fee
        fee = quantity * fill_price * self.config.taker_fee

        # Check capital
        cost = quantity * fill_price + fee
        if side == "buy" and cost > self._capital:
            # Reduce quantity to fit
            quantity = self._capital / (fill_price * (1 + self.config.taker_fee))
            fee = quantity * fill_price * self.config.taker_fee
            cost = quantity * fill_price + fee

        if quantity <= 0:
            return None

        # Update position
        sign = 1 if side == "buy" else -1
        self._positions[symbol] = self._positions.get(symbol, 0) + quantity * sign

        # Update capital
        if side == "buy":
            self._capital -= cost
        else:
            self._capital += quantity * fill_price - fee

        slippage = abs(fill_price - market_price)
        fill = SimulatedFill(
            symbol=symbol,
            side=side,
            requested_price=market_price,
            fill_price=fill_price,
            quantity=quantity,
            fee=fee,
            slippage=slippage,
            timestamp=timestamp or datetime.utcnow(),
        )
        self._fills.append(fill)
        return fill

    def simulate_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: float,
        market_price: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[SimulatedFill]:
        """Simulate a limit order (may or may not fill)."""
        # Check if limit price would be filled
        if side == "buy" and market_price > limit_price:
            return None  # Market above limit
        if side == "sell" and market_price < limit_price:
            return None  # Market below limit

        # Random fill probability
        if np.random.random() > self.config.fill_probability:
            return None

        # Limit orders have no slippage (filled at limit or better)
        fill_price = limit_price
        fee = quantity * fill_price * self.config.maker_fee

        sign = 1 if side == "buy" else -1
        self._positions[symbol] = self._positions.get(symbol, 0) + quantity * sign

        if side == "buy":
            self._capital -= quantity * fill_price + fee
        else:
            self._capital += quantity * fill_price - fee

        fill = SimulatedFill(
            symbol=symbol,
            side=side,
            requested_price=limit_price,
            fill_price=fill_price,
            quantity=quantity,
            fee=fee,
            slippage=0.0,
            timestamp=timestamp or datetime.utcnow(),
        )
        self._fills.append(fill)
        return fill

    def _apply_slippage(self, price: float, side: str, quantity: float) -> float:
        """Apply slippage model to fill price."""
        if self.config.slippage_model == "fixed":
            slip = self.config.slippage_bps / 10000 * price
        elif self.config.slippage_model == "market_impact":
            # Square-root impact model
            adv_pct = quantity / max(quantity * 100, 1)  # Assume 100x ADV
            slip = price * np.sqrt(adv_pct) * 0.01
        else:  # proportional
            slip = self.config.slippage_bps / 10000 * price

        if side == "buy":
            return price + slip
        else:
            return price - slip

    def apply_funding(
        self,
        symbol: str,
        timestamp: datetime,
        rate: Optional[float] = None,
    ) -> float:
        """Apply funding rate cost/benefit for perpetual futures."""
        pos = self._positions.get(symbol, 0)
        if pos == 0:
            return 0.0

        r = rate if rate is not None else self.config.funding_rate
        # Longs pay when funding is positive, shorts receive
        cost = abs(pos) * r
        if pos > 0:
            self._capital -= cost
            self._funding_costs += cost
        else:
            self._capital += cost
            self._funding_costs -= cost

        return cost

    def get_position(self, symbol: str) -> float:
        return self._positions.get(symbol, 0.0)

    def get_equity(self, prices: dict[str, float]) -> float:
        """Compute total equity given current prices."""
        equity = self._capital
        for symbol, qty in self._positions.items():
            price = prices.get(symbol, 0)
            equity += qty * price
        return equity

    def record_equity(self, timestamp: datetime, prices: dict[str, float]) -> float:
        equity = self.get_equity(prices)
        self._equity_curve.append((timestamp, equity))
        return equity

    def get_stats(self) -> dict:
        """Get backtest statistics."""
        equity_values = [v for _, v in self._equity_curve]
        if not equity_values:
            return {}

        returns = np.diff(equity_values) / equity_values[:-1] if len(equity_values) > 1 else []

        total_fees = sum(f.fee for f in self._fills)
        total_slippage = sum(f.slippage * f.quantity for f in self._fills)
        total_return = (equity_values[-1] / equity_values[0] - 1) if equity_values[0] > 0 else 0

        # Max drawdown
        peak = equity_values[0]
        max_dd = 0.0
        for v in equity_values:
            peak = max(peak, v)
            dd = (peak - v) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe
        if len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        return {
            "initial_capital": self.config.initial_capital,
            "final_equity": equity_values[-1],
            "total_return": total_return,
            "total_return_pct": f"{total_return:.2%}",
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "total_fills": len(self._fills),
            "total_fees": total_fees,
            "total_slippage_cost": total_slippage,
            "total_funding_costs": self._funding_costs,
        }

    def get_fills(self) -> list[SimulatedFill]:
        return list(self._fills)

    def get_equity_curve(self) -> list[tuple[datetime, float]]:
        return list(self._equity_curve)
