"""Position Tracker — Real-time position and P&L tracking.

Implements:
- Real-time position tracking
- Unrealized P&L calculation
- Position-level risk metrics
- Multi-asset portfolio view
- Trade reconciliation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """Detailed position information."""
    symbol: str = ""
    quantity: float = 0.0
    average_entry: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    market_value: float = 0.0
    cost_basis: float = 0.0
    side: str = ""  # long, short, flat
    opened_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    @property
    def net_pnl(self) -> float:
        return self.unrealized_pnl + self.realized_pnl - self.total_fees


class PositionTracker:
    """Real-time position tracking engine.

    Tracks positions, computes P&L, and provides portfolio views.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.initial_capital = config.get("initial_capital", 100_000.0)
        self._positions: dict[str, PositionInfo] = {}
        self._trade_log: list[dict] = []
        self._equity_curve: list[tuple[datetime, float]] = []

    def update_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        fee: float = 0.0,
    ) -> PositionInfo:
        """Update position with a new fill."""
        pos = self._positions.get(symbol)
        if pos is None:
            pos = PositionInfo(symbol=symbol, opened_at=datetime.utcnow())
            self._positions[symbol] = pos

        sign = 1 if side == "buy" else -1
        old_qty = pos.quantity
        old_cost = pos.average_entry * abs(old_qty)

        new_qty = old_qty + quantity * sign

        # Update average entry for increasing positions
        if (old_qty >= 0 and sign > 0) or (old_qty <= 0 and sign < 0):
            # Adding to position
            pos.average_entry = (old_cost + price * quantity) / abs(new_qty) if new_qty != 0 else 0
        elif abs(new_qty) < abs(old_qty):
            # Reducing position — realize P&L
            if old_qty > 0:
                pnl = (price - pos.average_entry) * quantity
            else:
                pnl = (pos.average_entry - price) * quantity
            pos.realized_pnl += pnl

        pos.quantity = new_qty
        pos.current_price = price
        pos.total_fees += fee
        pos.last_updated = datetime.utcnow()

        # Update side
        if pos.quantity > 0:
            pos.side = "long"
        elif pos.quantity < 0:
            pos.side = "short"
        else:
            pos.side = "flat"

        # Update unrealized P&L
        self._update_unrealized_pnl(symbol, price)

        # Log trade
        self._trade_log.append({
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "fee": fee,
            "timestamp": datetime.utcnow(),
        })

        return pos

    def update_price(self, symbol: str, price: float) -> Optional[PositionInfo]:
        """Update current price for a position."""
        pos = self._positions.get(symbol)
        if pos is None:
            return None

        pos.current_price = price
        pos.last_updated = datetime.utcnow()
        self._update_unrealized_pnl(symbol, price)
        return pos

    def _update_unrealized_pnl(self, symbol: str, price: float) -> None:
        pos = self._positions[symbol]
        if pos.quantity == 0:
            pos.unrealized_pnl = 0
            pos.unrealized_pnl_pct = 0
            pos.market_value = 0
            pos.cost_basis = 0
            return

        pos.cost_basis = pos.average_entry * abs(pos.quantity)
        pos.market_value = price * abs(pos.quantity)

        if pos.quantity > 0:
            pos.unrealized_pnl = (price - pos.average_entry) * pos.quantity
        else:
            pos.unrealized_pnl = (pos.average_entry - price) * abs(pos.quantity)

        if pos.cost_basis > 0:
            pos.unrealized_pnl_pct = pos.unrealized_pnl / pos.cost_basis

    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        return self._positions.get(symbol)

    def get_all_positions(self) -> list[PositionInfo]:
        return [p for p in self._positions.values() if p.quantity != 0]

    def get_portfolio_value(self) -> float:
        """Total portfolio value including market value of positions."""
        total = self.initial_capital
        for pos in self._positions.values():
            total += pos.net_pnl
        return total

    def get_portfolio_summary(self) -> dict:
        """Get portfolio summary."""
        positions = self.get_all_positions()
        total_market_value = sum(abs(p.market_value) for p in positions)
        total_unrealized = sum(p.unrealized_pnl for p in positions)
        total_realized = sum(p.realized_pnl for p in positions)
        total_fees = sum(p.total_fees for p in positions)

        return {
            "n_positions": len(positions),
            "total_market_value": total_market_value,
            "unrealized_pnl": total_unrealized,
            "realized_pnl": total_realized,
            "total_fees": total_fees,
            "net_pnl": total_unrealized + total_realized - total_fees,
            "portfolio_value": self.get_portfolio_value(),
            "positions": {
                p.symbol: {
                    "quantity": p.quantity,
                    "side": p.side,
                    "avg_entry": p.average_entry,
                    "current_price": p.current_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "unrealized_pnl_pct": p.unrealized_pnl_pct,
                }
                for p in positions
            },
        }

    def record_equity(self, timestamp: Optional[datetime] = None) -> float:
        """Record current equity to curve."""
        value = self.get_portfolio_value()
        self._equity_curve.append((timestamp or datetime.utcnow(), value))
        return value

    def get_equity_curve(self) -> list[tuple[datetime, float]]:
        return list(self._equity_curve)

    def get_max_drawdown(self) -> float:
        """Compute max drawdown from equity curve."""
        if len(self._equity_curve) < 2:
            return 0.0
        values = [v for _, v in self._equity_curve]
        peak = values[0]
        max_dd = 0.0
        for v in values:
            peak = max(peak, v)
            dd = (peak - v) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd

    def get_trade_log(self, symbol: str = "") -> list[dict]:
        if symbol:
            return [t for t in self._trade_log if t["symbol"] == symbol]
        return list(self._trade_log)
