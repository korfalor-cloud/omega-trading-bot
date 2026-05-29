"""Position Monitor — real-time position P&L tracking.

Implements:
- Real-time P&L per position
- Unrealized/realized P&L
- Position risk metrics
- Margin utilization
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PositionPnL:
    """Position P&L state."""
    symbol: str = ""
    side: str = ""
    entry_price: float = 0.0
    current_price: float = 0.0
    quantity: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    duration_hours: float = 0.0
    max_favorable: float = 0.0
    max_adverse: float = 0.0


class PositionMonitor:
    """Real-time position monitoring."""

    def __init__(self, config: dict = None):
        config = config or {}
        self._positions: dict[str, PositionPnL] = {}
        self._equity = config.get("initial_equity", 100000)

    def update_position(self, symbol: str, side: str, entry_price: float, quantity: float, current_price: float) -> PositionPnL:
        """Update position with current price."""
        sign = 1 if side == "buy" else -1
        unrealized = (current_price - entry_price) * quantity * sign
        unrealized_pct = unrealized / (entry_price * quantity) if entry_price * quantity > 0 else 0

        pos = self._positions.get(symbol, PositionPnL(symbol=symbol))
        pos.side = side
        pos.entry_price = entry_price
        pos.current_price = current_price
        pos.quantity = quantity
        pos.unrealized_pnl = unrealized
        pos.unrealized_pnl_pct = unrealized_pct
        pos.max_favorable = max(pos.max_favorable, unrealized)
        pos.max_adverse = min(pos.max_adverse, unrealized)

        self._positions[symbol] = pos
        return pos

    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close a position and return realized P&L."""
        pos = self._positions.get(symbol)
        if not pos:
            return 0

        sign = 1 if pos.side == "buy" else -1
        realized = (exit_price - pos.entry_price) * pos.quantity * sign
        pos.realized_pnl = realized
        del self._positions[symbol]
        return realized

    def get_all_positions(self) -> list[PositionPnL]:
        return list(self._positions.values())

    def get_total_unrealized(self) -> float:
        return sum(p.unrealized_pnl for p in self._positions.values())

    def get_total_exposure(self) -> float:
        return sum(abs(p.current_price * p.quantity) for p in self._positions.values())

    def get_leverage(self) -> float:
        exposure = self.get_total_exposure()
        return exposure / self._equity if self._equity > 0 else 0

    def get_summary(self) -> dict:
        positions = list(self._positions.values())
        return {
            "n_positions": len(positions),
            "total_unrealized": self.get_total_unrealized(),
            "total_exposure": self.get_total_exposure(),
            "leverage": self.get_leverage(),
            "symbols": [p.symbol for p in positions],
        }
