"""Trade Blotter — real-time trade display.

Implements:
- Real-time trade feed
- P&L tracking per trade
- Trade filtering
- Summary statistics
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BlotterEntry:
    """A trade blotter entry."""
    id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    side: str = ""
    price: float = 0.0
    quantity: float = 0.0
    value: float = 0.0
    fee: float = 0.0
    strategy_id: str = ""
    pnl: float = 0.0
    metadata: dict = field(default_factory=dict)


class TradeBlotter:
    """Real-time trade blotter."""

    def __init__(self, config: dict = None):
        config = config or {}
        self._entries: list[BlotterEntry] = []
        self._max_entries = config.get("max_entries", 10000)

    def add(self, entry: BlotterEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

    def add_trade(self, symbol: str, side: str, price: float, quantity: float, fee: float = 0, strategy_id: str = "", pnl: float = 0) -> None:
        self.add(BlotterEntry(
            symbol=symbol, side=side, price=price, quantity=quantity,
            value=price * quantity, fee=fee, strategy_id=strategy_id, pnl=pnl,
        ))

    def get_entries(self, symbol: str = "", strategy_id: str = "", limit: int = 100) -> list[BlotterEntry]:
        entries = self._entries
        if symbol:
            entries = [e for e in entries if e.symbol == symbol]
        if strategy_id:
            entries = [e for e in entries if e.strategy_id == strategy_id]
        return entries[-limit:]

    def get_summary(self) -> dict:
        if not self._entries:
            return {"total_trades": 0}

        pnls = [e.pnl for e in self._entries]
        return {
            "total_trades": len(self._entries),
            "total_volume": sum(e.value for e in self._entries),
            "total_fees": sum(e.fee for e in self._entries),
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls),
            "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
        }

    def clear(self) -> None:
        self._entries.clear()
