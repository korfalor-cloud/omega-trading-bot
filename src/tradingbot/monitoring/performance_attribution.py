"""Performance Attribution — P&L decomposition.

Implements:
- Factor-based P&L attribution
- Strategy attribution
- Symbol attribution
- Time-based attribution
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """P&L attribution result."""
    total_pnl: float = 0.0
    strategy_attribution: dict = field(default_factory=dict)
    symbol_attribution: dict = field(default_factory=dict)
    factor_attribution: dict = field(default_factory=dict)
    time_attribution: dict = field(default_factory=dict)


class PerformanceAttribution:
    """P&L attribution engine."""

    def __init__(self, config: dict = None):
        config = config or {}
        self._trades: list[dict] = []

    def add_trade(self, trade: dict) -> None:
        self._trades.append(trade)

    def attribute_by_strategy(self) -> dict[str, float]:
        """Attribute P&L by strategy."""
        result = {}
        for trade in self._trades:
            sid = trade.get("strategy_id", "unknown")
            result[sid] = result.get(sid, 0) + trade.get("pnl", 0)
        return result

    def attribute_by_symbol(self) -> dict[str, float]:
        """Attribute P&L by symbol."""
        result = {}
        for trade in self._trades:
            sym = trade.get("symbol", "unknown")
            result[sym] = result.get(sym, 0) + trade.get("pnl", 0)
        return result

    def attribute_by_factor(self, factor_loadings: dict[str, dict[str, float]]) -> dict[str, float]:
        """Attribute P&L by factor."""
        result = {}
        for trade in self._trades:
            strategy = trade.get("strategy_id", "")
            pnl = trade.get("pnl", 0)
            loadings = factor_loadings.get(strategy, {})
            for factor, loading in loadings.items():
                result[factor] = result.get(factor, 0) + pnl * loading
        return result

    def attribute_by_time(self, period: str = "hour") -> dict[str, float]:
        """Attribute P&L by time period."""
        result = {}
        for trade in self._trades:
            ts = trade.get("timestamp")
            if ts:
                if period == "hour":
                    key = f"{ts.hour:02d}:00"
                elif period == "day":
                    key = ts.strftime("%A")
                elif period == "month":
                    key = ts.strftime("%Y-%m")
                else:
                    key = str(ts.date())
                result[key] = result.get(key, 0) + trade.get("pnl", 0)
        return result

    def get_full_attribution(self) -> AttributionResult:
        """Get complete P&L attribution."""
        total = sum(t.get("pnl", 0) for t in self._trades)
        return AttributionResult(
            total_pnl=total,
            strategy_attribution=self.attribute_by_strategy(),
            symbol_attribution=self.attribute_by_symbol(),
            time_attribution=self.attribute_by_time(),
        )
