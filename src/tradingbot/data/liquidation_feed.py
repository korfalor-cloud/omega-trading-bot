"""Liquidation Feed — liquidation event monitoring.

Implements:
- Liquidation event tracking
- Liquidation clustering detection
- Cascade risk assessment
- Liquidation heatmap
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LiquidationEvent:
    """A liquidation event."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    side: str = ""
    price: float = 0.0
    quantity: float = 0.0
    value: float = 0.0


@dataclass
class LiquidationState:
    """Liquidation market state."""
    total_liquidations_1h: float = 0.0
    long_liquidations: float = 0.0
    short_liquidations: float = 0.0
    cascade_risk: str = "low"
    signal: str = ""


class LiquidationFeed:
    """Liquidation event feed and analysis."""

    def __init__(self, config: dict = None):
        config = config or {}
        self._events: list[LiquidationEvent] = []
        self._max_events = config.get("max_events", 10000)

    def add_event(self, event: LiquidationEvent) -> None:
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

    def add(self, symbol: str, side: str, price: float, quantity: float) -> None:
        self.add_event(LiquidationEvent(
            symbol=symbol, side=side, price=price, quantity=quantity,
            value=price * quantity,
        ))

    def get_recent(self, hours: float = 1) -> list[LiquidationEvent]:
        cutoff = datetime.utcnow().timestamp() - hours * 3600
        return [e for e in self._events if e.timestamp.timestamp() > cutoff]

    def analyze(self, hours: float = 1) -> LiquidationState:
        """Analyze recent liquidations."""
        recent = self.get_recent(hours)

        long_liq = sum(e.value for e in recent if e.side == "sell")
        short_liq = sum(e.value for e in recent if e.side == "buy")
        total = long_liq + short_liq

        # Cascade risk
        if total > 1e6:
            cascade = "high"
        elif total > 5e5:
            cascade = "medium"
        else:
            cascade = "low"

        # Signal
        signal = ""
        if long_liq > short_liq * 2:
            signal = "bearish_cascade"
        elif short_liq > long_liq * 2:
            signal = "bullish_cascade"

        return LiquidationState(
            total_liquidations_1h=total,
            long_liquidations=long_liq,
            short_liquidations=short_liq,
            cascade_risk=cascade,
            signal=signal,
        )

    def get_liquidation_levels(self, symbol: str, bins: int = 20) -> dict[float, float]:
        """Get liquidation price distribution."""
        events = [e for e in self._events if e.symbol == symbol]
        if not events:
            return {}

        prices = [e.price for e in events]
        hist, edges = np.histogram(prices, bins=bins)
        return {(edges[i] + edges[i + 1]) / 2: float(hist[i]) for i in range(len(hist))}
