"""Gamma Scalping Strategy.

Implements:
- Delta-hedged gamma trading
- Profit from realized vs implied vol
- Dynamic delta hedging
- P&L attribution (gamma + theta)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class GammaScalpStrategy(Strategy):
    """Gamma scalping — delta-hedged vol trading."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._hedge_threshold = feats.get("hedge_threshold", 0.10)
        self._lookback = feats.get("lookback", 30)

        self._bar_buffer: list[OHLCVBar] = []
        self._delta = 0.0
        self._gamma = 0.0
        self._theta = 0.0
        self._position = 0.0

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._lookback:
            return None

        if len(self._bar_buffer) > 200:
            self._bar_buffer = self._bar_buffer[-150:]

        prices = np.array([b.close for b in self._bar_buffer])
        returns = np.diff(np.log(prices))
        realized_vol = np.std(returns) * np.sqrt(365)

        # Simulated Greeks (would use BS in production)
        self._gamma = 0.01 / (bar.close * 0.01)
        self._theta = -bar.close * 0.0001
        self._delta += self._gamma * (prices[-1] - prices[-2]) if len(prices) > 1 else 0

        # Hedge when delta exceeds threshold
        if abs(self._delta) > self._hedge_threshold:
            hedge_qty = -self._delta
            side = Side.BUY if hedge_qty > 0 else Side.SELL
            self._delta = 0
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=side, strength=0.5, confidence=0.7,
                signal_type=SignalType.ENTRY, timeframe=Timeframe.H1,
                metadata={"delta_hedge": True, "gamma": self._gamma},
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0]] if "_" in self.genome.name else ["BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
