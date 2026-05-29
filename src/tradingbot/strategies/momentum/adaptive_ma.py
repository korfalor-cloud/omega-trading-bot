"""Adaptive Moving Average Strategy.

Implements:
- Kaufman Adaptive Moving Average (KAMA)
- Efficiency ratio-based adaptation
- Trend detection with adaptive periods
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class AdaptiveMAStrategy(Strategy):
    """Kaufman Adaptive Moving Average strategy."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._fast_period = feats.get("fast_period", 2)
        self._slow_period = feats.get("slow_period", 30)
        self._kama_period = feats.get("kama_period", 10)

        self._bar_buffer: list[OHLCVBar] = []
        self._in_trade = False
        self._trade_side = ""

    def _kama(self, prices: np.ndarray) -> np.ndarray:
        """Kaufman Adaptive Moving Average."""
        n = len(prices)
        kama = np.full(n, prices[0])

        fast_sc = 2 / (self._fast_period + 1)
        slow_sc = 2 / (self._slow_period + 1)

        for i in range(self._kama_period, n):
            direction = abs(prices[i] - prices[i - self._kama_period])
            volatility = np.sum(np.abs(np.diff(prices[i - self._kama_period:i + 1])))

            er = direction / volatility if volatility > 0 else 0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

            kama[i] = kama[i - 1] + sc * (prices[i] - kama[i - 1])

        return kama

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._kama_period + 10:
            return None

        if len(self._bar_buffer) > 200:
            self._bar_buffer = self._bar_buffer[-150:]

        prices = np.array([b.close for b in self._bar_buffer])
        kama = self._kama(prices)

        # Exit
        if self._in_trade:
            if self._trade_side == "buy" and prices[-1] < kama[-1]:
                self._in_trade = False
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.SELL, strength=0.6, confidence=0.65,
                    signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                )
            if self._trade_side == "sell" and prices[-1] > kama[-1]:
                self._in_trade = False
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.BUY, strength=0.6, confidence=0.65,
                    signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                )
            return None

        # Entry
        if kama[-2] < prices[-2] and kama[-1] > prices[-1]:
            self._in_trade = True
            self._trade_side = "sell"
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.SELL, strength=0.7, confidence=0.65,
                signal_type=SignalType.ENTRY, timeframe=Timeframe.H1,
            )

        if kama[-2] > prices[-2] and kama[-1] < prices[-1]:
            self._in_trade = True
            self._trade_side = "buy"
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.BUY, strength=0.7, confidence=0.65,
                signal_type=SignalType.ENTRY, timeframe=Timeframe.H1,
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0]] if "_" in self.genome.name else ["BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
