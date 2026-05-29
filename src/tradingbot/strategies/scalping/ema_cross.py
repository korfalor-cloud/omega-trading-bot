"""EMA Cross Scalping Strategy.

Implements:
- Fast/slow EMA crossover on short timeframes
- RSI filter for overbought/oversold
- Quick profit targets with tight stops
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class EMACrossScalpStrategy(Strategy):
    """EMA crossover scalping strategy."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._fast_period = feats.get("fast_period", 5)
        self._slow_period = feats.get("slow_period", 13)
        self._rsi_filter = feats.get("rsi_filter", True)
        self._hold_bars = feats.get("hold_bars", 3)

        self._bar_buffer: list[OHLCVBar] = []
        self._in_trade = False
        self._trade_side = ""
        self._trade_bars = 0

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._slow_period + 5:
            return None

        if len(self._bar_buffer) > 100:
            self._bar_buffer = self._bar_buffer[-80:]

        prices = np.array([b.close for b in self._bar_buffer])
        fast = self._ema(prices, self._fast_period)
        slow = self._ema(prices, self._slow_period)

        # Exit
        if self._in_trade:
            self._trade_bars += 1
            if self._trade_bars >= self._hold_bars:
                self._in_trade = False
                self._trade_bars = 0
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.SELL if self._trade_side == "buy" else Side.BUY,
                    strength=0.5, confidence=0.6,
                    signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                )
            # Exit on cross back
            if self._trade_side == "buy" and fast[-1] < slow[-1]:
                self._in_trade = False
                self._trade_bars = 0
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.SELL, strength=0.6, confidence=0.65,
                    signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                )
            if self._trade_side == "sell" and fast[-1] > slow[-1]:
                self._in_trade = False
                self._trade_bars = 0
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.BUY, strength=0.6, confidence=0.65,
                    signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                )
            return None

        # RSI filter
        if self._rsi_filter:
            rsi = self._rsi(prices, 14)
            curr_rsi = rsi[-1]
            if curr_rsi > 80 or curr_rsi < 20:
                return None  # Extreme, skip

        # Entry: fast crosses above slow
        if fast[-2] <= slow[-2] and fast[-1] > slow[-1]:
            self._in_trade = True
            self._trade_side = "buy"
            self._trade_bars = 0
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.BUY, strength=0.7, confidence=0.65,
                signal_type=SignalType.ENTRY, timeframe=Timeframe.H1,
            )

        # Fast crosses below slow
        if fast[-2] >= slow[-2] and fast[-1] < slow[-1]:
            self._in_trade = True
            self._trade_side = "sell"
            self._trade_bars = 0
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.SELL, strength=0.7, confidence=0.65,
                signal_type=SignalType.ENTRY, timeframe=Timeframe.H1,
            )

        return None

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    def _rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        rsi = np.full(len(prices), 50.0)
        if len(gains) < period:
            return rsi
        avg_gain = np.sum(gains[:period]) / period
        avg_loss = np.sum(losses[:period]) / period
        if avg_loss == 0:
            rsi[period] = 100.0
        elif avg_gain == 0:
            rsi[period] = 0.0
        else:
            rsi[period] = 100 - 100 / (1 + avg_gain / avg_loss)
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss < 1e-10:
                rsi[i + 1] = 100.0
            elif avg_gain < 1e-10:
                rsi[i + 1] = 0.0
            else:
                rsi[i + 1] = 100 - 100 / (1 + avg_gain / avg_loss)
        return rsi

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0]] if "_" in self.genome.name else ["BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.M5]
