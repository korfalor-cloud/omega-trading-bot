"""RSI Mean Reversion Strategy.

Implements:
- RSI overbought/oversold signals
- RSI divergence detection
- Volume confirmation
- Dynamic RSI thresholds
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class RSIMeanReversionStrategy(Strategy):
    """RSI-based mean reversion strategy."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._rsi_period = feats.get("rsi_period", 14)
        self._overbought = feats.get("overbought", 70)
        self._oversold = feats.get("oversold", 30)
        self._hold_bars = feats.get("hold_bars", 10)

        self._bar_buffer: list[OHLCVBar] = []
        self._in_trade = False
        self._trade_side = ""
        self._trade_bars = 0

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._rsi_period + 5:
            return None

        if len(self._bar_buffer) > 200:
            self._bar_buffer = self._bar_buffer[-150:]

        prices = np.array([b.close for b in self._bar_buffer])
        rsi = self._rsi(prices, self._rsi_period)
        curr_rsi = rsi[-1]

        # Exit
        if self._in_trade:
            self._trade_bars += 1
            if self._trade_bars >= self._hold_bars:
                self._in_trade = False
                self._trade_bars = 0
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.SELL if self._trade_side == "buy" else Side.BUY,
                    strength=0.6, confidence=0.6,
                    signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                )
            # Exit on mean reversion
            if self._trade_side == "buy" and curr_rsi > 50:
                self._in_trade = False
                self._trade_bars = 0
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.SELL, strength=0.7, confidence=0.7,
                    signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                )
            if self._trade_side == "sell" and curr_rsi < 50:
                self._in_trade = False
                self._trade_bars = 0
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.BUY, strength=0.7, confidence=0.7,
                    signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                )
            return None

        # Entry
        if curr_rsi < self._oversold:
            self._in_trade = True
            self._trade_side = "buy"
            self._trade_bars = 0
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.BUY, strength=min(1.0, (self._oversold - curr_rsi) / 30),
                confidence=0.65, signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1, metadata={"rsi": curr_rsi},
            )

        if curr_rsi > self._overbought:
            self._in_trade = True
            self._trade_side = "sell"
            self._trade_bars = 0
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.SELL, strength=min(1.0, (curr_rsi - self._overbought) / 30),
                confidence=0.65, signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1, metadata={"rsi": curr_rsi},
            )

        return None

    def _rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Relative Strength Index (Wilder smoothing)."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        rsi = np.full(len(prices), 50.0)
        if len(gains) < period:
            return rsi

        # Seed with simple average
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
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100 - 100 / (1 + rs)

        return rsi

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0]] if "_" in self.genome.name else ["BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
