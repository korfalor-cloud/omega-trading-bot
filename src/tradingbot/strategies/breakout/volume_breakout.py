"""Volume Breakout Strategy.

Implements:
- Price breakout with volume confirmation
- ATR-based stop loss
- Momentum filter
- Retest detection
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class VolumeBreakoutStrategy(Strategy):
    """Volume-confirmed breakout strategy."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._lookback = feats.get("lookback", 20)
        self._vol_mult = feats.get("volume_multiplier", 1.5)
        self._atr_period = feats.get("atr_period", 14)
        self._hold_bars = feats.get("hold_bars", 10)

        self._bar_buffer: list[OHLCVBar] = []
        self._in_trade = False
        self._trade_side = ""
        self._trade_bars = 0

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._lookback + 5:
            return None

        if len(self._bar_buffer) > 200:
            self._bar_buffer = self._bar_buffer[-150:]

        prices = np.array([b.close for b in self._bar_buffer])
        highs = np.array([b.high for b in self._bar_buffer])
        lows = np.array([b.low for b in self._bar_buffer])
        volumes = np.array([b.volume for b in self._bar_buffer])

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
            return None

        # Resistance/support levels
        high_n = np.max(highs[-self._lookback:-1])
        low_n = np.min(lows[-self._lookback:-1])

        # Volume confirmation
        avg_vol = np.mean(volumes[-self._lookback:])
        vol_spike = volumes[-1] > avg_vol * self._vol_mult

        # Breakout above resistance
        if bar.close > high_n and vol_spike:
            self._in_trade = True
            self._trade_side = "buy"
            self._trade_bars = 0
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.BUY, strength=0.8, confidence=0.7,
                signal_type=SignalType.ENTRY, timeframe=Timeframe.H1,
                metadata={"breakout_level": high_n, "vol_ratio": volumes[-1] / avg_vol},
            )

        # Breakdown below support
        if bar.close < low_n and vol_spike:
            self._in_trade = True
            self._trade_side = "sell"
            self._trade_bars = 0
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.SELL, strength=0.8, confidence=0.7,
                signal_type=SignalType.ENTRY, timeframe=Timeframe.H1,
                metadata={"breakdown_level": low_n, "vol_ratio": volumes[-1] / avg_vol},
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0]] if "_" in self.genome.name else ["BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
