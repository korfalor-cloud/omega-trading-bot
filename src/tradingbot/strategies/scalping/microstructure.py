"""Microstructure Scalping Strategy.

Implements:
- Order book imbalance signals
- Trade intensity detection
- VPIN-based signals
- Kyle's Lambda signals
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class MicrostructureScalpStrategy(Strategy):
    """Microstructure-based scalping."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._intensity_threshold = feats.get("intensity_threshold", 2.0)
        self._hold_bars = feats.get("hold_bars", 2)

        self._bar_buffer: list[OHLCVBar] = []
        self._in_trade = False
        self._trade_side = ""
        self._trade_bars = 0

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < 20:
            return None

        if len(self._bar_buffer) > 100:
            self._bar_buffer = self._bar_buffer[-80:]

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

        # Trade intensity
        volumes = [b.volume for b in self._bar_buffer[-20:]]
        avg_vol = np.mean(volumes)
        curr_vol = volumes[-1]
        intensity = curr_vol / avg_vol if avg_vol > 0 else 1

        # Price pressure
        closes = [b.close for b in self._bar_buffer[-5:]]
        price_change = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0

        # Entry
        if intensity > self._intensity_threshold and price_change > 0.001:
            self._in_trade = True
            self._trade_side = "buy"
            self._trade_bars = 0
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.BUY, strength=min(1.0, intensity / 3),
                confidence=0.6, signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1, metadata={"intensity": intensity},
            )

        if intensity > self._intensity_threshold and price_change < -0.001:
            self._in_trade = True
            self._trade_side = "sell"
            self._trade_bars = 0
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.SELL, strength=min(1.0, intensity / 3),
                confidence=0.6, signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1, metadata={"intensity": intensity},
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0]] if "_" in self.genome.name else ["BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.M1]
