"""Order Flow Scalping Strategy.

Implements:
- Order flow imbalance detection
- Trade intensity analysis
- Bid/ask pressure signals
- Quick in-and-out scalping
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class OrderFlowScalpStrategy(Strategy):
    """Order flow-based scalping strategy."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._imbalance_threshold = feats.get("imbalance_threshold", 0.3)
        self._intensity_threshold = feats.get("intensity_threshold", 2.0)
        self._hold_bars = feats.get("hold_bars", 3)

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

        # Order flow imbalance
        volumes = np.array([b.volume for b in self._bar_buffer[-20:]])
        closes = np.array([b.close for b in self._bar_buffer[-20:]])

        # Buy volume = volume * (close - low) / range
        # Sell volume = volume * (high - close) / range
        buy_vol = 0
        sell_vol = 0
        for b in self._bar_buffer[-5:]:
            rng = b.high - b.low
            if rng > 0:
                buy_vol += b.volume * (b.close - b.low) / rng
                sell_vol += b.volume * (b.high - b.close) / rng

        total = buy_vol + sell_vol
        if total == 0:
            return None

        imbalance = (buy_vol - sell_vol) / total

        # Trade intensity
        avg_vol = np.mean(volumes)
        curr_vol = volumes[-1]
        intensity = curr_vol / avg_vol if avg_vol > 0 else 1

        # Entry signals
        if imbalance > self._imbalance_threshold and intensity > self._intensity_threshold:
            self._in_trade = True
            self._trade_side = "buy"
            self._trade_bars = 0
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.BUY, strength=min(1.0, abs(imbalance)),
                confidence=0.6, signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                metadata={"imbalance": imbalance, "intensity": intensity},
            )

        if imbalance < -self._imbalance_threshold and intensity > self._intensity_threshold:
            self._in_trade = True
            self._trade_side = "sell"
            self._trade_bars = 0
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.SELL, strength=min(1.0, abs(imbalance)),
                confidence=0.6, signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                metadata={"imbalance": imbalance, "intensity": intensity},
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0]] if "_" in self.genome.name else ["BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.M1]
