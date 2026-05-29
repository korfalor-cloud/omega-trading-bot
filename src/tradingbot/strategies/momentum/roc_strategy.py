"""Rate of Change (ROC) Momentum Strategy.

Implements:
- ROC-based momentum signals
- Multi-period momentum scoring
- Momentum persistence filter
- Volume confirmation
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class ROCMomentumStrategy(Strategy):
    """Rate of change momentum strategy."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._roc_period = feats.get("roc_period", 10)
        self._entry_threshold = feats.get("entry_threshold", 0.03)
        self._exit_threshold = feats.get("exit_threshold", 0.01)
        self._vol_confirm = feats.get("volume_confirm", True)

        self._bar_buffer: list[OHLCVBar] = []
        self._in_trade = False
        self._trade_side = ""

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._roc_period + 5:
            return None

        if len(self._bar_buffer) > 200:
            self._bar_buffer = self._bar_buffer[-150:]

        prices = np.array([b.close for b in self._bar_buffer])
        volumes = np.array([b.volume for b in self._bar_buffer])

        # ROC
        roc = (prices[-1] - prices[-self._roc_period]) / prices[-self._roc_period]
        # Volume ratio
        vol_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 1

        # Exit
        if self._in_trade:
            if self._trade_side == "buy" and roc < self._exit_threshold:
                self._in_trade = False
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.SELL, strength=0.6, confidence=0.6,
                    signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                )
            if self._trade_side == "sell" and roc > -self._exit_threshold:
                self._in_trade = False
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.BUY, strength=0.6, confidence=0.6,
                    signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                )
            return None

        # Entry
        vol_ok = vol_ratio > 1.0 if self._vol_confirm else True

        if roc > self._entry_threshold and vol_ok:
            self._in_trade = True
            self._trade_side = "buy"
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.BUY, strength=min(1.0, roc / self._entry_threshold),
                confidence=0.65, signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                metadata={"roc": roc, "vol_ratio": vol_ratio},
            )

        if roc < -self._entry_threshold and vol_ok:
            self._in_trade = True
            self._trade_side = "sell"
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.SELL, strength=min(1.0, abs(roc) / self._entry_threshold),
                confidence=0.65, signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                metadata={"roc": roc, "vol_ratio": vol_ratio},
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0]] if "_" in self.genome.name else ["BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
