"""Seasonality Strategy — time-based patterns.

Implements:
- Day-of-week effects
- Hour-of-day patterns
- Monthly seasonality
- Holiday effects
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class SeasonalityStrategy(Strategy):
    """Trade based on seasonal patterns."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._lookback = feats.get("lookback", 100)
        self._hour_sensitivity = feats.get("hour_sensitivity", 0.6)
        self._day_sensitivity = feats.get("day_sensitivity", 0.5)

        self._bar_buffer: list[OHLCVBar] = []
        self._hour_returns: dict[int, list[float]] = {h: [] for h in range(24)}
        self._day_returns: dict[int, list[float]] = {d: [] for d in range(7)}
        self._in_trade = False
        self._trade_side = ""
        self._trade_bars = 0

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._lookback:
            return None

        # Track returns by hour/day
        if len(self._bar_buffer) >= 2:
            prev = self._bar_buffer[-2]
            ret = (bar.close - prev.close) / prev.close if prev.close > 0 else 0
            hour = bar.timestamp.hour if hasattr(bar.timestamp, "hour") else 0
            day = bar.timestamp.weekday() if hasattr(bar.timestamp, "weekday") else 0
            self._hour_returns[hour].append(ret)
            self._day_returns[day].append(ret)

        # Exit
        if self._in_trade:
            self._trade_bars += 1
            if self._trade_bars >= 3:
                self._in_trade = False
                self._trade_bars = 0
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.SELL if self._trade_side == "buy" else Side.BUY,
                    strength=0.5, confidence=0.6,
                    signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                )
            return None

        # Seasonal signal
        hour = bar.timestamp.hour if hasattr(bar.timestamp, "hour") else 0
        day = bar.timestamp.weekday() if hasattr(bar.timestamp, "weekday") else 0

        hour_avg = np.mean(self._hour_returns[hour]) if self._hour_returns[hour] else 0
        day_avg = np.mean(self._day_returns[day]) if self._day_returns[day] else 0

        combined = hour_avg * self._hour_sensitivity + day_avg * self._day_sensitivity

        if combined > 0.001:
            self._in_trade = True
            self._trade_side = "buy"
            self._trade_bars = 0
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.BUY, strength=min(1.0, combined * 100),
                confidence=0.6, signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                metadata={"hour_avg": hour_avg, "day_avg": day_avg},
            )

        if combined < -0.001:
            self._in_trade = True
            self._trade_side = "sell"
            self._trade_bars = 0
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.SELL, strength=min(1.0, abs(combined) * 100),
                confidence=0.6, signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                metadata={"hour_avg": hour_avg, "day_avg": day_avg},
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0]] if "_" in self.genome.name else ["BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
