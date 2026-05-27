"""Trend Following Strategy — Trade in the direction of the trend.

Uses EMA crossovers with ADX confirmation and ATR-based stops.
"""
from __future__ import annotations

import logging
from typing import Optional

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy
from ...features.technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(Strategy):
    """Trend following using EMA crossover + ADX filter.

    Entry: Fast EMA crosses above slow EMA AND ADX > threshold
    Exit: Fast EMA crosses below slow EMA OR trailing stop hit
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        self._fast_period = genome.features[0].get("fast_period", 21) if genome.features else 21
        self._slow_period = genome.features[0].get("slow_period", 55) if genome.features else 55
        self._adx_threshold = genome.features[0].get("adx_threshold", 25) if genome.features else 25
        self._atr_mult = genome.stop_loss_param
        self._bar_buffer: list[OHLCVBar] = []
        self._min_bars = max(self._slow_period, 30) + 5

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._min_bars:
            return None

        # Keep buffer manageable
        if len(self._bar_buffer) > 500:
            self._bar_buffer = self._bar_buffer[-300:]

        ti = TechnicalIndicators(self._bar_buffer)
        fast_ema = ti.ema(self._fast_period)
        slow_ema = ti.ema(self._slow_period)
        adx = ti.adx(14)

        # Need at least 2 values for crossover detection
        if len(fast_ema) < 2:
            return None

        curr_fast = fast_ema[-1]
        prev_fast = fast_ema[-2]
        curr_slow = slow_ema[-1]
        prev_slow = slow_ema[-2]
        curr_adx = adx[-1]

        if any(x != x for x in [curr_fast, prev_fast, curr_slow, prev_slow, curr_adx]):
            return None

        atr = ti.atr(14)[-1]
        if atr is None or atr != atr:
            atr = bar.close * 0.02

        # Bullish crossover: fast crosses above slow with trend strength
        if prev_fast <= prev_slow and curr_fast > curr_slow and curr_adx > self._adx_threshold:
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,
                strength=min(1.0, (curr_fast - curr_slow) / curr_slow * 100),
                confidence=min(1.0, curr_adx / 50),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe(genome.primary_timeframe) if hasattr(bar, 'genome') else Timeframe.H1,
                stop_loss=bar.close - self._atr_mult * atr,
                take_profit=bar.close + self._atr_mult * atr * self.genome.take_profit_ratio,
                trailing_stop_atr_mult=self._atr_mult,
            )

        # Bearish crossover
        if prev_fast >= prev_slow and curr_fast < curr_slow and curr_adx > self._adx_threshold:
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.SELL,
                strength=min(1.0, (curr_slow - curr_fast) / curr_slow * 100),
                confidence=min(1.0, curr_adx / 50),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                stop_loss=bar.close + self._atr_mult * atr,
                take_profit=bar.close - self._atr_mult * atr * self.genome.take_profit_ratio,
                trailing_stop_atr_mult=self._atr_mult,
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0] if "_" in self.genome.name else "BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
