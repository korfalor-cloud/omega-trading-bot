"""SuperTrend Strategy.

Trades based on SuperTrend indicator direction changes.
Simple trend-following with ATR-based trailing stops.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy
from ...features.advanced_indicators import AdvancedIndicators

logger = logging.getLogger(__name__)


class SuperTrendStrategy(Strategy):
    """SuperTrend indicator strategy.

    BUY when SuperTrend flips from bearish to bullish.
    SELL when SuperTrend flips from bullish to bearish.

    Parameters (from genome.features):
        st_period: ATR period for SuperTrend (default 10)
        st_multiplier: ATR multiplier (default 3.0)
        use_macd: Use MACD for confirmation (default True)
        use_volume: Require volume above average (default True)
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._st_period = feats.get("st_period", 10)
        self._st_multiplier = feats.get("st_multiplier", 3.0)
        self._use_macd = feats.get("use_macd", True)
        self._use_volume = feats.get("use_volume", True)
        self._atr_mult = genome.stop_loss_param
        self._bar_buffer: list[OHLCVBar] = []
        self._min_bars = self._st_period * 3 + 10

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._min_bars:
            return None

        if len(self._bar_buffer) > 300:
            self._bar_buffer = self._bar_buffer[-200:]

        ai = AdvancedIndicators(self._bar_buffer)
        st_line, st_dir = ai.supertrend(self._st_period, self._st_multiplier)

        curr_dir = st_dir[-1]
        prev_dir = st_dir[-2] if len(st_dir) > 1 else 0

        if np.isnan(curr_dir) or np.isnan(prev_dir):
            return None

        # Volume filter
        if self._use_volume:
            vol_avg = np.mean([b.volume for b in self._bar_buffer[-20:]])
            if vol_avg > 0 and bar.volume < vol_avg:
                return None

        # MACD confirmation
        if self._use_macd:
            from ...features.technical import TechnicalIndicators
            ti = TechnicalIndicators(self._bar_buffer)
            macd, signal, hist = ti.macd()
            curr_hist = hist[-1]
            if curr_hist != curr_hist:
                return None

        atr = AdvancedIndicators(self._bar_buffer)
        from ...features.technical import TechnicalIndicators
        ti = TechnicalIndicators(self._bar_buffer)
        curr_atr = ti.atr(14)[-1]
        if curr_atr is None or curr_atr != curr_atr:
            curr_atr = bar.close * 0.02

        # Direction change: bearish to bullish
        if prev_dir <= 0 and curr_dir > 0:
            if self._use_macd and curr_hist < 0:
                return None  # MACD doesn't confirm

            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,
                strength=0.6,
                confidence=0.7,
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                stop_loss=float(st_line[-1]) if not np.isnan(st_line[-1]) else bar.close - self._atr_mult * curr_atr,
                take_profit=bar.close + self._atr_mult * curr_atr * self.genome.take_profit_ratio,
                trailing_stop_atr_mult=self._atr_mult,
            )

        # Direction change: bullish to bearish
        if prev_dir >= 0 and curr_dir < 0:
            if self._use_macd and curr_hist > 0:
                return None

            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.SELL,
                strength=0.6,
                confidence=0.7,
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                stop_loss=float(st_line[-1]) if not np.isnan(st_line[-1]) else bar.close + self._atr_mult * curr_atr,
                take_profit=bar.close - self._atr_mult * curr_atr * self.genome.take_profit_ratio,
                trailing_stop_atr_mult=self._atr_mult,
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0] if "_" in self.genome.name else "BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
