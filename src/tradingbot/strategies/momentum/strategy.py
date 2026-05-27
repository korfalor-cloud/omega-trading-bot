"""Momentum Strategy.

Rides strong price moves confirmed by multiple momentum indicators.
Uses ROC, RSI, and volume to identify accelerating trends.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy
from ...features.technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class MomentumStrategy(Strategy):
    """Multi-factor momentum strategy.

    Entry conditions (BUY):
    - ROC > threshold (price accelerating)
    - RSI in momentum zone (50-70 for buy, 30-50 for sell)
    - Volume above average
    - Price above short EMA

    Exit: Momentum exhaustion (RSI divergence or ROC reversal).

    Parameters (from genome.features):
        roc_period: Rate of change period (default 10)
        roc_threshold: Min ROC for entry (default 0.02 = 2%)
        rsi_period: RSI period (default 14)
        volume_mult: Volume must be N * average (default 1.3)
        ema_period: Trend confirmation EMA (default 21)
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._roc_period = feats.get("roc_period", 10)
        self._roc_threshold = feats.get("roc_threshold", 0.02)
        self._rsi_period = feats.get("rsi_period", 14)
        self._volume_mult = feats.get("volume_mult", 1.3)
        self._ema_period = feats.get("ema_period", 21)
        self._atr_mult = genome.stop_loss_param
        self._bar_buffer: list[OHLCVBar] = []
        self._min_bars = max(self._roc_period, self._rsi_period, self._ema_period) + 10

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._min_bars:
            return None

        if len(self._bar_buffer) > 300:
            self._bar_buffer = self._bar_buffer[-200:]

        ti = TechnicalIndicators(self._bar_buffer)
        roc = ti.roc(self._roc_period)
        rsi = ti.rsi(self._rsi_period)
        ema = ti.ema(self._ema_period)
        atr = ti.atr(14)

        curr_roc = roc[-1]
        prev_roc = roc[-2] if len(roc) > 1 else 0
        curr_rsi = rsi[-1]
        curr_ema = ema[-1]
        curr_atr = atr[-1]

        if any(x != x for x in [curr_roc, curr_rsi, curr_ema]):
            return None

        if curr_atr is None or curr_atr != curr_atr:
            curr_atr = bar.close * 0.02

        # Volume filter
        vol_avg = np.mean([b.volume for b in self._bar_buffer[-20:]])
        if vol_avg > 0 and bar.volume < vol_avg * self._volume_mult:
            return None

        # Bullish momentum
        if (curr_roc > self._roc_threshold
                and 50 < curr_rsi < 70
                and bar.close > curr_ema):
            # Strength proportional to ROC magnitude
            strength = min(1.0, curr_roc / (self._roc_threshold * 3))
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,
                strength=strength,
                confidence=min(1.0, 0.5 + strength * 0.3),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                stop_loss=bar.close - self._atr_mult * curr_atr,
                take_profit=bar.close + self._atr_mult * curr_atr * self.genome.take_profit_ratio,
                trailing_stop_atr_mult=self._atr_mult,
            )

        # Bearish momentum
        if (curr_roc < -self._roc_threshold
                and 30 < curr_rsi < 50
                and bar.close < curr_ema):
            strength = min(1.0, abs(curr_roc) / (self._roc_threshold * 3))
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.SELL,
                strength=strength,
                confidence=min(1.0, 0.5 + strength * 0.3),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                stop_loss=bar.close + self._atr_mult * curr_atr,
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
