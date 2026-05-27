"""Breakout Strategy.

Trades breakouts from consolidation ranges with volume confirmation.
Uses Donchian channels and ATR for entry/exit levels.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy
from ...features.technical import TechnicalIndicators
from ...features.advanced_indicators import AdvancedIndicators

logger = logging.getLogger(__name__)


class BreakoutStrategy(Strategy):
    """Donchian Channel breakout with volume confirmation.

    Entry conditions:
    - Price breaks above Donchian upper (bullish) or below lower (bearish)
    - Volume spike confirms breakout
    - ADX shows trend is forming
    - Consolidation period preceded the breakout

    Parameters (from genome.features):
        donchian_period: Donchian channel lookback (default 20)
        volume_confirm_mult: Volume must be N * average (default 1.5)
        adx_min: Minimum ADX for breakout confirmation (default 20)
        consolidation_bars: Bars of consolidation before breakout (default 10)
        consolidation_range_pct: Max range for consolidation (default 0.03)
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._donchian_period = feats.get("donchian_period", 20)
        self._volume_confirm_mult = feats.get("volume_confirm_mult", 1.5)
        self._adx_min = feats.get("adx_min", 20)
        self._consolidation_bars = feats.get("consolidation_bars", 10)
        self._consolidation_range_pct = feats.get("consolidation_range_pct", 0.03)
        self._atr_mult = genome.stop_loss_param
        self._bar_buffer: list[OHLCVBar] = []
        self._min_bars = self._donchian_period + self._consolidation_bars + 5

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._min_bars:
            return None

        if len(self._bar_buffer) > 300:
            self._bar_buffer = self._bar_buffer[-200:]

        ti = TechnicalIndicators(self._bar_buffer)
        ai = AdvancedIndicators(self._bar_buffer)

        upper, middle, lower = ai.donchian_channel(self._donchian_period)
        adx = ti.adx(14)
        atr = ti.atr(14)

        curr_upper = upper[-1]
        curr_lower = lower[-1]
        curr_adx = adx[-1]
        curr_atr = atr[-1]
        prev_close = self._bar_buffer[-2].close

        if any(x != x for x in [curr_upper, curr_lower, curr_adx]):
            return None

        if curr_atr is None or curr_atr != curr_atr:
            curr_atr = bar.close * 0.02

        # Check consolidation: range should be tight before breakout
        recent_bars = self._bar_buffer[-self._consolidation_bars:]
        recent_high = max(b.high for b in recent_bars)
        recent_low = min(b.low for b in recent_bars)
        if recent_low > 0:
            range_pct = (recent_high - recent_low) / recent_low
        else:
            range_pct = 0

        is_consolidated = range_pct < self._consolidation_range_pct

        # Volume confirmation
        vol_avg = np.mean([b.volume for b in self._bar_buffer[-20:]])
        vol_confirmed = vol_avg > 0 and bar.volume > vol_avg * self._volume_confirm_mult

        # Bullish breakout
        if (bar.close > curr_upper
                and prev_close <= curr_upper
                and vol_confirmed
                and (curr_adx > self._adx_min or is_consolidated)):
            strength = min(1.0, (bar.close - curr_upper) / curr_atr) if curr_atr > 0 else 0.5
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,
                strength=max(0.3, strength),
                confidence=min(1.0, 0.6 + (0.4 if vol_confirmed else 0)),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                stop_loss=curr_upper - self._atr_mult * curr_atr,
                take_profit=bar.close + self._atr_mult * curr_atr * self.genome.take_profit_ratio,
                trailing_stop_atr_mult=self._atr_mult,
            )

        # Bearish breakout
        if (bar.close < curr_lower
                and prev_close >= curr_lower
                and vol_confirmed
                and (curr_adx > self._adx_min or is_consolidated)):
            strength = min(1.0, (curr_lower - bar.close) / curr_atr) if curr_atr > 0 else 0.5
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.SELL,
                strength=max(0.3, strength),
                confidence=min(1.0, 0.6 + (0.4 if vol_confirmed else 0)),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                stop_loss=curr_lower + self._atr_mult * curr_atr,
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
