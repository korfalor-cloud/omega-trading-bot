"""Grid Trading Strategy.

Places buy and sell orders at predefined price levels (the "grid").
Profits from price oscillation within a range.

Works best in sideways/range-bound markets.
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


class GridTradingStrategy(Strategy):
    """Grid trading with dynamic level adjustment.

    Places limit-like entries at fixed intervals around a moving average.
    Buys on dips, sells on rallies within the grid range.

    Parameters (from genome.features):
        grid_levels: Number of grid levels (default 10)
        grid_spacing_pct: Distance between levels as % of price (default 0.5)
        base_period: Period for the center moving average (default 20)
        atr_filter: Use ATR to detect ranging market (default True)
        adx_max: Max ADX for range detection (default 25)
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._grid_levels = feats.get("grid_levels", 10)
        self._grid_spacing_pct = feats.get("grid_spacing_pct", 0.005)
        self._base_period = feats.get("base_period", 20)
        self._adx_max = feats.get("adx_max", 25)
        self._atr_mult = genome.stop_loss_param
        self._bar_buffer: list[OHLCVBar] = []
        self._min_bars = self._base_period + 10
        self._active_grid_center: float = 0.0
        self._last_signal_side: Side | None = None

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._min_bars:
            return None

        if len(self._bar_buffer) > 300:
            self._bar_buffer = self._bar_buffer[-200:]

        ti = TechnicalIndicators(self._bar_buffer)
        sma = ti.sma(self._base_period)
        adx = ti.adx(14)
        atr = ti.atr(14)

        curr_sma = sma[-1]
        curr_adx = adx[-1]
        curr_atr = atr[-1]

        if any(x != x for x in [curr_sma, curr_adx]):
            return None

        if curr_atr is None or curr_atr != curr_atr:
            curr_atr = bar.close * 0.02

        # Only trade in range-bound markets (low ADX)
        if curr_adx > self._adx_max:
            return None

        # Update grid center
        if self._active_grid_center == 0:
            self._active_grid_center = curr_sma

        # Calculate grid levels
        spacing = self._active_grid_center * self._grid_spacing_pct
        half_grid = self._grid_levels * spacing / 2
        grid_low = self._active_grid_center - half_grid
        grid_high = self._active_grid_center + half_grid

        # Check if price is near a grid level
        price = bar.close
        if price <= grid_low:
            # At bottom of grid — buy
            distance_from_center = (self._active_grid_center - price) / self._active_grid_center
            strength = min(1.0, distance_from_center / (self._grid_spacing_pct * self._grid_levels / 2))
            self._last_signal_side = Side.BUY
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,
                strength=strength,
                confidence=min(1.0, 0.5 + strength * 0.5),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                stop_loss=price - self._atr_mult * curr_atr,
                take_profit=self._active_grid_center,
            )

        elif price >= grid_high:
            # At top of grid — sell
            distance_from_center = (price - self._active_grid_center) / self._active_grid_center
            strength = min(1.0, distance_from_center / (self._grid_spacing_pct * self._grid_levels / 2))
            self._last_signal_side = Side.SELL
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.SELL,
                strength=strength,
                confidence=min(1.0, 0.5 + strength * 0.5),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                stop_loss=price + self._atr_mult * curr_atr,
                take_profit=self._active_grid_center,
            )

        # Adjust grid center slowly toward current SMA
        self._active_grid_center = 0.95 * self._active_grid_center + 0.05 * curr_sma
        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0] if "_" in self.genome.name else "BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
