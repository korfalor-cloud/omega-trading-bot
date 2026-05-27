"""Dollar-Cost Averaging (DCA) Strategy.

Systematically buys at regular intervals with optional value averaging.
Enhanced with market condition filters and dynamic sizing.
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


class DCAStrategy(Strategy):
    """Enhanced DCA with market-aware sizing.

    Base: Buy every N bars regardless of price.
    Enhancement: Increase buys when price is below SMA (value averaging),
    decrease or skip when price is extended above SMA.

    Parameters (from genome.features):
        dca_interval: Bars between purchases (default 24 — 1 day on 1h)
        base_size_pct: Base position size as % of equity (default 0.02)
        value_avg: Use value averaging (default True)
        rsi_threshold: Skip buys when RSI > threshold (default 70)
        max_buys: Max consecutive buys before pausing (default 20)
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._dca_interval = feats.get("dca_interval", 24)
        self._base_size_pct = feats.get("base_size_pct", 0.02)
        self._value_avg = feats.get("value_avg", True)
        self._rsi_threshold = feats.get("rsi_threshold", 70)
        self._max_buys = feats.get("max_buys", 20)
        self._bar_buffer: list[OHLCVBar] = []
        self._bars_since_last_buy = 0
        self._consecutive_buys = 0
        self._total_invested = 0.0

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        self._bars_since_last_buy += 1

        if len(self._bar_buffer) > 300:
            self._bar_buffer = self._bar_buffer[-200:]

        if len(self._bar_buffer) < 30:
            return None

        # Check interval
        if self._bars_since_last_buy < self._dca_interval:
            return None

        # Check max buys
        if self._consecutive_buys >= self._max_buys:
            # Reset after a pause
            if self._bars_since_last_buy >= self._dca_interval * 3:
                self._consecutive_buys = 0
            return None

        ti = TechnicalIndicators(self._bar_buffer)
        sma = ti.sma(50)
        rsi = ti.rsi(14)

        curr_sma = sma[-1]
        curr_rsi = rsi[-1]

        if curr_sma != curr_sma:
            return None

        # Skip if overbought
        if not np.isnan(curr_rsi) and curr_rsi > self._rsi_threshold:
            return None

        # Calculate size multiplier
        strength = 0.5  # Base strength
        if self._value_avg and curr_sma > 0:
            price_ratio = bar.close / curr_sma
            if price_ratio < 0.95:
                # Below SMA — buy more (value)
                strength = min(1.0, 0.5 + (1 - price_ratio) * 5)
            elif price_ratio > 1.05:
                # Above SMA — buy less
                strength = max(0.2, 0.5 - (price_ratio - 1) * 3)
            else:
                strength = 0.5

        self._bars_since_last_buy = 0
        self._consecutive_buys += 1

        return Signal(
            strategy_id=self.strategy_id,
            symbol=bar.symbol,
            side=Side.BUY,
            strength=strength,
            confidence=0.6,  # DCA has moderate confidence by design
            signal_type=SignalType.ENTRY,
            timeframe=Timeframe.H1,
            metadata={"dca_size_pct": self._base_size_pct * strength},
        )

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0] if "_" in self.genome.name else "BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
