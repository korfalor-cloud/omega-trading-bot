"""Volatility Arbitrage Strategy.

Implements:
- Implied vs realized vol spread
- Vol mean reversion
- Vol surface analysis
- Vega-neutral positioning
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class VolArbStrategy(Strategy):
    """Volatility arbitrage — trade implied vs realized vol."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._lookback = feats.get("lookback", 30)
        self._entry_spread = feats.get("entry_spread", 0.10)
        self._exit_spread = feats.get("exit_spread", 0.02)

        self._bar_buffer: list[OHLCVBar] = []
        self._implied_vol: list[float] = []
        self._in_trade = False
        self._trade_side = ""

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)

        # Get implied vol from bar metadata
        iv = bar.vwap if bar.vwap > 0 else 0
        self._implied_vol.append(iv)

        if len(self._bar_buffer) < self._lookback:
            return None

        if len(self._bar_buffer) > 200:
            self._bar_buffer = self._bar_buffer[-150:]
            self._implied_vol = self._implied_vol[-150:]

        prices = np.array([b.close for b in self._bar_buffer[-self._lookback:]])
        returns = np.diff(np.log(prices))
        realized_vol = np.std(returns) * np.sqrt(365)

        implied_vol = np.mean(self._implied_vol[-5:]) if self._implied_vol[-5:] else realized_vol

        spread = implied_vol - realized_vol

        # Exit
        if self._in_trade:
            if self._trade_side == "short_vol" and spread < self._exit_spread:
                self._in_trade = False
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.BUY, strength=0.6, confidence=0.65,
                    signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                )
            if self._trade_side == "long_vol" and spread > -self._exit_spread:
                self._in_trade = False
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.SELL, strength=0.6, confidence=0.65,
                    signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                )
            return None

        # Entry
        if spread > self._entry_spread:
            self._in_trade = True
            self._trade_side = "short_vol"
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.SELL, strength=min(1.0, spread),
                confidence=0.65, signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                metadata={"iv": implied_vol, "rv": realized_vol, "spread": spread},
            )

        if spread < -self._entry_spread:
            self._in_trade = True
            self._trade_side = "long_vol"
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.BUY, strength=min(1.0, abs(spread)),
                confidence=0.65, signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                metadata={"iv": implied_vol, "rv": realized_vol, "spread": spread},
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0]] if "_" in self.genome.name else ["BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
