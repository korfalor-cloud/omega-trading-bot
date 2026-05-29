"""Calendar Spread Strategy.

Implements:
- Futures calendar spread trading
- Term structure analysis
- Roll yield capture
- Spread mean reversion
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class CalendarSpreadStrategy(Strategy):
    """Calendar spread trading on futures."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._lookback = feats.get("lookback", 30)
        self._entry_zscore = feats.get("entry_zscore", 2.0)
        self._exit_zscore = feats.get("exit_zscore", 0.5)

        self._near_buffer: list[OHLCVBar] = []
        self._far_buffer: list[OHLCVBar] = []
        self._spread_history: list[float] = []
        self._in_trade = False
        self._trade_side = ""

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        # Route by symbol suffix
        if "_NEAR" in bar.symbol:
            self._near_buffer.append(bar)
        elif "_FAR" in bar.symbol:
            self._far_buffer.append(bar)
        else:
            return None

        if len(self._near_buffer) < self._lookback or len(self._far_buffer) < self._lookback:
            return None

        near_price = self._near_buffer[-1].close
        far_price = self._far_buffer[-1].close
        spread = far_price - near_price
        self._spread_history.append(spread)

        if len(self._spread_history) < self._lookback:
            return None

        spreads = np.array(self._spread_history[-self._lookback:])
        zscore = (spread - np.mean(spreads)) / np.std(spreads) if np.std(spreads) > 0 else 0

        # Exit
        if self._in_trade:
            if self._trade_side == "long_spread" and zscore > -self._exit_zscore:
                self._in_trade = False
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.SELL, strength=0.6, confidence=0.65,
                    signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                )
            if self._trade_side == "short_spread" and zscore < self._exit_zscore:
                self._in_trade = False
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.BUY, strength=0.6, confidence=0.65,
                    signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                )
            return None

        # Entry
        if zscore < -self._entry_zscore:
            self._in_trade = True
            self._trade_side = "long_spread"
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.BUY, strength=min(1.0, abs(zscore) / 3),
                confidence=0.65, signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1, metadata={"zscore": zscore},
            )

        if zscore > self._entry_zscore:
            self._in_trade = True
            self._trade_side = "short_spread"
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.SELL, strength=min(1.0, abs(zscore) / 3),
                confidence=0.65, signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1, metadata={"zscore": zscore},
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return ["BTC_NEAR", "BTC_FAR"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
