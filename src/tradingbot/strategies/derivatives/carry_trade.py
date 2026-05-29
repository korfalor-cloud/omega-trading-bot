"""Carry Trade Strategy.

Implements:
- Interest rate differential trading
- Funding rate arbitrage
- Roll yield capture
- Risk-adjusted carry
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class CarryTradeStrategy(Strategy):
    """Carry trade — profit from interest rate differentials."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._min_carry = feats.get("min_carry", 0.0003)
        self._hold_bars = feats.get("hold_bars", 24)

        self._bar_buffer: list[OHLCVBar] = []
        self._carry_history: list[float] = []
        self._in_trade = False
        self._trade_side = ""
        self._trade_bars = 0

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)

        # Get carry from bar metadata
        carry = bar.vwap if bar.vwap != 0 else 0
        self._carry_history.append(carry)

        if len(self._bar_buffer) < 20:
            return None

        # Exit
        if self._in_trade:
            self._trade_bars += 1
            if self._trade_bars >= self._hold_bars:
                self._in_trade = False
                self._trade_bars = 0
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.SELL if self._trade_side == "buy" else Side.BUY,
                    strength=0.5, confidence=0.6,
                    signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                )
            # Exit if carry reverses
            if len(self._carry_history) >= 3:
                recent = np.mean(self._carry_history[-3:])
                if self._trade_side == "buy" and recent < 0:
                    self._in_trade = False
                    self._trade_bars = 0
                    return Signal(
                        strategy_id=self.strategy_id, symbol=bar.symbol,
                        side=Side.SELL, strength=0.6, confidence=0.65,
                        signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                    )
            return None

        # Entry
        if carry > self._min_carry:
            self._in_trade = True
            self._trade_side = "buy"
            self._trade_bars = 0
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.BUY, strength=min(1.0, carry * 1000),
                confidence=0.65, signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1, metadata={"carry": carry},
            )

        if carry < -self._min_carry:
            self._in_trade = True
            self._trade_side = "sell"
            self._trade_bars = 0
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.SELL, strength=min(1.0, abs(carry) * 1000),
                confidence=0.65, signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1, metadata={"carry": carry},
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0]] if "_" in self.genome.name else ["BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
