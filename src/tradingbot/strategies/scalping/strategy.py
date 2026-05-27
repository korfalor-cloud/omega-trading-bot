"""Scalping Strategy.

High-frequency short-term trades targeting small price moves.
Uses order book imbalance, short-term momentum, and tight stops.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, OrderBookSnapshot, Signal, StrategyGenome
from ...core.interfaces import Strategy
from ...features.technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class ScalpingStrategy(Strategy):
    """Order flow and microstructure scalping.

    Entry conditions:
    - Short-term EMA cross (3/8)
    - RSI extreme (oversold for buy, overbought for sell)
    - Price near Bollinger Band edge
    - Optional: order book imbalance confirmation

    Parameters (from genome.features):
        fast_ema: Fast EMA period (default 3)
        slow_ema: Slow EMA period (default 8)
        rsi_period: RSI period (default 7)
        rsi_oversold: RSI oversold level (default 25)
        rsi_overbought: RSI overbought level (default 75)
        target_bps: Target profit in bps (default 15)
        stop_bps: Stop loss in bps (default 30)
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._fast_ema = feats.get("fast_ema", 3)
        self._slow_ema = feats.get("slow_ema", 8)
        self._rsi_period = feats.get("rsi_period", 7)
        self._rsi_oversold = feats.get("rsi_oversold", 25)
        self._rsi_overbought = feats.get("rsi_overbought", 75)
        self._target_bps = feats.get("target_bps", 15)
        self._stop_bps = feats.get("stop_bps", 30)
        self._bar_buffer: list[OHLCVBar] = []
        self._book: OrderBookSnapshot | None = None
        self._min_bars = max(self._slow_ema, self._rsi_period) + 5

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._min_bars:
            return None

        if len(self._bar_buffer) > 100:
            self._bar_buffer = self._bar_buffer[-50:]

        ti = TechnicalIndicators(self._bar_buffer)
        fast = ti.ema(self._fast_ema)
        slow = ti.ema(self._slow_ema)
        rsi = ti.rsi(self._rsi_period)
        bb_upper, bb_mid, bb_lower = ti.bollinger_bands(20, 2.0)

        curr_fast = fast[-1]
        prev_fast = fast[-2] if len(fast) > 1 else 0
        curr_slow = slow[-1]
        prev_slow = slow[-2] if len(slow) > 1 else 0
        curr_rsi = rsi[-1]
        curr_bb_lower = bb_lower[-1]
        curr_bb_upper = bb_upper[-1]

        if any(x != x for x in [curr_fast, curr_slow, curr_rsi]):
            return None

        price = bar.close

        # Target and stop in price terms
        target_move = price * self._target_bps / 10000
        stop_move = price * self._stop_bps / 10000

        # Bullish scalp: fast EMA crosses above slow + RSI oversold + near lower BB
        if (prev_fast <= prev_slow and curr_fast > curr_slow
                and curr_rsi < self._rsi_oversold
                and not np.isnan(curr_bb_lower) and price <= curr_bb_lower * 1.005):

            confidence = 0.5
            # Order book confirmation
            if self._book and self._book.imbalance is not None:
                if self._book.imbalance > 0.2:
                    confidence += 0.2

            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,
                strength=0.7,
                confidence=min(1.0, confidence),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.M5,
                stop_loss=price - stop_move,
                take_profit=price + target_move,
            )

        # Bearish scalp
        if (prev_fast >= prev_slow and curr_fast < curr_slow
                and curr_rsi > self._rsi_overbought
                and not np.isnan(curr_bb_upper) and price >= curr_bb_upper * 0.995):

            confidence = 0.5
            if self._book and self._book.imbalance is not None:
                if self._book.imbalance < -0.2:
                    confidence += 0.2

            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.SELL,
                strength=0.7,
                confidence=min(1.0, confidence),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.M5,
                stop_loss=price + stop_move,
                take_profit=price - target_move,
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    async def on_order_book(self, book: OrderBookSnapshot) -> Optional[Signal]:
        self._book = book
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0] if "_" in self.genome.name else "BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.M5]
