"""Candlestick Pattern Strategy — Engulfing, Hammer, Doji.

Implements:
- Bullish/Bearish engulfing pattern detection
- Hammer/Shooting star detection
- Doji detection
- Volume confirmation
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class CandlestickStrategy(Strategy):
    """Candlestick pattern-based strategy."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._min_body_ratio = feats.get("min_body_ratio", 0.6)
        self._volume_confirm = feats.get("volume_confirm", True)
        self._hold_bars = feats.get("hold_bars", 5)

        self._bar_buffer: list[OHLCVBar] = []
        self._in_trade = False
        self._trade_side = ""
        self._trade_bars = 0

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < 5:
            return None

        if len(self._bar_buffer) > 100:
            self._bar_buffer = self._bar_buffer[-80:]

        # Exit logic
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
            return None

        prev = self._bar_buffer[-2]
        curr = bar

        # Volume check
        vol_avg = np.mean([b.volume for b in self._bar_buffer[-20:]]) if len(self._bar_buffer) >= 20 else curr.volume
        vol_ok = curr.volume > vol_avg if self._volume_confirm else True

        # Engulfing patterns
        prev_body = prev.close - prev.open
        curr_body = curr.close - curr.open

        # Bullish engulfing
        if prev_body < -10 and curr_body > 10:
            if curr.open <= prev.close and curr.close >= prev.open:
                if abs(curr_body) > abs(prev_body) * self._min_body_ratio and vol_ok:
                    self._in_trade = True
                    self._trade_side = "buy"
                    self._trade_bars = 0
                    return Signal(
                        strategy_id=self.strategy_id, symbol=bar.symbol,
                        side=Side.BUY, strength=0.7, confidence=0.65,
                        signal_type=SignalType.ENTRY, timeframe=Timeframe.H1,
                        metadata={"pattern": "bullish_engulfing"},
                    )

        # Bearish engulfing
        if prev_body > 10 and curr_body < -10:
            if curr.open >= prev.close and curr.close <= prev.open:
                if abs(curr_body) > abs(prev_body) * self._min_body_ratio and vol_ok:
                    self._in_trade = True
                    self._trade_side = "sell"
                    self._trade_bars = 0
                    return Signal(
                        strategy_id=self.strategy_id, symbol=bar.symbol,
                        side=Side.SELL, strength=0.7, confidence=0.65,
                        signal_type=SignalType.ENTRY, timeframe=Timeframe.H1,
                        metadata={"pattern": "bearish_engulfing"},
                    )

        # Hammer (bullish)
        if self._is_hammer(curr) and vol_ok:
            self._in_trade = True
            self._trade_side = "buy"
            self._trade_bars = 0
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.BUY, strength=0.6, confidence=0.6,
                signal_type=SignalType.ENTRY, timeframe=Timeframe.H1,
                metadata={"pattern": "hammer"},
            )

        # Shooting star (bearish)
        if self._is_shooting_star(curr) and vol_ok:
            self._in_trade = True
            self._trade_side = "sell"
            self._trade_bars = 0
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.SELL, strength=0.6, confidence=0.6,
                signal_type=SignalType.ENTRY, timeframe=Timeframe.H1,
                metadata={"pattern": "shooting_star"},
            )

        return None

    def _is_hammer(self, bar: OHLCVBar) -> bool:
        """Hammer: small body at top, long lower shadow."""
        body = abs(bar.close - bar.open)
        total_range = bar.high - bar.low
        lower_shadow = min(bar.open, bar.close) - bar.low

        if total_range == 0 or body < 5:
            return False
        return lower_shadow > body * 2 and body / total_range < 0.3

    def _is_shooting_star(self, bar: OHLCVBar) -> bool:
        """Shooting star: small body at bottom, long upper shadow."""
        body = abs(bar.close - bar.open)
        total_range = bar.high - bar.low
        upper_shadow = bar.high - max(bar.open, bar.close)

        if total_range == 0 or body < 5:
            return False
        return upper_shadow > body * 2 and body / total_range < 0.3

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0]] if "_" in self.genome.name else ["BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
