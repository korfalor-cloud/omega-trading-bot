"""Mean Reversion Strategy — Trade bounces from Bollinger Bands.

Buys when price touches lower band with RSI oversold confirmation.
Sells when price touches upper band with RSI overbought confirmation.
"""
from __future__ import annotations

import logging
from typing import Optional

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy
from ...features.technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class BollingerMeanReversion(Strategy):
    """Mean reversion using Bollinger Bands + RSI.

    Entry conditions:
    - BUY: Price < lower BB AND RSI < oversold AND volume > avg
    - SELL: Price > upper BB AND RSI > overbought AND volume > avg

    Uses ATR for stop placement and BB midline as target.
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        self._bb_period = 20
        self._bb_std = 2.0
        self._rsi_period = 14
        self._rsi_oversold = 30
        self._rsi_overbought = 70
        self._volume_mult = 1.2  # Volume must be 1.2x average
        self._atr_mult = genome.stop_loss_param
        self._bar_buffer: list[OHLCVBar] = []
        self._min_bars = max(self._bb_period, self._rsi_period, 30) + 5

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._min_bars:
            return None

        if len(self._bar_buffer) > 500:
            self._bar_buffer = self._bar_buffer[-300:]

        ti = TechnicalIndicators(self._bar_buffer)
        bb_upper, bb_mid, bb_lower = ti.bollinger_bands(self._bb_period, self._bb_std)
        rsi = ti.rsi(self._rsi_period)
        atr = ti.atr(14)

        curr_price = bar.close
        curr_upper = bb_upper[-1]
        curr_lower = bb_lower[-1]
        curr_mid = bb_mid[-1]
        curr_rsi = rsi[-1]
        curr_atr = atr[-1]

        if any(x != x for x in [curr_upper, curr_lower, curr_rsi]):
            return None

        if curr_atr is None or curr_atr != curr_atr:
            curr_atr = bar.close * 0.02

        # Volume filter
        vol_avg = sum(b.volume for b in self._bar_buffer[-20:]) / 20
        if bar.volume < vol_avg * self._volume_mult:
            return None

        # Buy: price at lower band + RSI oversold
        if curr_price <= curr_lower and curr_rsi < self._rsi_oversold:
            distance = (curr_mid - curr_price) / curr_price
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,
                strength=min(1.0, distance * 20),
                confidence=min(1.0, (self._rsi_oversold - curr_rsi) / 30),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                stop_loss=curr_price - self._atr_mult * curr_atr,
                take_profit=curr_mid,
            )

        # Sell: price at upper band + RSI overbought
        if curr_price >= curr_upper and curr_rsi > self._rsi_overbought:
            distance = (curr_price - curr_mid) / curr_price
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.SELL,
                strength=min(1.0, distance * 20),
                confidence=min(1.0, (curr_rsi - self._rsi_overbought) / 30),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                stop_loss=curr_price + self._atr_mult * curr_atr,
                take_profit=curr_mid,
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return ["BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
