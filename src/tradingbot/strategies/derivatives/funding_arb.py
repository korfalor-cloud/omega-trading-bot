"""Funding Rate Arbitrage Strategy.

Exploits funding rate differentials between perpetual futures and spot.
When funding is positive (longs pay shorts), short perp + long spot.
When funding is negative (shorts pay longs), long perp + short spot.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class FundingRateArbitrage(Strategy):
    """Funding rate arbitrage strategy.

    Monitors funding rates and executes basis trades:
    - Positive funding > threshold: Short perp, long spot (collect funding)
    - Negative funding < -threshold: Long perp, short spot (collect funding)

    Parameters (from genome.features):
        funding_threshold: Min funding rate to trade (default 0.0003 = 0.03%)
        max_hold_bars: Max bars to hold position (default 72 — 3 days on 1h)
        use_trend_filter: Only trade when trend is neutral (default True)
        adx_max: Max ADX for trend filter (default 30)
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._funding_threshold = feats.get("funding_threshold", 0.0003)
        self._max_hold_bars = feats.get("max_hold_bars", 72)
        self._use_trend_filter = feats.get("use_trend_filter", True)
        self._adx_max = feats.get("adx_max", 30)
        self._bar_buffer: list[OHLCVBar] = []
        self._min_bars = 30
        self._in_trade = False
        self._trade_bars = 0
        self._funding_history: list[float] = []

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._min_bars:
            return None

        if len(self._bar_buffer) > 300:
            self._bar_buffer = self._bar_buffer[-200:]

        # Get funding rate from metadata (injected by data pipeline)
        funding_rate = bar.vwap if bar.vwap != 0 else 0
        self._funding_history.append(funding_rate)

        # Exit signal
        if self._in_trade:
            self._trade_bars += 1
            if self._trade_bars >= self._max_hold_bars:
                self._in_trade = False
                self._trade_bars = 0
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=Side.SELL,
                    strength=0.5,
                    confidence=0.6,
                    signal_type=SignalType.EXIT,
                    timeframe=Timeframe.H1,
                )
            # Exit if funding reverts
            if len(self._funding_history) > 3:
                recent_avg = np.mean(self._funding_history[-3:])
                if abs(recent_avg) < self._funding_threshold / 2:
                    self._in_trade = False
                    self._trade_bars = 0
                    return Signal(
                        strategy_id=self.strategy_id,
                        symbol=bar.symbol,
                        side=Side.SELL,
                        strength=0.5,
                        confidence=0.6,
                        signal_type=SignalType.EXIT,
                        timeframe=Timeframe.H1,
                    )
            return None

        # Trend filter
        if self._use_trend_filter:
            from ...features.technical import TechnicalIndicators
            ti = TechnicalIndicators(self._bar_buffer)
            adx = ti.adx(14)
            curr_adx = adx[-1]
            if curr_adx != curr_adx or curr_adx > self._adx_max:
                return None

        # Need sustained funding rate
        if len(self._funding_history) < 3:
            return None

        recent_funding = np.mean(self._funding_history[-3:])

        # Positive funding: short perp + long spot
        if recent_funding > self._funding_threshold:
            self._in_trade = True
            self._trade_bars = 0
            annual_return = recent_funding * 3 * 365  # 8h funding * 3 * 365
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.SELL,  # Short perp
                strength=min(1.0, recent_funding / self._funding_threshold),
                confidence=min(1.0, 0.5 + recent_funding / self._funding_threshold * 0.3),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                metadata={
                    "funding_rate": recent_funding,
                    "annual_return_est": annual_return,
                    "hedge_side": "long_spot",
                },
            )

        # Negative funding: long perp + short spot
        if recent_funding < -self._funding_threshold:
            self._in_trade = True
            self._trade_bars = 0
            annual_return = abs(recent_funding) * 3 * 365
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,  # Long perp
                strength=min(1.0, abs(recent_funding) / self._funding_threshold),
                confidence=min(1.0, 0.5 + abs(recent_funding) / self._funding_threshold * 0.3),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                metadata={
                    "funding_rate": recent_funding,
                    "annual_return_est": annual_return,
                    "hedge_side": "short_spot",
                },
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0] if "_" in self.genome.name else "BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
