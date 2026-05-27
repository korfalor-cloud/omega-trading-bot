"""Triangular Arbitrage Strategy.

Exploits price discrepancies between three currency pairs.
E.g., BTC/USDT → ETH/BTC → ETH/USDT should equal 1.0 but may deviate.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class TriangularArbitrageStrategy(Strategy):
    """Triangular arbitrage detector.

    Monitors three pairs forming a triangle and detects when the
    implied cross-rate deviates from the actual rate.

    Example triangle:
        BTC/USDT → ETH/BTC → ETH/USDT
        If (BTC/USDT * ETH/BTC) / ETH/USDT > 1 + threshold, arbitrage exists.

    Parameters (from genome.features):
        pair_a: First pair (default "BTC/USDT")
        pair_b: Second pair (default "ETH/BTC")
        pair_c: Third pair (default "ETH/USDT")
        threshold_bps: Min deviation for entry in bps (default 5)
        max_hold_bars: Max bars to hold (default 3)
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._pair_a = feats.get("pair_a", "BTC/USDT")
        self._pair_b = feats.get("pair_b", "ETH/BTC")
        self._pair_c = feats.get("pair_c", "ETH/USDT")
        self._threshold_bps = feats.get("threshold_bps", 5)
        self._max_hold_bars = feats.get("max_hold_bars", 3)

        self._prices: dict[str, float] = {}
        self._bars_in_trade = 0
        self._in_trade = False

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._prices[bar.symbol] = bar.close

        # Need all three prices
        if not all(p in self._prices for p in [self._pair_a, self._pair_b, self._pair_c]):
            return None

        price_a = self._prices[self._pair_a]
        price_b = self._prices[self._pair_b]
        price_c = self._prices[self._pair_c]

        if any(p <= 0 for p in [price_a, price_b, price_c]):
            return None

        # Exit signal
        if self._in_trade:
            self._bars_in_trade += 1
            if self._bars_in_trade >= self._max_hold_bars:
                self._in_trade = False
                self._bars_in_trade = 0
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=Side.SELL,  # Exit
                    strength=0.5,
                    confidence=0.7,
                    signal_type=SignalType.EXIT,
                    timeframe=Timeframe.M1,
                )
            return None

        # Compute triangular rate
        # Forward: buy A with USDT, buy B with A, sell B for USDT
        forward_rate = price_a * price_b / price_c
        # Backward: sell A for USDT, buy B with USDT, sell B for A
        backward_rate = price_c / (price_a * price_b)

        forward_deviation_bps = (forward_rate - 1) * 10000
        backward_deviation_bps = (1 - backward_rate) * 10000

        # Forward arbitrage: buy pair_c, sell pair_a * pair_b
        if forward_deviation_bps > self._threshold_bps:
            self._in_trade = True
            self._bars_in_trade = 0
            return Signal(
                strategy_id=self.strategy_id,
                symbol=self._pair_c,
                side=Side.BUY,
                strength=min(1.0, forward_deviation_bps / (self._threshold_bps * 5)),
                confidence=min(1.0, forward_deviation_bps / 20),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.M1,
                metadata={
                    "arb_type": "forward",
                    "deviation_bps": forward_deviation_bps,
                    "triangle": [self._pair_a, self._pair_b, self._pair_c],
                },
            )

        # Backward arbitrage
        if backward_deviation_bps > self._threshold_bps:
            self._in_trade = True
            self._bars_in_trade = 0
            return Signal(
                strategy_id=self.strategy_id,
                symbol=self._pair_c,
                side=Side.SELL,
                strength=min(1.0, backward_deviation_bps / (self._threshold_bps * 5)),
                confidence=min(1.0, backward_deviation_bps / 20),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.M1,
                metadata={
                    "arb_type": "backward",
                    "deviation_bps": backward_deviation_bps,
                    "triangle": [self._pair_a, self._pair_b, self._pair_c],
                },
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self._pair_a, self._pair_b, self._pair_c]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.M1]
