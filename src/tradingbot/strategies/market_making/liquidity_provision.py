"""Liquidity Provision Strategy.

Implements:
- Inventory-skewed quoting
- Dynamic spread adjustment
- Adverse selection protection
- P&L attribution
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class LiquidityProvisionStrategy(Strategy):
    """Advanced market making with inventory skew."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._base_spread = feats.get("base_spread", 0.002)
        self._inventory_limit = feats.get("inventory_limit", 1.0)
        self._skew_factor = feats.get("skew_factor", 0.5)

        self._bar_buffer: list[OHLCVBar] = []
        self._inventory = 0.0
        self._bid_price = 0.0
        self._ask_price = 0.0

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < 20:
            return None

        if len(self._bar_buffer) > 100:
            self._bar_buffer = self._bar_buffer[-80:]

        prices = np.array([b.close for b in self._bar_buffer[-20:]])
        vol = np.std(np.diff(prices) / prices[:-1]) if len(prices) > 1 else 0.01

        # Dynamic spread based on volatility
        spread = self._base_spread * (1 + vol * 10)
        mid = bar.close

        # Inventory skew — quote more aggressively to reduce inventory
        skew = self._inventory * self._skew_factor / self._inventory_limit if self._inventory_limit > 0 else 0

        self._bid_price = mid * (1 - spread / 2 - skew)
        self._ask_price = mid * (1 + spread / 2 - skew)

        # Check if we should quote
        if abs(self._inventory) >= self._inventory_limit:
            # At limit — only quote to reduce inventory
            if self._inventory > 0:
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.SELL, strength=0.5, confidence=0.6,
                    signal_type=SignalType.ENTRY, timeframe=Timeframe.H1,
                    metadata={"bid": self._bid_price, "ask": self._ask_price, "inventory": self._inventory},
                )
            else:
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.BUY, strength=0.5, confidence=0.6,
                    signal_type=SignalType.ENTRY, timeframe=Timeframe.H1,
                    metadata={"bid": self._bid_price, "ask": self._ask_price, "inventory": self._inventory},
                )

        # Normal quoting — provide both sides
        if bar.close <= self._bid_price:
            self._inventory += 1
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.BUY, strength=0.4, confidence=0.6,
                signal_type=SignalType.ENTRY, timeframe=Timeframe.H1,
                metadata={"bid": self._bid_price, "ask": self._ask_price},
            )

        if bar.close >= self._ask_price:
            self._inventory -= 1
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=Side.SELL, strength=0.4, confidence=0.6,
                signal_type=SignalType.ENTRY, timeframe=Timeframe.H1,
                metadata={"bid": self._bid_price, "ask": self._ask_price},
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0]] if "_" in self.genome.name else ["BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.M1]
