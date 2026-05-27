"""Market Making Strategy.

Provides liquidity by placing limit orders on both sides of the book.
Profits from the bid-ask spread while managing inventory risk.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import OrderType, Side, SignalType, Timeframe
from ...core.types import OHLCVBar, OrderBookSnapshot, Signal, StrategyGenome
from ...core.interfaces import Strategy
from ...features.technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class MarketMakingStrategy(Strategy):
    """Market making with inventory management.

    Places symmetric quotes around a fair value estimate.
    Adjusts spread based on volatility and inventory skew.

    Parameters (from genome.features):
        base_spread_bps: Base spread in bps (default 10)
        max_inventory: Max position size (default 1.0)
        volatility_mult: Widen spread by N * volatility (default 2.0)
        skew_factor: Inventory skew adjustment (default 0.5)
        fair_value_period: Period for fair value EMA (default 20)
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._base_spread_bps = feats.get("base_spread_bps", 10)
        self._max_inventory = feats.get("max_inventory", 1.0)
        self._volatility_mult = feats.get("volatility_mult", 2.0)
        self._skew_factor = feats.get("skew_factor", 0.5)
        self._fair_value_period = feats.get("fair_value_period", 20)
        self._bar_buffer: list[OHLCVBar] = []
        self._current_inventory = 0.0
        self._min_bars = self._fair_value_period + 5

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._min_bars:
            return None

        if len(self._bar_buffer) > 200:
            self._bar_buffer = self._bar_buffer[-100:]

        ti = TechnicalIndicators(self._bar_buffer)
        ema = ti.ema(self._fair_value_period)
        atr = ti.atr(14)

        fair_value = ema[-1]
        curr_atr = atr[-1]

        if fair_value != fair_value:
            return None
        if curr_atr is None or curr_atr != curr_atr:
            curr_atr = bar.close * 0.01

        # Volatility-adjusted spread
        vol_pct = curr_atr / bar.close if bar.close > 0 else 0.01
        spread_pct = self._base_spread_bps / 10000 + self._volatility_mult * vol_pct

        # Inventory skew: shift quotes to encourage mean-reversion
        inventory_ratio = self._current_inventory / self._max_inventory if self._max_inventory > 0 else 0
        skew = self._skew_factor * inventory_ratio * spread_pct

        # Bid and ask prices
        bid_price = fair_value * (1 - spread_pct / 2 - skew)
        ask_price = fair_value * (1 + spread_pct / 2 - skew)

        # Determine action based on inventory
        if abs(self._current_inventory) >= self._max_inventory:
            # At inventory limit — only quote the reducing side
            if self._current_inventory > 0:
                # Long inventory — only sell
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=Side.SELL,
                    strength=0.5,
                    confidence=0.8,
                    signal_type=SignalType.ENTRY,
                    timeframe=Timeframe.M1,
                    target_price=ask_price,
                    metadata={"order_type": OrderType.LIMIT.value, "inventory": self._current_inventory},
                )
            else:
                # Short inventory — only buy
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=Side.BUY,
                    strength=0.5,
                    confidence=0.8,
                    signal_type=SignalType.ENTRY,
                    timeframe=Timeframe.M1,
                    target_price=bid_price,
                    metadata={"order_type": OrderType.LIMIT.value, "inventory": self._current_inventory},
                )

        # Normal quoting — return the side with better edge
        # Use price relative to fair value to decide
        price_vs_fair = (bar.close - fair_value) / fair_value if fair_value > 0 else 0

        if price_vs_fair < -spread_pct / 4:
            # Price below fair — opportunity to buy
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,
                strength=0.4,
                confidence=0.6,
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.M1,
                target_price=bid_price,
                stop_loss=bid_price * (1 - spread_pct * 3),
                metadata={"order_type": OrderType.LIMIT.value, "spread_bps": spread_pct * 10000},
            )
        elif price_vs_fair > spread_pct / 4:
            # Price above fair — opportunity to sell
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.SELL,
                strength=0.4,
                confidence=0.6,
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.M1,
                target_price=ask_price,
                stop_loss=ask_price * (1 + spread_pct * 3),
                metadata={"order_type": OrderType.LIMIT.value, "spread_bps": spread_pct * 10000},
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    async def on_order_book(self, book: OrderBookSnapshot) -> Optional[Signal]:
        return None

    def update_inventory(self, quantity: float, side: Side) -> None:
        """Update inventory after a fill."""
        if side == Side.BUY:
            self._current_inventory += quantity
        else:
            self._current_inventory -= quantity

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0] if "_" in self.genome.name else "BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.M1]
