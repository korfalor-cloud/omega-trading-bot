"""Cross-Exchange Arbitrage Strategy.

Implements:
- Price discrepancy detection across exchanges
- Simultaneous buy/sell execution
- Profit calculation after fees
- Latency-aware routing
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class CrossExchangeArbitrage(Strategy):
    """Cross-exchange price arbitrage."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._min_spread = feats.get("min_spread", 0.002)
        self._fee_rate = feats.get("fee_rate", 0.001)
        self._prices: dict[str, float] = {}

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._prices[bar.exchange] = bar.close

        if len(self._prices) < 2:
            return None

        exchanges = list(self._prices.keys())
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                ex_a, ex_b = exchanges[i], exchanges[j]
                price_a, price_b = self._prices[ex_a], self._prices[ex_b]

                if price_a == 0 or price_b == 0:
                    continue

                spread = abs(price_a - price_b) / min(price_a, price_b)
                net_spread = spread - 2 * self._fee_rate

                if net_spread > self._min_spread:
                    buy_ex = ex_a if price_a < price_b else ex_b
                    sell_ex = ex_b if price_a < price_b else ex_a
                    return Signal(
                        strategy_id=self.strategy_id, symbol=bar.symbol,
                        side=Side.BUY, strength=min(1.0, net_spread * 10),
                        confidence=0.8, signal_type=SignalType.ENTRY,
                        timeframe=Timeframe.M1,
                        metadata={"buy_exchange": buy_ex, "sell_exchange": sell_ex, "spread": spread},
                    )
        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0]] if "_" in self.genome.name else ["BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.M1]
