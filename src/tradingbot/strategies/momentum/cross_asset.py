"""Cross-Asset Momentum Strategy.

Implements:
- Momentum across multiple assets
- Relative strength ranking
- Sector rotation
- Momentum persistence filter
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class CrossAssetMomentumStrategy(Strategy):
    """Cross-asset momentum rotation strategy."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._lookback = feats.get("lookback", 30)
        self._top_n = feats.get("top_n", 3)
        self._rebalance_bars = feats.get("rebalance_bars", 24)

        self._bar_buffers: dict[str, list[OHLCVBar]] = {}
        self._bars_since_rebalance = 0
        self._rankings: list[str] = []

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        if bar.symbol not in self._bar_buffers:
            self._bar_buffers[bar.symbol] = []
        self._bar_buffers[bar.symbol].append(bar)

        self._bars_since_rebalance += 1
        if self._bars_since_rebalance < self._rebalance_bars:
            return None

        if len(self._bar_buffers) < 2:
            return None

        # Compute momentum for each asset
        momentum = {}
        for symbol, bars in self._bar_buffers.items():
            if len(bars) < self._lookback:
                continue
            prices = np.array([b.close for b in bars[-self._lookback:]])
            roc = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
            momentum[symbol] = roc

        if len(momentum) < 2:
            return None

        # Rank by momentum
        ranked = sorted(momentum.items(), key=lambda x: x[1], reverse=True)
        top = [s for s, _ in ranked[:self._top_n]]

        # Signal for top ranked asset
        if top and top[0] != self._rankings[0] if self._rankings else True:
            self._rankings = top
            self._bars_since_rebalance = 0
            return Signal(
                strategy_id=self.strategy_id, symbol=top[0],
                side=Side.BUY, strength=0.7, confidence=0.65,
                signal_type=SignalType.ENTRY, timeframe=Timeframe.H1,
                metadata={"rankings": top, "momentum": momentum[top[0]]},
            )

        self._bars_since_rebalance = 0
        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
