"""Sector Rotation Strategy.

Implements:
- Crypto sector classification
- Sector momentum scoring
- Dynamic sector allocation
- Cross-sector spread trading
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)

# Crypto sector classification
SECTORS = {
    "L1": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"],
    "DeFi": ["UNI/USDT", "AAVE/USDT", "MKR/USDT", "COMP/USDT"],
    "L2": ["MATIC/USDT", "ARB/USDT", "OP/USDT"],
    "Meme": ["DOGE/USDT", "SHIB/USDT", "PEPE/USDT"],
}


class SectorRotationStrategy(Strategy):
    """Rotate capital across crypto sectors."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._lookback = feats.get("lookback", 30)
        self._rebalance_bars = feats.get("rebalance_bars", 72)
        self._top_sectors = feats.get("top_sectors", 1)

        self._bar_buffers: dict[str, list[OHLCVBar]] = {}
        self._sector_returns: dict[str, list[float]] = {s: [] for s in SECTORS}
        self._bars_since_rebalance = 0

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        if bar.symbol not in self._bar_buffers:
            self._bar_buffers[bar.symbol] = []
        self._bar_buffers[bar.symbol].append(bar)

        self._bars_since_rebalance += 1
        if self._bars_since_rebalance < self._rebalance_bars:
            return None

        # Compute sector momentum
        sector_momentum = {}
        for sector, symbols in SECTORS.items():
            returns = []
            for sym in symbols:
                if sym in self._bar_buffers and len(self._bar_buffers[sym]) >= self._lookback:
                    prices = [b.close for b in self._bar_buffers[sym][-self._lookback:]]
                    roc = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
                    returns.append(roc)
            if returns:
                sector_momentum[sector] = np.mean(returns)

        if not sector_momentum:
            return None

        # Rank sectors
        ranked = sorted(sector_momentum.items(), key=lambda x: x[1], reverse=True)
        top_sector = ranked[0][0]

        self._bars_since_rebalance = 0
        return Signal(
            strategy_id=self.strategy_id, symbol=SECTORS[top_sector][0],
            side=Side.BUY, strength=0.7, confidence=0.65,
            signal_type=SignalType.ENTRY, timeframe=Timeframe.H1,
            metadata={"sector": top_sector, "momentum": ranked[0][1], "rankings": dict(ranked)},
        )

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [s for symbols in SECTORS.values() for s in symbols]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
