"""Basis Trade Strategy.

Exploits the difference between spot and futures prices (basis).
When basis is elevated, short futures + long spot.
When basis is inverted, long futures + short spot.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class BasisTradeStrategy(Strategy):
    """Cash-and-carry basis trade.

    Parameters (from genome.features):
        basis_threshold_pct: Min annualized basis to enter (default 10%)
        max_hold_bars: Max bars to hold (default 168 — 1 week on 1h)
        use_volume: Require volume confirmation (default True)
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._basis_threshold_pct = feats.get("basis_threshold_pct", 0.10)
        self._max_hold_bars = feats.get("max_hold_bars", 168)
        self._use_volume = feats.get("use_volume", True)
        self._bar_buffer: list[OHLCVBar] = []
        self._min_bars = 20
        self._in_trade = False
        self._trade_bars = 0
        self._basis_history: list[float] = []

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._min_bars:
            return None

        if len(self._bar_buffer) > 300:
            self._bar_buffer = self._bar_buffer[-200:]

        # Basis = (futures - spot) / spot
        # We approximate using bar metadata if available
        spot_price = bar.close
        futures_price = bar.open if bar.open > 0 else spot_price  # Approximate
        basis_pct = (futures_price - spot_price) / spot_price if spot_price > 0 else 0

        # Annualize (assuming daily basis)
        annual_basis = basis_pct * 365
        self._basis_history.append(annual_basis)

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
            # Exit if basis reverts
            if abs(annual_basis) < self._basis_threshold_pct / 2:
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

        # Volume filter
        if self._use_volume:
            vol_avg = np.mean([b.volume for b in self._bar_buffer[-20:]])
            if vol_avg > 0 and bar.volume < vol_avg * 0.8:
                return None

        # Elevated basis: short futures + long spot
        if annual_basis > self._basis_threshold_pct:
            self._in_trade = True
            self._trade_bars = 0
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,  # Long spot
                strength=min(1.0, annual_basis / (self._basis_threshold_pct * 3)),
                confidence=min(1.0, 0.5 + annual_basis / (self._basis_threshold_pct * 5)),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                metadata={
                    "annual_basis": annual_basis,
                    "hedge_side": "short_futures",
                },
            )

        # Inverted basis: long futures + short spot
        if annual_basis < -self._basis_threshold_pct:
            self._in_trade = True
            self._trade_bars = 0
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.SELL,  # Short spot
                strength=min(1.0, abs(annual_basis) / (self._basis_threshold_pct * 3)),
                confidence=min(1.0, 0.5 + abs(annual_basis) / (self._basis_threshold_pct * 5)),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                metadata={
                    "annual_basis": annual_basis,
                    "hedge_side": "long_futures",
                },
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0] if "_" in self.genome.name else "BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
