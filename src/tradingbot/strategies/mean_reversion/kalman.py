"""Kalman Filter Mean Reversion Strategy.

Implements:
- Kalman filter for dynamic hedge ratio
- Mean reversion on filtered spread
- Adaptive thresholds
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class KalmanMeanReversion(Strategy):
    """Kalman filter-based mean reversion."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._entry_zscore = feats.get("entry_zscore", 2.0)
        self._exit_zscore = feats.get("exit_zscore", 0.5)

        self._bar_buffer_a: list[OHLCVBar] = []
        self._bar_buffer_b: list[OHLCVBar] = []
        self._spread_history: list[float] = []
        self._hedge_ratio = 1.0
        self._P = 1.0
        self._R = 0.01
        self._Q = 0.001
        self._in_trade = False
        self._trade_side = ""

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        symbols = self.required_symbols()
        if len(symbols) < 2:
            return None

        if bar.symbol == symbols[0]:
            self._bar_buffer_a.append(bar)
        elif bar.symbol == symbols[1]:
            self._bar_buffer_b.append(bar)
        else:
            return None

        if len(self._bar_buffer_a) < 30 or len(self._bar_buffer_b) < 30:
            return None

        price_a = self._bar_buffer_a[-1].close
        price_b = self._bar_buffer_b[-1].close

        # Kalman update
        P_pred = self._P + self._Q
        S = P_pred + self._R
        K = P_pred / S
        observed = price_a / price_b if price_b > 0 else self._hedge_ratio
        self._hedge_ratio = self._hedge_ratio + K * (observed - self._hedge_ratio)
        self._P = (1 - K) * P_pred

        # Spread
        spread = price_a - self._hedge_ratio * price_b
        self._spread_history.append(spread)

        if len(self._spread_history) < 30:
            return None

        spreads = np.array(self._spread_history[-30:])
        zscore = (spread - np.mean(spreads)) / np.std(spreads) if np.std(spreads) > 0 else 0

        # Exit
        if self._in_trade:
            if self._trade_side == "long" and zscore > -self._exit_zscore:
                self._in_trade = False
                return Signal(strategy_id=self.strategy_id, symbol=bar.symbol, side=Side.SELL, strength=0.6, confidence=0.65, signal_type=SignalType.EXIT, timeframe=Timeframe.H1)
            if self._trade_side == "short" and zscore < self._exit_zscore:
                self._in_trade = False
                return Signal(strategy_id=self.strategy_id, symbol=bar.symbol, side=Side.BUY, strength=0.6, confidence=0.65, signal_type=SignalType.EXIT, timeframe=Timeframe.H1)
            return None

        # Entry
        if zscore < -self._entry_zscore:
            self._in_trade = True
            self._trade_side = "long"
            return Signal(strategy_id=self.strategy_id, symbol=bar.symbol, side=Side.BUY, strength=min(1.0, abs(zscore) / 3), confidence=0.65, signal_type=SignalType.ENTRY, timeframe=Timeframe.H1, metadata={"zscore": zscore})

        if zscore > self._entry_zscore:
            self._in_trade = True
            self._trade_side = "short"
            return Signal(strategy_id=self.strategy_id, symbol=bar.symbol, side=Side.SELL, strength=min(1.0, abs(zscore) / 3), confidence=0.65, signal_type=SignalType.ENTRY, timeframe=Timeframe.H1, metadata={"zscore": zscore})

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        name = self.genome.name
        if "_" in name:
            parts = name.split("_")
            return [parts[0], parts[1]] if len(parts) >= 2 else ["BTC/USDT", "ETH/USDT"]
        return ["BTC/USDT", "ETH/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
