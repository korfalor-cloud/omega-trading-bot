"""Pairs Trading Strategy.

Statistical arbitrage between two cointegrated assets.
Trades the spread when it deviates from its mean.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class PairsTradingStrategy(Strategy):
    """Pairs trading based on spread z-score.

    Monitors the price ratio/spread between two assets.
    Enters when spread deviates > z_threshold standard deviations.
    Exits when spread reverts to mean.

    Parameters (from genome.features):
        symbol_a: First asset (default "BTC/USDT")
        symbol_b: Second asset (default "ETH/USDT")
        lookback: Period for spread statistics (default 60)
        z_entry: Z-score threshold for entry (default 2.0)
        z_exit: Z-score threshold for exit (default 0.5)
        use_ratio: Use price ratio instead of spread (default True)
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._symbol_a = feats.get("symbol_a", "BTC/USDT")
        self._symbol_b = feats.get("symbol_b", "ETH/USDT")
        self._lookback = feats.get("lookback", 60)
        self._z_entry = feats.get("z_entry", 2.0)
        self._z_exit = feats.get("z_exit", 0.5)
        self._use_ratio = feats.get("use_ratio", True)
        self._atr_mult = genome.stop_loss_param

        self._bars_a: list[OHLCVBar] = []
        self._bars_b: list[OHLCVBar] = []
        self._in_position = False
        self._position_side: Side | None = None

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        # Route bar to correct buffer
        if bar.symbol == self._symbol_a:
            self._bars_a.append(bar)
        elif bar.symbol == self._symbol_b:
            self._bars_b.append(bar)
        else:
            return None

        # Need both series
        if len(self._bars_a) < self._lookback or len(self._bars_b) < self._lookback:
            return None

        # Trim buffers
        if len(self._bars_a) > 500:
            self._bars_a = self._bars_a[-300:]
        if len(self._bars_b) > 500:
            self._bars_b = self._bars_b[-300:]

        # Compute spread
        prices_a = np.array([b.close for b in self._bars_a[-self._lookback:]])
        prices_b = np.array([b.close for b in self._bars_b[-self._lookback:]])

        if self._use_ratio:
            # Log price ratio
            if np.any(prices_b <= 0):
                return None
            spread = np.log(prices_a / prices_b)
        else:
            spread = prices_a - prices_b

        # Z-score of current spread
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        if spread_std < 1e-10:
            return None

        z_score = (spread[-1] - spread_mean) / spread_std

        # Exit signal
        if self._in_position:
            if self._position_side == Side.BUY and z_score < self._z_exit:
                self._in_position = False
                self._position_side = None
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=self._symbol_a,
                    side=Side.SELL,
                    strength=0.5,
                    confidence=0.7,
                    signal_type=SignalType.EXIT,
                    timeframe=Timeframe.H1,
                )
            elif self._position_side == Side.SELL and z_score > -self._z_exit:
                self._in_position = False
                self._position_side = None
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=self._symbol_a,
                    side=Side.BUY,
                    strength=0.5,
                    confidence=0.7,
                    signal_type=SignalType.EXIT,
                    timeframe=Timeframe.H1,
                )
            return None

        # Entry signals
        if abs(z_score) > self._z_entry:
            curr_atr_a = self._compute_atr(self._bars_a)
            if z_score > self._z_entry:
                # Spread too high — sell A, buy B (spread should decrease)
                self._in_position = True
                self._position_side = Side.SELL
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=self._symbol_a,
                    side=Side.SELL,
                    strength=min(1.0, abs(z_score) / 4),
                    confidence=min(1.0, abs(z_score) / 3),
                    signal_type=SignalType.ENTRY,
                    timeframe=Timeframe.H1,
                    stop_loss=bar.close + self._atr_mult * curr_atr_a,
                    metadata={"z_score": z_score, "pair": self._symbol_b, "pair_side": "buy"},
                )
            elif z_score < -self._z_entry:
                # Spread too low — buy A, sell B (spread should increase)
                self._in_position = True
                self._position_side = Side.BUY
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=self._symbol_a,
                    side=Side.BUY,
                    strength=min(1.0, abs(z_score) / 4),
                    confidence=min(1.0, abs(z_score) / 3),
                    signal_type=SignalType.ENTRY,
                    timeframe=Timeframe.H1,
                    stop_loss=bar.close - self._atr_mult * curr_atr_a,
                    metadata={"z_score": z_score, "pair": self._symbol_b, "pair_side": "sell"},
                )

        return None

    def _compute_atr(self, bars: list[OHLCVBar], period: int = 14) -> float:
        if len(bars) < period + 1:
            return bars[-1].close * 0.02 if bars else 0.02
        trs = []
        for i in range(-period, 0):
            h = bars[i].high
            l = bars[i].low
            pc = bars[i - 1].close
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)
        return float(np.mean(trs))

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self._symbol_a, self._symbol_b]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
