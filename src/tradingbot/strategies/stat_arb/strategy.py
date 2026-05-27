"""Statistical Arbitrage Strategy.

Implements:
- Mean-reversion on z-scored spread
- Multi-asset spread trading
- Kalman filter hedge ratio estimation
- Adaptive entry/exit thresholds
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class StatArbStrategy(Strategy):
    """Statistical arbitrage on cointegrated pairs.

    Parameters (from genome.features):
        entry_zscore: Z-score threshold to enter (default 2.0)
        exit_zscore: Z-score threshold to exit (default 0.5)
        lookback: Lookback for z-score calculation (default 60)
        use_kalman: Use Kalman filter for hedge ratio (default True)
        max_hold_bars: Max bars to hold (default 100)
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._entry_zscore = feats.get("entry_zscore", 2.0)
        self._exit_zscore = feats.get("exit_zscore", 0.5)
        self._lookback = feats.get("lookback", 60)
        self._use_kalman = feats.get("use_kalman", True)
        self._max_hold_bars = feats.get("max_hold_bars", 100)

        self._bar_buffer_a: list[OHLCVBar] = []
        self._bar_buffer_b: list[OHLCVBar] = []
        self._spread_history: list[float] = []
        self._hedge_ratio = 1.0
        self._in_trade = False
        self._trade_bars = 0
        self._trade_side = ""

        # Kalman filter state
        self._kalman_P = 1.0  # Covariance
        self._kalman_R = 0.01  # Measurement noise
        self._kalman_Q = 0.001  # Process noise

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        # Route to correct buffer based on symbol
        symbols = self.required_symbols()
        if len(symbols) < 2:
            return None

        if bar.symbol == symbols[0]:
            self._bar_buffer_a.append(bar)
        elif bar.symbol == symbols[1]:
            self._bar_buffer_b.append(bar)
        else:
            return None

        if len(self._bar_buffer_a) < self._lookback or len(self._bar_buffer_b) < self._lookback:
            return None

        # Trim buffers
        if len(self._bar_buffer_a) > 300:
            self._bar_buffer_a = self._bar_buffer_a[-200:]
            self._bar_buffer_b = self._bar_buffer_b[-200:]

        prices_a = np.array([b.close for b in self._bar_buffer_a[-self._lookback:]])
        prices_b = np.array([b.close for b in self._bar_buffer_b[-self._lookback:]])

        # Update hedge ratio
        if self._use_kalman:
            self._update_kalman(prices_a[-1], prices_b[-1])
        else:
            self._hedge_ratio = self._ols_hedge_ratio(prices_a, prices_b)

        # Compute spread and z-score
        spread = prices_a - self._hedge_ratio * prices_b
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)

        if spread_std < 1e-10:
            return None

        zscore = (spread[-1] - spread_mean) / spread_std
        self._spread_history.append(zscore)

        # Exit logic
        if self._in_trade:
            self._trade_bars += 1

            if self._trade_bars >= self._max_hold_bars:
                self._in_trade = False
                self._trade_bars = 0
                return self._make_exit_signal(bar)

            # Exit on mean reversion
            if self._trade_side == "short_spread" and zscore < self._exit_zscore:
                self._in_trade = False
                self._trade_bars = 0
                return self._make_exit_signal(bar)

            if self._trade_side == "long_spread" and zscore > -self._exit_zscore:
                self._in_trade = False
                self._trade_bars = 0
                return self._make_exit_signal(bar)

            return None

        # Entry logic
        if zscore > self._entry_zscore:
            # Spread is high — short spread (short A, long B)
            self._in_trade = True
            self._trade_bars = 0
            self._trade_side = "short_spread"
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.SELL,
                strength=min(1.0, abs(zscore) / (self._entry_zscore * 2)),
                confidence=min(1.0, 0.5 + abs(zscore) / (self._entry_zscore * 3)),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                metadata={"zscore": zscore, "hedge_ratio": self._hedge_ratio, "spread_side": "short"},
            )

        if zscore < -self._entry_zscore:
            # Spread is low — long spread (long A, short B)
            self._in_trade = True
            self._trade_bars = 0
            self._trade_side = "long_spread"
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,
                strength=min(1.0, abs(zscore) / (self._entry_zscore * 2)),
                confidence=min(1.0, 0.5 + abs(zscore) / (self._entry_zscore * 3)),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                metadata={"zscore": zscore, "hedge_ratio": self._hedge_ratio, "spread_side": "long"},
            )

        return None

    def _update_kalman(self, price_a: float, price_b: float) -> None:
        """Update hedge ratio using Kalman filter."""
        # Prediction
        P_pred = self._kalman_P + self._kalman_Q

        # Update
        S = P_pred + self._kalman_R  # Innovation covariance
        K = P_pred / S  # Kalman gain

        # Observation: price_a = hedge_ratio * price_b
        observed_ratio = price_a / price_b if price_b > 0 else self._hedge_ratio
        self._hedge_ratio = self._hedge_ratio + K * (observed_ratio - self._hedge_ratio)
        self._kalman_P = (1 - K) * P_pred

    def _ols_hedge_ratio(self, prices_a: np.ndarray, prices_b: np.ndarray) -> float:
        """OLS hedge ratio."""
        try:
            X = np.column_stack([np.ones(len(prices_b)), prices_b])
            coeffs = np.linalg.lstsq(X, prices_a, rcond=None)[0]
            return coeffs[1]
        except np.linalg.LinAlgError:
            return 1.0

    def _make_exit_signal(self, bar: OHLCVBar) -> Signal:
        return Signal(
            strategy_id=self.strategy_id,
            symbol=bar.symbol,
            side=Side.SELL,
            strength=0.5,
            confidence=0.6,
            signal_type=SignalType.EXIT,
            timeframe=Timeframe.H1,
        )

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
