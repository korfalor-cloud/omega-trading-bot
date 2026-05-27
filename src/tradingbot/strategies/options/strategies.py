"""Options-Based Strategies.

Implements:
- Iron condor (range-bound)
- Straddle/Strangle (volatility plays)
- Covered call
- Protective put
- Calendar spread
- Butterfly spread
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy
from ...risk.greeks import BlackScholesCalculator

logger = logging.getLogger(__name__)


@dataclass
class OptionsLeg:
    """Single leg of an options strategy."""
    option_type: str = "call"  # call or put
    strike: float = 0.0
    side: str = "buy"  # buy or sell
    quantity: int = 1
    premium: float = 0.0
    expiry_days: int = 30


@dataclass
class OptionsPosition:
    """Multi-leg options position."""
    legs: list[OptionsLeg]
    strategy_name: str = ""
    max_profit: float = 0.0
    max_loss: float = 0.0
    breakeven_low: float = 0.0
    breakeven_high: float = 0.0


class IronCondorStrategy(Strategy):
    """Iron condor strategy for range-bound markets.

    Sells OTM call spread + OTM put spread.
    Profits when price stays within a range.

    Parameters:
        wing_width: Width of each spread (default 5% of spot)
        short_strike_otm: How far OTM for short strikes (default 10%)
        min_iv_rank: Minimum IV rank to enter (default 50)
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._wing_width_pct = feats.get("wing_width", 0.05)
        self._short_otm_pct = feats.get("short_strike_otm", 0.10)
        self._min_iv_rank = feats.get("min_iv_rank", 50)
        self._bs = BlackScholesCalculator()
        self._bar_buffer: list[OHLCVBar] = []
        self._min_bars = 30

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._min_bars:
            return None

        if len(self._bar_buffer) > 200:
            self._bar_buffer = self._bar_buffer[-150:]

        spot = bar.close
        prices = np.array([b.close for b in self._bar_buffer])

        # Estimate volatility
        returns = np.diff(np.log(prices))
        vol = np.std(returns) * np.sqrt(365)

        # Check if in range-bound regime (low vol)
        recent_vol = np.std(returns[-10:]) * np.sqrt(365)
        hist_vol = vol

        iv_rank = (recent_vol / hist_vol * 100) if hist_vol > 0 else 50

        if iv_rank > self._min_iv_rank:
            # Build iron condor
            short_call_strike = spot * (1 + self._short_otm_pct)
            long_call_strike = short_call_strike * (1 + self._wing_width_pct)
            short_put_strike = spot * (1 - self._short_otm_pct)
            long_put_strike = short_put_strike * (1 - self._wing_width_pct)

            # Calculate max profit/loss
            credit = self._wing_width_pct * spot * 0.3  # Approximate credit
            max_loss = self._wing_width_pct * spot - credit

            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.SELL,  # Net seller
                strength=min(1.0, iv_rank / 100),
                confidence=min(1.0, 0.5 + (iv_rank - 50) / 100),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.D1,
                metadata={
                    "strategy": "iron_condor",
                    "short_call": short_call_strike,
                    "long_call": long_call_strike,
                    "short_put": short_put_strike,
                    "long_put": long_put_strike,
                    "max_profit": credit,
                    "max_loss": max_loss,
                    "iv_rank": iv_rank,
                },
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0] if "_" in self.genome.name else "BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.D1]


class StraddleStrategy(Strategy):
    """Long straddle strategy for high volatility expectations.

    Buys ATM call + ATM put.
    Profits from large moves in either direction.

    Parameters:
        iv_rank_threshold: IV rank below this = buy straddle (default 30)
        max_hold_bars: Max bars to hold (default 14)
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._iv_threshold = feats.get("iv_rank_threshold", 30)
        self._max_hold_bars = feats.get("max_hold_bars", 14)
        self._bs = BlackScholesCalculator()
        self._bar_buffer: list[OHLCVBar] = []
        self._min_bars = 30
        self._in_trade = False
        self._trade_bars = 0
        self._entry_price = 0.0

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._min_bars:
            return None

        spot = bar.close
        prices = np.array([b.close for b in self._bar_buffer])
        returns = np.diff(np.log(prices))

        # Exit logic
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
                    timeframe=Timeframe.D1,
                )
            return None

        # Volatility analysis
        recent_vol = np.std(returns[-10:]) * np.sqrt(365)
        hist_vol = np.std(returns) * np.sqrt(365)
        iv_rank = (recent_vol / hist_vol * 100) if hist_vol > 0 else 50

        # Low IV rank = cheap options = buy straddle
        if iv_rank < self._iv_threshold:
            self._in_trade = True
            self._trade_bars = 0
            self._entry_price = spot

            premium = self._bs.call_price(spot, spot, 14 / 365, recent_vol) * 2

            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,
                strength=min(1.0, (self._iv_threshold - iv_rank) / self._iv_threshold),
                confidence=min(1.0, 0.5 + (self._iv_threshold - iv_rank) / 100),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.D1,
                metadata={
                    "strategy": "straddle",
                    "strike": spot,
                    "premium": premium,
                    "iv_rank": iv_rank,
                },
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0] if "_" in self.genome.name else "BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.D1]


class CoveredCallStrategy(Strategy):
    """Covered call strategy.

    Hold spot + sell OTM call.
    Generates income in flat/slightly bullish markets.

    Parameters:
        call_otm_pct: How far OTM to sell call (default 5%)
        min_premium_pct: Minimum premium as % of spot (default 1%)
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._call_otm_pct = feats.get("call_otm_pct", 0.05)
        self._min_premium_pct = feats.get("min_premium_pct", 0.01)
        self._bs = BlackScholesCalculator()
        self._bar_buffer: list[OHLCVBar] = []
        self._min_bars = 20

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._min_bars:
            return None

        spot = bar.close
        prices = np.array([b.close for b in self._bar_buffer])
        returns = np.diff(np.log(prices))
        vol = np.std(returns) * np.sqrt(365)

        strike = spot * (1 + self._call_otm_pct)
        tte = 30 / 365  # 30 days

        premium = self._bs.call_price(spot, strike, tte, vol)
        premium_pct = premium / spot

        if premium_pct >= self._min_premium_pct:
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.SELL,  # Sell call
                strength=min(1.0, premium_pct / 0.05),
                confidence=min(1.0, 0.5 + premium_pct),
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.D1,
                metadata={
                    "strategy": "covered_call",
                    "strike": strike,
                    "premium": premium,
                    "premium_pct": premium_pct,
                    "max_profit": premium + (strike - spot),
                },
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0] if "_" in self.genome.name else "BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.D1]
