"""Tests for options strategies."""
from __future__ import annotations

import pytest
import asyncio
from datetime import datetime, timezone

from tradingbot.core.enums import Side, SignalType, Timeframe
from tradingbot.core.types import OHLCVBar, StrategyGenome
from tradingbot.strategies.options.strategies import (
    IronCondorStrategy,
    StraddleStrategy,
    CoveredCallStrategy,
)


def make_bar(symbol="BTC/USDT", close=50000.0):
    return OHLCVBar(
        timestamp=datetime.now(timezone.utc),
        symbol=symbol,
        timeframe=Timeframe.D1,
        open=close * 0.999,
        high=close * 1.005,
        low=close * 0.995,
        close=close,
        volume=100.0,
        exchange="binance",
    )


def make_genome(name="BTC_iron_condor", features=None):
    if features is None:
        features = [{}]
    return StrategyGenome(name=name, features=features)


class TestIronCondorStrategy:
    @pytest.fixture
    def strategy(self):
        genome = make_genome(features=[{
            "wing_width": 0.05,
            "short_strike_otm": 0.10,
            "min_iv_rank": 50,
        }])
        return IronCondorStrategy("test_ic", genome)

    @pytest.mark.asyncio
    async def test_warmup(self, strategy):
        for _ in range(20):
            result = await strategy.on_bar(make_bar())
            assert result is None

    @pytest.mark.asyncio
    async def test_signal_in_range(self, strategy):
        # Feed 30 bars of low-vol data
        for i in range(35):
            price = 50000 + (i % 5) * 100  # Range-bound
            await strategy.on_bar(make_bar(close=price))

        # Check if we get a signal (depends on IV rank)
        # At minimum, should not crash
        result = await strategy.on_bar(make_bar(close=50200))
        # May or may not signal depending on vol regime

    def test_required_symbols(self, strategy):
        symbols = strategy.required_symbols()
        assert len(symbols) > 0

    def test_required_timeframes(self, strategy):
        tfs = strategy.required_timeframes()
        assert Timeframe.D1 in tfs

    @pytest.mark.asyncio
    async def test_on_tick_returns_none(self, strategy):
        assert await strategy.on_tick(None) is None


class TestStraddleStrategy:
    @pytest.fixture
    def strategy(self):
        genome = make_genome(name="BTC_straddle", features=[{
            "iv_rank_threshold": 30,
            "max_hold_bars": 14,
        }])
        return StraddleStrategy("test_straddle", genome)

    @pytest.mark.asyncio
    async def test_warmup(self, strategy):
        for _ in range(20):
            result = await strategy.on_bar(make_bar())
            assert result is None

    @pytest.mark.asyncio
    async def test_low_iv_entry(self, strategy):
        # Feed 30 bars of high vol followed by low vol
        import numpy as np
        rng = np.random.default_rng(42)
        for i in range(35):
            # Decreasing volatility
            noise = rng.standard_normal() * (100 - i * 2)
            price = 50000 + noise
            await strategy.on_bar(make_bar(close=max(1, price)))

        # After warmup, check behavior
        result = await strategy.on_bar(make_bar(close=50000))

    @pytest.mark.asyncio
    async def test_exit_on_max_hold(self, strategy):
        # Warm up first
        for _ in range(30):
            await strategy.on_bar(make_bar())

        # Force entry
        strategy._in_trade = True
        strategy._trade_bars = 0
        strategy._max_hold_bars = 5

        for _ in range(4):
            result = await strategy.on_bar(make_bar())
            assert result is None

        result = await strategy.on_bar(make_bar())
        assert result is not None
        assert result.signal_type == SignalType.EXIT

    def test_required_symbols(self, strategy):
        assert len(strategy.required_symbols()) > 0


class TestCoveredCallStrategy:
    @pytest.fixture
    def strategy(self):
        genome = make_genome(name="BTC_covered_call", features=[{
            "call_otm_pct": 0.05,
            "min_premium_pct": 0.01,
        }])
        return CoveredCallStrategy("test_cc", genome)

    @pytest.mark.asyncio
    async def test_warmup(self, strategy):
        for _ in range(15):
            result = await strategy.on_bar(make_bar())
            assert result is None

    @pytest.mark.asyncio
    async def test_signal_possible(self, strategy):
        # Feed enough bars
        for _ in range(25):
            await strategy.on_bar(make_bar())

        # With 5% OTM call on BTC, should have some premium
        result = await strategy.on_bar(make_bar())
        # May or may not signal based on premium threshold

    def test_required_symbols(self, strategy):
        assert len(strategy.required_symbols()) > 0

    def test_required_timeframes(self, strategy):
        assert Timeframe.D1 in strategy.required_timeframes()
