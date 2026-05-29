"""Tests for volume breakout strategy."""
from __future__ import annotations

import pytest
import asyncio
import numpy as np
from datetime import datetime, timezone

from tradingbot.core.enums import Side, SignalType, Timeframe
from tradingbot.core.types import OHLCVBar, StrategyGenome
from tradingbot.strategies.breakout.volume_breakout import VolumeBreakoutStrategy


def make_bar(close=50000.0, high=None, low=None, volume=100.0):
    if high is None:
        high = close * 1.002
    if low is None:
        low = close * 0.998
    return OHLCVBar(
        timestamp=datetime.now(timezone.utc), symbol="BTC/USDT",
        timeframe=Timeframe.H1, open=close * 0.999, high=high, low=low,
        close=close, volume=volume, exchange="binance",
    )


def make_genome():
    return StrategyGenome(
        name="BTC_vol_breakout",
        features=[{"lookback": 10, "volume_multiplier": 1.5, "hold_bars": 5}],
    )


class TestVolumeBreakoutStrategy:
    @pytest.fixture
    def strategy(self):
        return VolumeBreakoutStrategy("test_vb", make_genome())

    @pytest.mark.asyncio
    async def test_warmup(self, strategy):
        for _ in range(5):
            r = await strategy.on_bar(make_bar())
            assert r is None

    @pytest.mark.asyncio
    async def test_no_breakout(self, strategy):
        for _ in range(30):
            await strategy.on_bar(make_bar(close=50000, volume=100))
        r = await strategy.on_bar(make_bar(close=50000, volume=100))
        assert r is None

    @pytest.mark.asyncio
    async def test_breakout_up(self, strategy):
        for _ in range(20):
            await strategy.on_bar(make_bar(close=50000, high=50100, low=49900, volume=100))
        # Breakout above resistance with volume spike
        r = await strategy.on_bar(make_bar(close=50200, high=50300, low=50100, volume=300))
        assert r is not None
        assert r.side == Side.BUY

    @pytest.mark.asyncio
    async def test_breakout_down(self, strategy):
        for _ in range(20):
            await strategy.on_bar(make_bar(close=50000, high=50100, low=49900, volume=100))
        r = await strategy.on_bar(make_bar(close=49800, high=49900, low=49700, volume=300))
        assert r is not None
        assert r.side == Side.SELL

    @pytest.mark.asyncio
    async def test_no_breakout_low_volume(self, strategy):
        for _ in range(20):
            await strategy.on_bar(make_bar(close=50000, high=50100, low=49900, volume=100))
        r = await strategy.on_bar(make_bar(close=50200, high=50300, low=50100, volume=50))
        assert r is None  # Volume too low

    @pytest.mark.asyncio
    async def test_exit_on_hold(self, strategy):
        for _ in range(20):
            await strategy.on_bar(make_bar(close=50000, high=50100, low=49900, volume=100))
        await strategy.on_bar(make_bar(close=50200, high=50300, low=50100, volume=300))
        assert strategy._in_trade is True

        for _ in range(4):
            r = await strategy.on_bar(make_bar())
            assert r is None
        r = await strategy.on_bar(make_bar())
        assert r is not None
        assert r.signal_type == SignalType.EXIT

    def test_required_symbols(self, strategy):
        assert len(strategy.required_symbols()) == 1

    def test_required_timeframes(self, strategy):
        assert Timeframe.H1 in strategy.required_timeframes()

    @pytest.mark.asyncio
    async def test_on_tick_returns_none(self, strategy):
        assert await strategy.on_tick(None) is None
