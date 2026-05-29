"""Tests for EMA cross scalping strategy."""
from __future__ import annotations

import pytest
import asyncio
from datetime import datetime, timezone

from tradingbot.core.enums import Side, SignalType, Timeframe
from tradingbot.core.types import OHLCVBar, StrategyGenome
from tradingbot.strategies.scalping.ema_cross import EMACrossScalpStrategy


def make_bar(close=50000.0):
    return OHLCVBar(
        timestamp=datetime.now(timezone.utc), symbol="BTC/USDT",
        timeframe=Timeframe.H1, open=close * 0.999, high=close * 1.002,
        low=close * 0.998, close=close, volume=100.0, exchange="binance",
    )


def make_genome():
    return StrategyGenome(
        name="BTC_ema",
        features=[{"fast_period": 3, "slow_period": 8, "rsi_filter": False, "hold_bars": 3}],
    )


class TestEMACrossScalpStrategy:
    @pytest.fixture
    def strategy(self):
        return EMACrossScalpStrategy("test_ema", make_genome())

    @pytest.mark.asyncio
    async def test_warmup(self, strategy):
        for _ in range(5):
            r = await strategy.on_bar(make_bar())
            assert r is None

    @pytest.mark.asyncio
    async def test_no_signal_flat(self, strategy):
        for _ in range(30):
            await strategy.on_bar(make_bar(close=50000))
        r = await strategy.on_bar(make_bar(close=50000))
        assert r is None

    @pytest.mark.asyncio
    async def test_required_symbols(self, strategy):
        assert len(strategy.required_symbols()) == 1

    def test_required_timeframes(self, strategy):
        assert Timeframe.M5 in strategy.required_timeframes()

    @pytest.mark.asyncio
    async def test_on_tick_returns_none(self, strategy):
        assert await strategy.on_tick(None) is None

    def test_ema_calculation(self, strategy):
        import numpy as np
        data = np.array([100, 101, 102, 103, 104], dtype=float)
        ema = strategy._ema(data, 3)
        assert ema[0] == 100
        assert ema[-1] > ema[0]  # Trending up
