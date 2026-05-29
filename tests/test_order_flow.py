"""Tests for order flow scalping strategy."""
from __future__ import annotations

import pytest
import asyncio
from datetime import datetime, timezone

from tradingbot.core.enums import Side, SignalType, Timeframe
from tradingbot.core.types import OHLCVBar, StrategyGenome
from tradingbot.strategies.scalping.order_flow import OrderFlowScalpStrategy


def make_bar(close=50000.0, high=50100.0, low=49900.0, volume=100.0):
    return OHLCVBar(
        timestamp=datetime.now(timezone.utc), symbol="BTC/USDT",
        timeframe=Timeframe.H1, open=close * 0.999, high=high, low=low,
        close=close, volume=volume, exchange="binance",
    )


def make_genome():
    return StrategyGenome(
        name="BTC_of",
        features=[{"imbalance_threshold": 0.3, "intensity_threshold": 1.5, "hold_bars": 3}],
    )


class TestOrderFlowScalpStrategy:
    @pytest.fixture
    def strategy(self):
        return OrderFlowScalpStrategy("test_of", make_genome())

    @pytest.mark.asyncio
    async def test_warmup(self, strategy):
        for _ in range(5):
            r = await strategy.on_bar(make_bar())
            assert r is None

    @pytest.mark.asyncio
    async def test_no_signal_flat(self, strategy):
        for _ in range(30):
            await strategy.on_bar(make_bar(close=50000, volume=100))
        r = await strategy.on_bar(make_bar(close=50000, volume=100))
        assert r is None

    @pytest.mark.asyncio
    async def test_required_symbols(self, strategy):
        assert len(strategy.required_symbols()) == 1

    def test_required_timeframes(self, strategy):
        assert Timeframe.M1 in strategy.required_timeframes()

    @pytest.mark.asyncio
    async def test_on_tick_returns_none(self, strategy):
        assert await strategy.on_tick(None) is None
