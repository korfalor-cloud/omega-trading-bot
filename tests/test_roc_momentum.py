"""Tests for ROC momentum strategy."""
from __future__ import annotations

import pytest
import asyncio
from datetime import datetime, timezone

from tradingbot.core.enums import Side, SignalType, Timeframe
from tradingbot.core.types import OHLCVBar, StrategyGenome
from tradingbot.strategies.momentum.roc_strategy import ROCMomentumStrategy


def make_bar(close=50000.0, volume=100.0):
    return OHLCVBar(
        timestamp=datetime.now(timezone.utc), symbol="BTC/USDT",
        timeframe=Timeframe.H1, open=close * 0.999, high=close * 1.002,
        low=close * 0.998, close=close, volume=volume,
        exchange="binance",
    )


def make_genome():
    return StrategyGenome(
        name="BTC_roc",
        features=[{"roc_period": 5, "entry_threshold": 0.02, "exit_threshold": 0.005, "volume_confirm": True}],
    )


class TestROCMomentumStrategy:
    @pytest.fixture
    def strategy(self):
        return ROCMomentumStrategy("test_roc", make_genome())

    @pytest.mark.asyncio
    async def test_warmup(self, strategy):
        for _ in range(3):
            r = await strategy.on_bar(make_bar())
            assert r is None

    @pytest.mark.asyncio
    async def test_bullish_entry(self, strategy):
        # Feed rising prices
        for i in range(15):
            price = 50000 + i * 200
            await strategy.on_bar(make_bar(close=price, volume=100 + i * 10))

    @pytest.mark.asyncio
    async def test_no_signal_small_roc(self, strategy):
        # Flat prices
        for _ in range(15):
            await strategy.on_bar(make_bar(close=50000))
        r = await strategy.on_bar(make_bar(close=50010))
        assert r is None

    def test_required_symbols(self, strategy):
        symbols = strategy.required_symbols()
        assert len(symbols) == 1

    def test_required_timeframes(self, strategy):
        assert Timeframe.H1 in strategy.required_timeframes()

    @pytest.mark.asyncio
    async def test_on_tick_returns_none(self, strategy):
        assert await strategy.on_tick(None) is None
