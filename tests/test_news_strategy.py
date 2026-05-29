"""Tests for news-driven strategy."""
from __future__ import annotations

import pytest
import asyncio
from datetime import datetime, timezone

from tradingbot.core.enums import Side, SignalType, Timeframe
from tradingbot.core.types import OHLCVBar, StrategyGenome
from tradingbot.strategies.news.strategy import NewsDrivenStrategy


def make_bar(vwap=0.0, close=50000.0):
    return OHLCVBar(
        timestamp=datetime.now(timezone.utc), symbol="BTC/USDT",
        timeframe=Timeframe.H1, open=close * 0.999, high=close * 1.002,
        low=close * 0.998, close=close, volume=100.0,
        exchange="binance", vwap=vwap,
    )


def make_genome(fade=False):
    return StrategyGenome(
        name="BTC_news",
        features=[{"sentiment_threshold": 0.3, "fade_mode": fade, "lookback": 20, "hold_bars": 5}],
    )


class TestNewsDrivenStrategy:
    @pytest.fixture
    def strategy(self):
        return NewsDrivenStrategy("test_news", make_genome())

    @pytest.fixture
    def fade_strategy(self):
        return NewsDrivenStrategy("test_fade", make_genome(fade=True))

    @pytest.mark.asyncio
    async def test_warmup(self, strategy):
        for _ in range(5):
            r = await strategy.on_bar(make_bar(vwap=0.0))
            assert r is None

    @pytest.mark.asyncio
    async def test_follow_bullish(self, strategy):
        for i in range(25):
            await strategy.on_bar(make_bar(vwap=0.1 + i * 0.1))
        r = await strategy.on_bar(make_bar(vwap=3.0))
        assert r is not None
        assert r.side == Side.BUY

    @pytest.mark.asyncio
    async def test_follow_bearish(self, strategy):
        for i in range(25):
            await strategy.on_bar(make_bar(vwap=-0.1 - i * 0.1))
        r = await strategy.on_bar(make_bar(vwap=-3.0))
        assert r is not None
        assert r.side == Side.SELL

    @pytest.mark.asyncio
    async def test_neutral_no_signal(self, strategy):
        for _ in range(25):
            await strategy.on_bar(make_bar(vwap=0.0))
        r = await strategy.on_bar(make_bar(vwap=0.0))
        assert r is None

    @pytest.mark.asyncio
    async def test_fade_mode_bullish(self, fade_strategy):
        for i in range(25):
            await fade_strategy.on_bar(make_bar(vwap=0.1 + i * 0.1, close=50000 + i * 200))
        r = await fade_strategy.on_bar(make_bar(vwap=3.0, close=55000))
        assert r is not None
        assert r.side == Side.SELL  # Fade bullish = sell

    @pytest.mark.asyncio
    async def test_fade_mode_bearish(self, fade_strategy):
        for i in range(25):
            await fade_strategy.on_bar(make_bar(vwap=-0.1 - i * 0.1, close=50000 - i * 200))
        r = await fade_strategy.on_bar(make_bar(vwap=-3.0, close=45000))
        assert r is not None
        assert r.side == Side.BUY  # Fade bearish = buy

    @pytest.mark.asyncio
    async def test_exit_on_hold(self, strategy):
        for i in range(25):
            await strategy.on_bar(make_bar(vwap=0.1 + i * 0.1))
        await strategy.on_bar(make_bar(vwap=3.0))
        assert strategy._in_trade is True

        for _ in range(4):
            r = await strategy.on_bar(make_bar(vwap=0.5))
            assert r is None

        r = await strategy.on_bar(make_bar(vwap=0.5))
        assert r is not None
        assert r.signal_type == SignalType.EXIT

    def test_required_symbols(self, strategy):
        symbols = strategy.required_symbols()
        assert len(symbols) == 1

    def test_required_timeframes(self, strategy):
        assert Timeframe.H1 in strategy.required_timeframes()

    @pytest.mark.asyncio
    async def test_on_tick_returns_none(self, strategy):
        assert await strategy.on_tick(None) is None
