"""Tests for candlestick pattern strategy."""
from __future__ import annotations

import pytest
import asyncio
from datetime import datetime, timezone

from tradingbot.core.enums import Side, SignalType, Timeframe
from tradingbot.core.types import OHLCVBar, StrategyGenome
from tradingbot.strategies.mean_reversion.engulfing import CandlestickStrategy


def make_bar(open_p=50000, close=50100, high=None, low=None, volume=100):
    if high is None:
        high = max(open_p, close) + abs(close - open_p) * 0.5 + 1
    if low is None:
        low = min(open_p, close) - abs(close - open_p) * 0.5 - 1
    return OHLCVBar(
        timestamp=datetime.now(timezone.utc), symbol="BTC/USDT",
        timeframe=Timeframe.H1, open=open_p, high=high, low=low,
        close=close, volume=volume, exchange="binance",
    )


def make_genome():
    return StrategyGenome(
        name="BTC_candle",
        features=[{"min_body_ratio": 0.5, "volume_confirm": False, "hold_bars": 3}],
    )


class TestCandlestickStrategy:
    @pytest.fixture
    def strategy(self):
        return CandlestickStrategy("test_candle", make_genome())

    @pytest.mark.asyncio
    async def test_warmup(self, strategy):
        for _ in range(3):
            r = await strategy.on_bar(make_bar())
            assert r is None

    @pytest.mark.asyncio
    async def test_bullish_engulfing(self, strategy):
        # Use tiny-body bars for warmup (won't trigger patterns)
        for _ in range(10):
            await strategy.on_bar(make_bar(open_p=50000, close=50001))
        # Bearish bar
        await strategy.on_bar(make_bar(open_p=50200, close=50000))
        # Bullish engulfing
        r = await strategy.on_bar(make_bar(open_p=49900, close=50300))
        assert r is not None
        assert r.side == Side.BUY
        assert r.metadata.get("pattern") == "bullish_engulfing"

    @pytest.mark.asyncio
    async def test_bearish_engulfing(self, strategy):
        for _ in range(10):
            await strategy.on_bar(make_bar(open_p=50000, close=50001))
        # Bullish bar
        await strategy.on_bar(make_bar(open_p=50000, close=50200))
        # Bearish engulfing
        r = await strategy.on_bar(make_bar(open_p=50300, close=49900))
        assert r is not None
        assert r.side == Side.SELL
        assert r.metadata.get("pattern") == "bearish_engulfing"

    @pytest.mark.asyncio
    async def test_hammer(self, strategy):
        for _ in range(10):
            await strategy.on_bar(make_bar(open_p=50000, close=50001))
        # Hammer: small body at top, long lower shadow
        r = await strategy.on_bar(make_bar(open_p=50000, close=50050, high=50100, low=49000))
        assert r is not None
        assert r.side == Side.BUY
        assert r.metadata.get("pattern") == "hammer"

    @pytest.mark.asyncio
    async def test_shooting_star(self, strategy):
        for _ in range(10):
            await strategy.on_bar(make_bar(open_p=50000, close=50001))
        # Shooting star: small body at bottom, long upper shadow
        r = await strategy.on_bar(make_bar(open_p=50000, close=49950, high=51000, low=49900))
        assert r is not None
        assert r.side == Side.SELL
        assert r.metadata.get("pattern") == "shooting_star"

    @pytest.mark.asyncio
    async def test_exit_on_hold(self, strategy):
        for _ in range(10):
            await strategy.on_bar(make_bar(open_p=50000, close=50001))
        # Enter via engulfing
        await strategy.on_bar(make_bar(open_p=50200, close=50000))
        r = await strategy.on_bar(make_bar(open_p=49900, close=50300))
        assert r is not None
        assert strategy._in_trade is True

        # Hold until exit
        for _ in range(2):
            r = await strategy.on_bar(make_bar())
            assert r is None
        r = await strategy.on_bar(make_bar())
        assert r is not None
        assert r.signal_type == SignalType.EXIT

    @pytest.mark.asyncio
    async def test_on_tick_returns_none(self, strategy):
        assert await strategy.on_tick(None) is None

    def test_required_symbols(self, strategy):
        assert len(strategy.required_symbols()) == 1
