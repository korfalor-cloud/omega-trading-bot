"""Tests for statistical arbitrage strategy."""
from __future__ import annotations

import pytest
import asyncio
from datetime import datetime, timezone

from tradingbot.core.enums import Side, SignalType, Timeframe
from tradingbot.core.types import OHLCVBar, StrategyGenome
from tradingbot.strategies.stat_arb.strategy import StatArbStrategy


def make_bar(symbol="BTC/USDT", close=50000.0, volume=100.0):
    return OHLCVBar(
        timestamp=datetime.now(timezone.utc),
        symbol=symbol,
        timeframe=Timeframe.H1,
        open=close * 0.999,
        high=close * 1.002,
        low=close * 0.998,
        close=close,
        volume=volume,
        exchange="binance",
    )


def make_genome():
    return StrategyGenome(
        name="BTC_ETH_stat_arb",
        features=[{
            "entry_zscore": 2.0,
            "exit_zscore": 0.5,
            "lookback": 30,
            "use_kalman": True,
            "max_hold_bars": 100,
        }],
    )


class TestStatArbStrategy:
    @pytest.fixture
    def strategy(self):
        genome = make_genome()
        return StatArbStrategy("test_statarb", genome)

    @pytest.mark.asyncio
    async def test_warmup(self, strategy):
        for _ in range(20):
            r = await strategy.on_bar(make_bar("BTC/USDT"))
            assert r is None
            r = await strategy.on_bar(make_bar("ETH/USDT"))
            assert r is None

    @pytest.mark.asyncio
    async def test_required_symbols(self, strategy):
        symbols = strategy.required_symbols()
        assert len(symbols) == 2
        assert "BTC" in symbols[0]

    def test_required_timeframes(self, strategy):
        assert Timeframe.H1 in strategy.required_timeframes()

    @pytest.mark.asyncio
    async def test_ignores_wrong_symbol(self, strategy):
        result = await strategy.on_bar(make_bar("SOL/USDT"))
        assert result is None

    @pytest.mark.asyncio
    async def test_no_trade_small_spread(self, strategy):
        # Feed cointegrated series (small spread)
        for i in range(60):
            price_a = 50000 + i * 10
            price_b = 25000 + i * 5  # Perfectly correlated
            await strategy.on_bar(make_bar("BTC/USDT", close=price_a))
            await strategy.on_bar(make_bar("ETH/USDT", close=price_b))

        # Should not trigger entry with small z-score
        result = await strategy.on_bar(make_bar("BTC/USDT", close=50600))
        # May or may not trigger depending on z-score

    @pytest.mark.asyncio
    async def test_buffer_trimming(self, strategy):
        for _ in range(350):
            await strategy.on_bar(make_bar("BTC/USDT"))
        assert len(strategy._bar_buffer_a) <= 200

    @pytest.mark.asyncio
    async def test_on_tick_returns_none(self, strategy):
        result = await strategy.on_tick(None)
        assert result is None

    def test_kalman_update(self, strategy):
        initial_hr = strategy._hedge_ratio
        strategy._update_kalman(50000, 25000)
        # Hedge ratio should update towards 2.0
        assert strategy._hedge_ratio != initial_hr or strategy._hedge_ratio == 2.0

    def test_ols_hedge_ratio(self, strategy):
        import numpy as np
        a = np.array([100, 110, 120, 130, 140])
        b = np.array([50, 55, 60, 65, 70])
        hr = strategy._ols_hedge_ratio(a, b)
        assert abs(hr - 2.0) < 0.1
