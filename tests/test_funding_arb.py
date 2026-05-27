"""Tests for funding rate arbitrage strategy."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from tradingbot.core.enums import Side, SignalType, Timeframe
from tradingbot.core.types import OHLCVBar, StrategyGenome
from tradingbot.strategies.derivatives.funding_arb import FundingRateArbitrage


def make_bar(symbol="BTC/USDT", close=50000.0, volume=100.0, vwap=0.0):
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
        vwap=vwap,
    )


def make_genome():
    return StrategyGenome(
        name="BTC_funding_arb",
        features=[{
            "funding_threshold": 0.0003,
            "max_hold_bars": 72,
            "use_trend_filter": False,  # Disable for easier testing
        }],
    )


class TestFundingRateArbitrage:
    @pytest.fixture
    def strategy(self):
        genome = make_genome()
        return FundingRateArbitrage("test_funding", genome)

    @pytest.mark.asyncio
    async def test_warmup_period(self, strategy):
        # Not enough bars yet
        for i in range(5):
            result = await strategy.on_bar(make_bar())
            assert result is None

    @pytest.mark.asyncio
    async def test_positive_funding_signal(self, strategy):
        # Warm up
        for _ in range(30):
            await strategy.on_bar(make_bar())

        # Feed positive funding rate (via vwap field)
        for _ in range(5):
            result = await strategy.on_bar(make_bar(vwap=0.001))
            if result is not None:
                break

        assert result is not None
        assert result.signal_type == SignalType.ENTRY
        assert result.side == Side.SELL  # Short perp
        assert "funding_rate" in result.metadata

    @pytest.mark.asyncio
    async def test_negative_funding_signal(self, strategy):
        # Warm up
        for _ in range(30):
            await strategy.on_bar(make_bar())

        # Feed negative funding rate
        for _ in range(5):
            result = await strategy.on_bar(make_bar(vwap=-0.001))
            if result is not None:
                break

        assert result is not None
        assert result.signal_type == SignalType.ENTRY
        assert result.side == Side.BUY  # Long perp

    @pytest.mark.asyncio
    async def test_no_signal_below_threshold(self, strategy):
        # Warm up
        for _ in range(30):
            await strategy.on_bar(make_bar())

        # Feed low funding rate (below threshold)
        for _ in range(5):
            result = await strategy.on_bar(make_bar(vwap=0.0001))
            assert result is None

    @pytest.mark.asyncio
    async def test_exit_on_max_hold(self, strategy):
        # Warm up
        for _ in range(30):
            await strategy.on_bar(make_bar())

        # Enter trade with high funding
        for _ in range(5):
            result = await strategy.on_bar(make_bar(vwap=0.001))
            if result is not None:
                break
        assert strategy._in_trade is True

        # Hold — keep feeding non-zero funding so revert exit doesn't fire
        for _ in range(71):
            await strategy.on_bar(make_bar(vwap=0.0008))

        # Next bar should trigger max_hold exit
        result = await strategy.on_bar(make_bar(vwap=0.0008))
        assert result is not None
        assert result.signal_type == SignalType.EXIT

    @pytest.mark.asyncio
    async def test_exit_on_funding_revert(self, strategy):
        # Warm up
        for _ in range(30):
            await strategy.on_bar(make_bar())

        # Enter trade with high funding
        for _ in range(5):
            result = await strategy.on_bar(make_bar(vwap=0.001))
            if result is not None:
                break
        assert strategy._in_trade is True

        # Feed zero funding — should trigger revert exit
        for _ in range(5):
            result = await strategy.on_bar(make_bar(vwap=0.0))
            if result is not None:
                break

        assert result is not None
        assert result.signal_type == SignalType.EXIT

    def test_required_symbols(self, strategy):
        symbols = strategy.required_symbols()
        assert "BTC" in symbols[0]

    def test_required_timeframes(self, strategy):
        tfs = strategy.required_timeframes()
        assert Timeframe.H1 in tfs

    @pytest.mark.asyncio
    async def test_on_tick_returns_none(self, strategy):
        result = await strategy.on_tick(None)
        assert result is None
