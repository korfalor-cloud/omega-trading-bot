"""Tests for basis trade strategy."""
from __future__ import annotations

import pytest
import asyncio
from datetime import datetime, timezone

from tradingbot.core.enums import Side, SignalType, Timeframe
from tradingbot.core.types import OHLCVBar, StrategyGenome
from tradingbot.strategies.derivatives.basis_trade import BasisTradeStrategy


def make_bar(symbol="BTC/USDT", close=50000.0, open_price=None, volume=100.0):
    if open_price is None:
        open_price = close
    return OHLCVBar(
        timestamp=datetime.now(timezone.utc),
        symbol=symbol,
        timeframe=Timeframe.H1,
        open=open_price,
        high=max(close, open_price) * 1.002,
        low=min(close, open_price) * 0.998,
        close=close,
        volume=volume,
        exchange="binance",
    )


def make_genome():
    return StrategyGenome(
        name="BTC_basis_trade",
        features=[{
            "basis_threshold_pct": 0.10,
            "max_hold_bars": 168,
            "use_volume": False,  # Disable for easier testing
        }],
    )


class TestBasisTradeStrategy:
    @pytest.fixture
    def strategy(self):
        genome = make_genome()
        return BasisTradeStrategy("test_basis", genome)

    @pytest.mark.asyncio
    async def test_warmup(self, strategy):
        for i in range(5):
            result = await strategy.on_bar(make_bar())
            assert result is None

    @pytest.mark.asyncio
    async def test_elevated_basis_long_spot(self, strategy):
        # Warm up
        for _ in range(20):
            await strategy.on_bar(make_bar(close=50000, open_price=50000))

        # Elevated basis: futures (open) >> spot (close)
        # Need annual_basis > 0.10
        # basis_pct = (futures - spot) / spot
        # annual_basis = basis_pct * 365
        # For annual_basis > 0.10, need basis_pct > 0.000274
        # open=50200, close=50000 => basis_pct = 200/50000 = 0.004 => annual = 1.46
        result = await strategy.on_bar(make_bar(close=50000, open_price=50200))
        assert result is not None
        assert result.signal_type == SignalType.ENTRY
        assert result.side == Side.BUY  # Long spot
        assert "annual_basis" in result.metadata
        assert result.metadata["hedge_side"] == "short_futures"

    @pytest.mark.asyncio
    async def test_inverted_basis_short_spot(self, strategy):
        # Warm up
        for _ in range(20):
            await strategy.on_bar(make_bar(close=50000, open_price=50000))

        # Inverted basis: futures (open) << spot (close)
        result = await strategy.on_bar(make_bar(close=50000, open_price=49800))
        assert result is not None
        assert result.signal_type == SignalType.ENTRY
        assert result.side == Side.SELL  # Short spot
        assert result.metadata["hedge_side"] == "long_futures"

    @pytest.mark.asyncio
    async def test_no_signal_small_basis(self, strategy):
        # Warm up
        for _ in range(20):
            await strategy.on_bar(make_bar(close=50000, open_price=50000))

        # Small basis: open very close to close
        result = await strategy.on_bar(make_bar(close=50000, open_price=50001))
        assert result is None

    @pytest.mark.asyncio
    async def test_exit_on_max_hold(self, strategy):
        # Warm up
        for _ in range(20):
            await strategy.on_bar(make_bar())

        # Enter with elevated basis
        await strategy.on_bar(make_bar(close=50000, open_price=50200))
        assert strategy._in_trade is True

        # Hold to max — keep feeding bars with non-zero basis so revert exit doesn't fire
        for _ in range(167):
            await strategy.on_bar(make_bar(close=50000, open_price=50100))

        # Next bar should trigger max_hold exit
        result = await strategy.on_bar(make_bar(close=50000, open_price=50100))
        assert result is not None
        assert result.signal_type == SignalType.EXIT

    @pytest.mark.asyncio
    async def test_exit_on_basis_revert(self, strategy):
        # Warm up
        for _ in range(20):
            await strategy.on_bar(make_bar())

        # Enter with elevated basis
        await strategy.on_bar(make_bar(close=50000, open_price=50200))
        assert strategy._in_trade is True

        # Feed zero basis — should trigger revert exit
        result = await strategy.on_bar(make_bar(close=50000, open_price=50000))
        assert result is not None
        assert result.signal_type == SignalType.EXIT

    @pytest.mark.asyncio
    async def test_buffer_trimming(self, strategy):
        for _ in range(600):
            await strategy.on_bar(make_bar())
        # Trims at 300→200, then grows to ~500, trims again to 200
        assert len(strategy._bar_buffer) <= 300

    def test_required_symbols(self, strategy):
        symbols = strategy.required_symbols()
        assert len(symbols) > 0

    def test_required_timeframes(self, strategy):
        tfs = strategy.required_timeframes()
        assert Timeframe.H1 in tfs
