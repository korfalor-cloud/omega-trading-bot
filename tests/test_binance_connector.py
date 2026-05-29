"""Tests for exchange connectors — CCXTAdapter and BaseExchangeConnector."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tradingbot.core.enums import OrderState, OrderType, Side, Timeframe
from tradingbot.core.errors import ExchangeError, OrderError
from tradingbot.core.types import Order
from tradingbot.exchanges.base_connector import (
    BalanceInfo,
    BaseExchangeConnector,
    ConnectionState,
    PositionInfo,
    RateLimitRule,
    TokenBucketRateLimiter,
)
from tradingbot.exchanges.ccxt_adapter import CCXTAdapter, TIMEFRAME_MAP


# ---------------------------------------------------------------------------
# TokenBucketRateLimiter
# ---------------------------------------------------------------------------

class TestTokenBucketRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_within_capacity(self):
        limiter = TokenBucketRateLimiter(max_tokens=10, refill_rate=10)
        await limiter.acquire(5)
        assert limiter.tokens == pytest.approx(5.0, abs=1.0)

    @pytest.mark.asyncio
    async def test_refill(self):
        limiter = TokenBucketRateLimiter(max_tokens=100, refill_rate=1000)
        await limiter.acquire(100)
        assert limiter.tokens == pytest.approx(0.0, abs=1.0)
        # After sleeping a bit, tokens should refill
        await asyncio.sleep(0.05)
        async with limiter._lock:
            limiter._refill()
        assert limiter.tokens > 0


# ---------------------------------------------------------------------------
# ConnectionState and PositionInfo
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_connection_state_values(self):
        assert ConnectionState.DISCONNECTED is not None
        assert ConnectionState.CONNECTING is not None
        assert ConnectionState.CONNECTED is not None
        assert ConnectionState.RECONNECTING is not None
        assert ConnectionState.CLOSED is not None

    def test_balance_info_available(self):
        b = BalanceInfo(asset="BTC", free=1.5, locked=0.5, total=2.0)
        assert b.available == 1.5

    def test_position_info_notional(self):
        p = PositionInfo(symbol="BTC/USDT", quantity=2.0, current_price=50000.0)
        assert p.notional_value == pytest.approx(100000.0)

    def test_position_info_pnl_pct_long(self):
        p = PositionInfo(symbol="BTC/USDT", side=Side.BUY, avg_entry_price=50000.0, current_price=55000.0)
        assert p.pnl_pct == pytest.approx(0.1)

    def test_position_info_pnl_pct_short(self):
        p = PositionInfo(symbol="BTC/USDT", side=Side.SELL, avg_entry_price=50000.0, current_price=45000.0)
        assert p.pnl_pct == pytest.approx(0.1)

    def test_position_info_pnl_pct_zero_entry(self):
        p = PositionInfo(symbol="BTC/USDT", avg_entry_price=0.0, current_price=55000.0)
        assert p.pnl_pct == 0.0

    def test_rate_limit_rule_defaults(self):
        r = RateLimitRule()
        assert r.max_requests == 1200
        assert r.window_seconds == 60.0


# ---------------------------------------------------------------------------
# CCXTAdapter
# ---------------------------------------------------------------------------

class TestCCXTAdapter:
    def test_name_property(self):
        adapter = CCXTAdapter("binance", api_key="k", api_secret="s")
        assert adapter.name == "binance"

    def test_is_connected_default(self):
        adapter = CCXTAdapter("binance")
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_with_mock_ccxt(self):
        adapter = CCXTAdapter("binance", testnet=True)
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()

        # Build a mock ccxt.async_support module where the exchange class
        # constructor returns our mock_exchange
        mock_exchange_class = MagicMock(return_value=mock_exchange)
        mock_ccxt_module = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class

        import sys
        real_async_support = sys.modules.get("ccxt.async_support")
        sys.modules["ccxt.async_support"] = mock_ccxt_module
        try:
            await adapter.connect()
            assert adapter.is_connected is True
            mock_exchange.load_markets.assert_awaited_once()
        finally:
            if real_async_support is not None:
                sys.modules["ccxt.async_support"] = real_async_support
            else:
                sys.modules.pop("ccxt.async_support", None)

    @pytest.mark.asyncio
    async def test_connect_unsupported_exchange(self):
        adapter = CCXTAdapter("nonexistent_exchange")
        mock_ccxt_module = MagicMock(spec=[])  # no attributes

        import sys
        real_async_support = sys.modules.get("ccxt.async_support")
        sys.modules["ccxt.async_support"] = mock_ccxt_module
        try:
            with pytest.raises(ExchangeError):
                await adapter.connect()
        finally:
            if real_async_support is not None:
                sys.modules["ccxt.async_support"] = real_async_support
            else:
                sys.modules.pop("ccxt.async_support", None)

    @pytest.mark.asyncio
    async def test_disconnect(self):
        adapter = CCXTAdapter("binance")
        mock_exchange = AsyncMock()
        adapter._exchange = mock_exchange
        adapter._connected = True
        await adapter.disconnect()
        assert adapter.is_connected is False
        mock_exchange.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_candles(self):
        adapter = CCXTAdapter("binance")
        raw_candles = [
            [1704067200000, 50000, 51000, 49000, 50500, 100.0],
            [1704070800000, 50500, 52000, 50000, 51500, 120.0],
        ]
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=raw_candles)
        adapter._exchange = mock_exchange

        bars = await adapter.fetch_candles("BTC/USDT", Timeframe.H1)
        assert len(bars) == 2
        assert bars[0].open == 50000
        assert bars[0].close == 50500
        assert bars[1].high == 52000

    @pytest.mark.asyncio
    async def test_submit_order_success(self):
        adapter = CCXTAdapter("binance")
        mock_exchange = AsyncMock()
        mock_exchange.create_order = AsyncMock(return_value={"id": "12345"})
        adapter._exchange = mock_exchange

        order = Order(symbol="BTC/USDT", side=Side.BUY, order_type=OrderType.MARKET, quantity=0.1)
        result = await adapter.submit_order(order)
        assert result.state == OrderState.SUBMITTED
        assert result.exchange_order_id == "12345"

    @pytest.mark.asyncio
    async def test_submit_order_failure(self):
        adapter = CCXTAdapter("binance")
        mock_exchange = AsyncMock()
        mock_exchange.create_order = AsyncMock(side_effect=Exception("Insufficient balance"))
        adapter._exchange = mock_exchange

        order = Order(symbol="BTC/USDT", side=Side.BUY, quantity=0.1)
        with pytest.raises(OrderError):
            await adapter.submit_order(order)
        assert order.state == OrderState.REJECTED

    @pytest.mark.asyncio
    async def test_cancel_order(self):
        adapter = CCXTAdapter("binance")
        mock_exchange = AsyncMock()
        mock_exchange.cancel_order = AsyncMock(return_value={})
        adapter._exchange = mock_exchange

        result = await adapter.cancel_order("order-123", "BTC/USDT")
        assert result.state == OrderState.CANCELLED

    @pytest.mark.asyncio
    async def test_fetch_positions(self):
        adapter = CCXTAdapter("binance")
        mock_exchange = AsyncMock()
        mock_exchange.fetch_positions = AsyncMock(return_value=[
            {"symbol": "BTC/USDT", "side": "long", "contracts": 0.5, "entryPrice": 50000, "markPrice": 51000, "unrealizedPnl": 500},
            {"symbol": "ETH/USDT", "side": "short", "contracts": 0, "entryPrice": 0, "markPrice": 0, "unrealizedPnl": 0},
        ])
        adapter._exchange = mock_exchange

        positions = await adapter.fetch_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "BTC/USDT"
        assert positions[0].quantity == 0.5

    def test_timeframe_map_completeness(self):
        assert Timeframe.M1 in TIMEFRAME_MAP
        assert Timeframe.H1 in TIMEFRAME_MAP
        assert Timeframe.D1 in TIMEFRAME_MAP
        assert TIMEFRAME_MAP[Timeframe.H1] == "1h"
