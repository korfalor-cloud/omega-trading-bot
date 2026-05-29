"""Tests for enhanced paper trading — fills, fees, slippage, P&L, position management."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from tradingbot.core.enums import OrderState, Side
from tradingbot.core.types import Order, Position
from tradingbot.exchanges.paper_exchange import PaperExecutionBackend


class TestPaperExecutionBackendInit:
    def test_default_config(self):
        backend = PaperExecutionBackend({})
        assert backend.slippage_bps == 5.0
        assert backend.commission_bps == 10.0
        assert backend.latency_ms == 50
        assert backend.initial_balance == 100_000.0

    def test_custom_config(self):
        backend = PaperExecutionBackend({
            "slippage_bps": 10.0,
            "commission_bps": 20.0,
            "latency_ms": 0,
            "initial_balance": 50_000.0,
        })
        assert backend.slippage_bps == 10.0
        assert backend.commission_bps == 20.0
        assert backend.initial_balance == 50_000.0

    def test_initial_balance(self):
        backend = PaperExecutionBackend({"initial_balance": 200_000.0})
        assert backend._balances["USDT"] == 200_000.0


class TestPaperBuyOrder:
    @pytest.mark.asyncio
    async def test_buy_order_filled(self):
        backend = PaperExecutionBackend({"slippage_bps": 0, "commission_bps": 0, "latency_ms": 0, "initial_balance": 100_000})
        backend.update_price("BTC/USDT", 50000.0)

        order = Order(symbol="BTC/USDT", side=Side.BUY, quantity=1.0, strategy_id="s1")
        result = await backend.execute(order)

        assert result.state == OrderState.FILLED
        assert result.filled_quantity == 1.0
        assert result.avg_fill_price == 50000.0

    @pytest.mark.asyncio
    async def test_buy_reduces_balance(self):
        backend = PaperExecutionBackend({"slippage_bps": 0, "commission_bps": 0, "latency_ms": 0, "initial_balance": 100_000})
        backend.update_price("BTC/USDT", 50000.0)

        order = Order(symbol="BTC/USDT", side=Side.BUY, quantity=1.0, strategy_id="s1")
        await backend.execute(order)

        balance = await backend.get_balance()
        assert balance["USDT"] == pytest.approx(50000.0)

    @pytest.mark.asyncio
    async def test_buy_creates_position(self):
        backend = PaperExecutionBackend({"slippage_bps": 0, "commission_bps": 0, "latency_ms": 0, "initial_balance": 100_000})
        backend.update_price("BTC/USDT", 50000.0)

        order = Order(symbol="BTC/USDT", side=Side.BUY, quantity=1.0, strategy_id="s1")
        await backend.execute(order)

        positions = await backend.get_positions("s1")
        assert len(positions) == 1
        assert positions[0].quantity == 1.0
        assert positions[0].avg_entry_price == 50000.0

    @pytest.mark.asyncio
    async def test_buy_insufficient_balance_rejected(self):
        backend = PaperExecutionBackend({"slippage_bps": 0, "commission_bps": 0, "latency_ms": 0, "initial_balance": 1000})
        backend.update_price("BTC/USDT", 50000.0)

        order = Order(symbol="BTC/USDT", side=Side.BUY, quantity=1.0, strategy_id="s1")
        result = await backend.execute(order)

        assert result.state == OrderState.REJECTED
        assert "Insufficient" in result.metadata.get("reject_reason", "")


class TestPaperSellOrder:
    @pytest.mark.asyncio
    async def test_sell_after_buy(self):
        backend = PaperExecutionBackend({"slippage_bps": 0, "commission_bps": 0, "latency_ms": 0, "initial_balance": 100_000})
        backend.update_price("BTC/USDT", 50000.0)

        buy = Order(symbol="BTC/USDT", side=Side.BUY, quantity=1.0, strategy_id="s1")
        await backend.execute(buy)

        backend.update_price("BTC/USDT", 55000.0)
        sell = Order(symbol="BTC/USDT", side=Side.SELL, quantity=1.0, strategy_id="s1")
        result = await backend.execute(sell)

        assert result.state == OrderState.FILLED
        positions = await backend.get_positions("s1")
        open_positions = [p for p in positions if p.quantity > 0]
        assert len(open_positions) == 0

    @pytest.mark.asyncio
    async def test_sell_without_position_rejected(self):
        backend = PaperExecutionBackend({"slippage_bps": 0, "commission_bps": 0, "latency_ms": 0, "initial_balance": 100_000})
        backend.update_price("BTC/USDT", 50000.0)

        sell = Order(symbol="BTC/USDT", side=Side.SELL, quantity=1.0, strategy_id="s1")
        result = await backend.execute(sell)

        assert result.state == OrderState.REJECTED

    @pytest.mark.asyncio
    async def test_sell_increases_balance(self):
        backend = PaperExecutionBackend({"slippage_bps": 0, "commission_bps": 0, "latency_ms": 0, "initial_balance": 100_000})
        backend.update_price("BTC/USDT", 50000.0)
        await backend.execute(Order(symbol="BTC/USDT", side=Side.BUY, quantity=1.0, strategy_id="s1"))

        backend.update_price("BTC/USDT", 55000.0)
        await backend.execute(Order(symbol="BTC/USDT", side=Side.SELL, quantity=1.0, strategy_id="s1"))

        balance = await backend.get_balance()
        assert balance["USDT"] == pytest.approx(105000.0)


class TestPaperSlippage:
    @pytest.mark.asyncio
    async def test_buy_slippage_increases_price(self):
        backend = PaperExecutionBackend({"slippage_bps": 10, "commission_bps": 0, "latency_ms": 0, "initial_balance": 100_000})
        backend.update_price("BTC/USDT", 50000.0)

        order = Order(symbol="BTC/USDT", side=Side.BUY, quantity=1.0, strategy_id="s1")
        result = await backend.execute(order)

        # 10 bps slippage: 50000 * 1.001 = 50050
        assert result.avg_fill_price == pytest.approx(50050.0)

    @pytest.mark.asyncio
    async def test_sell_slippage_decreases_price(self):
        backend = PaperExecutionBackend({"slippage_bps": 10, "commission_bps": 0, "latency_ms": 0, "initial_balance": 100_000})
        backend.update_price("BTC/USDT", 50000.0)
        await backend.execute(Order(symbol="BTC/USDT", side=Side.BUY, quantity=1.0, strategy_id="s1"))

        backend.update_price("BTC/USDT", 55000.0)
        result = await backend.execute(Order(symbol="BTC/USDT", side=Side.SELL, quantity=1.0, strategy_id="s1"))

        # 10 bps slippage: 55000 * 0.999 = 54945
        assert result.avg_fill_price == pytest.approx(54945.0)


class TestPaperCommission:
    @pytest.mark.asyncio
    async def test_buy_commission_charged(self):
        backend = PaperExecutionBackend({"slippage_bps": 0, "commission_bps": 10, "latency_ms": 0, "initial_balance": 100_000})
        backend.update_price("BTC/USDT", 50000.0)

        order = Order(symbol="BTC/USDT", side=Side.BUY, quantity=1.0, strategy_id="s1")
        result = await backend.execute(order)

        # 10 bps commission: 50000 * 0.001 = 50
        assert result.commission == pytest.approx(50.0)

    @pytest.mark.asyncio
    async def test_commission_deducted_from_balance(self):
        backend = PaperExecutionBackend({"slippage_bps": 0, "commission_bps": 10, "latency_ms": 0, "initial_balance": 100_000})
        backend.update_price("BTC/USDT", 50000.0)

        await backend.execute(Order(symbol="BTC/USDT", side=Side.BUY, quantity=1.0, strategy_id="s1"))
        balance = await backend.get_balance()
        # Cost = 50000 + 50 commission = 50050
        assert balance["USDT"] == pytest.approx(49950.0)


class TestPaperNoPriceData:
    @pytest.mark.asyncio
    async def test_no_price_rejects_order(self):
        backend = PaperExecutionBackend({"latency_ms": 0, "initial_balance": 100_000})
        order = Order(symbol="UNKNOWN/USDT", side=Side.BUY, quantity=1.0, strategy_id="s1")
        result = await backend.execute(order)
        assert result.state == OrderState.REJECTED
        assert "No price data" in result.metadata.get("reject_reason", "")


class TestPaperCancelOrder:
    @pytest.mark.asyncio
    async def test_cancel_order(self):
        backend = PaperExecutionBackend({})
        order = Order(symbol="BTC/USDT", side=Side.BUY, quantity=1.0)
        result = await backend.cancel(order)
        assert result.state == OrderState.CANCELLED


class TestPaperMultipleBuys:
    @pytest.mark.asyncio
    async def test_averaging_into_position(self):
        backend = PaperExecutionBackend({"slippage_bps": 0, "commission_bps": 0, "latency_ms": 0, "initial_balance": 200_000})
        backend.update_price("BTC/USDT", 50000.0)
        await backend.execute(Order(symbol="BTC/USDT", side=Side.BUY, quantity=1.0, strategy_id="s1"))

        backend.update_price("BTC/USDT", 60000.0)
        await backend.execute(Order(symbol="BTC/USDT", side=Side.BUY, quantity=1.0, strategy_id="s1"))

        positions = await backend.get_positions("s1")
        assert len(positions) == 1
        assert positions[0].quantity == 2.0
        # Average: (50000*1 + 60000*1) / 2 = 55000
        assert positions[0].avg_entry_price == pytest.approx(55000.0)
