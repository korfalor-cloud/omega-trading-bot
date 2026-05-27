"""Tests for order management system."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from tradingbot.execution.order_manager import (
    Fill,
    ManagedOrder,
    OrderManager,
    OrderStatus,
)


class TestOrderManager:
    @pytest.fixture
    def oms(self):
        return OrderManager(config={"max_open_orders": 10, "default_fee_rate": 0.001})

    def test_create_order(self, oms):
        order = oms.create_order("BTC/USDT", "buy", 0.1, price=50000)
        assert order.status == OrderStatus.SUBMITTED
        assert order.symbol == "BTC/USDT"
        assert order.quantity == 0.1

    def test_cancel_order(self, oms):
        order = oms.create_order("BTC/USDT", "buy", 0.1, price=50000)
        result = oms.cancel_order(order.id)
        assert result is True
        assert order.status == OrderStatus.CANCELLED

    def test_cancel_nonexistent(self, oms):
        assert oms.cancel_order("nonexistent") is False

    def test_amend_order(self, oms):
        order = oms.create_order("BTC/USDT", "buy", 0.1, price=50000)
        result = oms.amend_order(order.id, new_price=51000)
        assert result is True
        assert order.price == 51000

    def test_amend_cancelled_order(self, oms):
        order = oms.create_order("BTC/USDT", "buy", 0.1, price=50000)
        oms.cancel_order(order.id)
        assert oms.amend_order(order.id, new_price=51000) is False

    def test_process_fill_full(self, oms):
        order = oms.create_order("BTC/USDT", "buy", 0.1, price=50000)
        fill = oms.process_fill(order.id, price=50000, quantity=0.1)
        assert fill is not None
        assert fill.price == 50000
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 0.1

    def test_process_fill_partial(self, oms):
        order = oms.create_order("BTC/USDT", "buy", 0.1, price=50000)
        oms.process_fill(order.id, price=50000, quantity=0.05)
        assert order.status == OrderStatus.PARTIAL
        assert order.remaining_quantity == 0.05

    def test_process_fill_complete_after_partial(self, oms):
        order = oms.create_order("BTC/USDT", "buy", 0.1, price=50000)
        oms.process_fill(order.id, price=50000, quantity=0.05)
        oms.process_fill(order.id, price=50100, quantity=0.05)
        assert order.status == OrderStatus.FILLED
        assert order.average_fill_price == pytest.approx(50050, abs=1)

    def test_fill_auto_fee(self, oms):
        order = oms.create_order("BTC/USDT", "buy", 1.0, price=50000)
        fill = oms.process_fill(order.id, price=50000, quantity=1.0)
        assert fill.fee == pytest.approx(50.0, abs=1)  # 0.1% of 50000

    def test_max_open_orders(self, oms):
        for _ in range(10):
            oms.create_order("BTC/USDT", "buy", 0.1, price=50000)
        with pytest.raises(ValueError):
            oms.create_order("BTC/USDT", "buy", 0.1, price=50000)

    def test_get_active_orders(self, oms):
        oms.create_order("BTC/USDT", "buy", 0.1, price=50000)
        oms.create_order("ETH/USDT", "buy", 1.0, price=3000)
        active = oms.get_active_orders()
        assert len(active) == 2

    def test_get_active_orders_by_symbol(self, oms):
        oms.create_order("BTC/USDT", "buy", 0.1, price=50000)
        oms.create_order("ETH/USDT", "buy", 1.0, price=3000)
        active = oms.get_active_orders("BTC/USDT")
        assert len(active) == 1

    def test_get_position(self, oms):
        order = oms.create_order("BTC/USDT", "buy", 0.1, price=50000)
        oms.process_fill(order.id, price=50000, quantity=0.1)
        pos = oms.get_position("BTC/USDT")
        assert pos["net_quantity"] == 0.1
        assert pos["average_price"] == 50000

    def test_cancel_all(self, oms):
        oms.create_order("BTC/USDT", "buy", 0.1, price=50000)
        oms.create_order("ETH/USDT", "buy", 1.0, price=3000)
        cancelled = oms.cancel_all()
        assert cancelled == 2

    def test_cancel_all_by_symbol(self, oms):
        oms.create_order("BTC/USDT", "buy", 0.1, price=50000)
        oms.create_order("ETH/USDT", "buy", 1.0, price=3000)
        cancelled = oms.cancel_all("BTC/USDT")
        assert cancelled == 1

    def test_get_stats(self, oms):
        oms.create_order("BTC/USDT", "buy", 0.1, price=50000)
        order = oms.create_order("ETH/USDT", "buy", 1.0, price=3000)
        oms.process_fill(order.id, price=3000, quantity=1.0)
        stats = oms.get_stats()
        assert stats["total_orders"] == 2
        assert stats["filled_orders"] == 1
        assert stats["active_orders"] == 1
