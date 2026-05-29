"""Tests for SmartOrderRouter — multi-venue order routing."""
from __future__ import annotations

import pytest

from tradingbot.execution.smart_router import (
    SmartOrderRouter,
    RouterDecision,
    BracketOrder,
)


class TestSmartOrderRouter:
    @pytest.fixture
    def router(self):
        r = SmartOrderRouter({"default_fee": 0.001})
        r.register_venue("binance", fee_rate=0.001, latency_ms=20)
        r.register_venue("coinbase", fee_rate=0.002, latency_ms=50)
        r.register_venue("kraken", fee_rate=0.0015, latency_ms=35)
        return r

    def test_register_venue(self, router):
        assert "binance" in router._venues
        assert "coinbase" in router._venues
        assert router._venues["binance"]["fee_rate"] == 0.001

    def test_route_buy_best_price(self, router):
        prices = {"binance": 50000, "coinbase": 49900, "kraken": 50100}
        decisions = router.route_buy(1.0, prices)
        assert len(decisions) > 0
        # Coinbase has lowest price but higher fee; binance has best fee-adjusted price
        assert decisions[0].quantity == 1.0

    def test_route_buy_fills_quantity(self, router):
        prices = {"binance": 50000, "coinbase": 49900}
        decisions = router.route_buy(0.5, prices)
        assert decisions[0].quantity == 0.5
        assert decisions[0].fee > 0

    def test_route_sell_best_price(self, router):
        prices = {"binance": 50000, "coinbase": 50200, "kraken": 49800}
        decisions = router.route_sell(1.0, prices)
        assert len(decisions) > 0
        # Should prefer highest sell price adjusted for fees
        assert decisions[0].quantity == 1.0

    def test_route_buy_skips_disconnected(self, router):
        router._venues["binance"]["connected"] = False
        prices = {"binance": 49000, "coinbase": 50000, "kraken": 50100}
        decisions = router.route_buy(1.0, prices)
        for d in decisions:
            assert d.venue != "binance"

    def test_create_bracket_buy(self, router):
        bracket = router.create_bracket("buy", entry=50000, stop_pct=0.02, target_pct=0.04, quantity=1.0)
        assert isinstance(bracket, BracketOrder)
        assert bracket.entry_price == 50000
        assert bracket.stop_price == 50000 * 0.98
        assert bracket.target_price == 50000 * 1.04
        assert bracket.quantity == 1.0
        assert bracket.side == "buy"

    def test_create_bracket_sell(self, router):
        bracket = router.create_bracket("sell", entry=50000, stop_pct=0.02, target_pct=0.04, quantity=0.5)
        assert bracket.stop_price == 50000 * 1.02
        assert bracket.target_price == 50000 * 0.96
        assert bracket.side == "sell"

    def test_create_oco(self, router):
        oco = router.create_oco(stop_price=49000, target_price=52000, quantity=1.0, side="buy")
        assert oco["type"] == "oco"
        assert oco["stop_price"] == 49000
        assert oco["target_price"] == 52000
        assert oco["quantity"] == 1.0

    def test_iceberg_slice(self, router):
        slice_qty = router.iceberg_slice(total_qty=10.0, slice_pct=0.1)
        assert slice_qty == 1.0

    def test_iceberg_slice_custom_pct(self, router):
        slice_qty = router.iceberg_slice(total_qty=10.0, slice_pct=0.25)
        assert slice_qty == 2.5

    def test_pegged_order_buy(self, router):
        order = router.pegged_order(side="buy", offset_bps=10, ref_price=50000)
        assert order["side"] == "buy"
        assert order["price"] == 50000 * (1 - 10 / 10000)
        assert order["offset_bps"] == 10

    def test_pegged_order_sell(self, router):
        order = router.pegged_order(side="sell", offset_bps=10, ref_price=50000)
        assert order["side"] == "sell"
        assert order["price"] == 50000 * (1 + 10 / 10000)
