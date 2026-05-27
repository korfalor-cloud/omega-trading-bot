"""Tests for realistic backtester."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from tradingbot.backtesting.realistic import (
    BacktestConfig,
    RealisticBacktester,
    SimulatedFill,
)


class TestRealisticBacktester:
    @pytest.fixture
    def bt(self):
        return RealisticBacktester(BacktestConfig(
            initial_capital=100000,
            taker_fee=0.0005,
            slippage_bps=5.0,
        ))

    def test_market_order_buy(self, bt):
        fill = bt.simulate_market_order("BTC/USDT", "buy", 0.1, 50000)
        assert fill is not None
        assert fill.quantity == 0.1
        assert fill.fill_price > 50000  # Slippage
        assert bt.get_position("BTC/USDT") == 0.1

    def test_market_order_sell(self, bt):
        bt.simulate_market_order("BTC/USDT", "buy", 0.1, 50000)
        fill = bt.simulate_market_order("BTC/USDT", "sell", 0.1, 55000)
        assert fill is not None
        assert bt.get_position("BTC/USDT") == 0

    def test_slippage_buy(self, bt):
        fill = bt.simulate_market_order("BTC/USDT", "buy", 0.1, 50000)
        # 5 bps slippage on 50000 = 2.5
        assert fill.fill_price == pytest.approx(50002.5, abs=5)

    def test_slippage_sell(self, bt):
        fill = bt.simulate_market_order("BTC/USDT", "sell", 0.1, 50000)
        assert fill.fill_price < 50000

    def test_fee_deduction(self, bt):
        fill = bt.simulate_market_order("BTC/USDT", "buy", 1.0, 50000)
        assert fill.fee > 0
        assert fill.fee == pytest.approx(50000 * 1.0 * 0.0005, abs=10)

    def test_limit_order_fill(self, bt):
        fill = bt.simulate_limit_order("BTC/USDT", "buy", 0.1, 50000, 49500)
        # Market at 49500, limit at 50000 — should fill
        assert fill is not None
        assert fill.fill_price == 50000
        assert fill.slippage == 0  # No slippage on limit

    def test_limit_order_no_fill(self, bt):
        # Limit buy at 49000, market at 50000 — won't fill
        fill = bt.simulate_limit_order("BTC/USDT", "buy", 0.1, 49000, 50000)
        assert fill is None

    def test_funding_rate(self, bt):
        bt.simulate_market_order("BTC/USDT", "buy", 1.0, 50000)
        cost = bt.apply_funding("BTC/USDT", datetime.now(timezone.utc))
        assert cost > 0  # Long pays positive funding

    def test_equity_calculation(self, bt):
        bt.simulate_market_order("BTC/USDT", "buy", 0.1, 50000)
        equity = bt.get_equity({"BTC/USDT": 55000})
        # Capital - (0.1 * 50000 + fee) + (0.1 * 55000)
        assert equity > 100000

    def test_equity_curve(self, bt):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bt.record_equity(ts, {})
        bt.simulate_market_order("BTC/USDT", "buy", 0.1, 50000)
        bt.record_equity(ts, {"BTC/USDT": 55000})
        curve = bt.get_equity_curve()
        assert len(curve) == 2

    def test_stats(self, bt):
        bt.simulate_market_order("BTC/USDT", "buy", 0.1, 50000)
        bt.record_equity(datetime(2024, 1, 1, tzinfo=timezone.utc), {"BTC/USDT": 50000})
        bt.simulate_market_order("BTC/USDT", "sell", 0.1, 55000)
        bt.record_equity(datetime(2024, 1, 2, tzinfo=timezone.utc), {"BTC/USDT": 55000})
        stats = bt.get_stats()
        assert "total_return" in stats
        assert "total_fees" in stats
        assert "total_slippage_cost" in stats

    def test_capital_constraint(self, bt):
        # Try to buy more than capital allows
        fill = bt.simulate_market_order("BTC/USDT", "buy", 100, 50000)
        assert fill is not None
        assert fill.quantity < 100  # Should be reduced

    def test_multiple_symbols(self, bt):
        bt.simulate_market_order("BTC/USDT", "buy", 0.1, 50000)
        bt.simulate_market_order("ETH/USDT", "buy", 1.0, 3000)
        assert bt.get_position("BTC/USDT") == 0.1
        assert bt.get_position("ETH/USDT") == 1.0

    def test_custom_config(self):
        config = BacktestConfig(maker_fee=0.0001, slippage_model="fixed")
        bt = RealisticBacktester(config)
        assert bt.config.maker_fee == 0.0001
