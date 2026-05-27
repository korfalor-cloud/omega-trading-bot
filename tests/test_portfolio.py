"""Tests for portfolio management."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tradingbot.core.enums import Side
from tradingbot.core.types import Fill, Position
from tradingbot.portfolio.portfolio_manager import PortfolioManager


class TestPortfolioManager:
    @pytest.fixture
    def pm(self):
        return PortfolioManager(initial_cash=100000.0)

    @pytest.fixture
    def buy_fill(self):
        return Fill(
            order_id="order-1",
            symbol="BTC/USDT",
            side=Side.BUY,
            price=50000.0,
            quantity=0.1,
            commission=5.0,
            exchange="binance",
            timestamp=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def sell_fill(self):
        return Fill(
            order_id="order-2",
            symbol="BTC/USDT",
            side=Side.SELL,
            price=50000.0,
            quantity=0.1,
            commission=5.0,
            exchange="binance",
            timestamp=datetime.now(timezone.utc),
        )

    def test_initial_state(self, pm):
        assert pm.cash == 100000.0
        assert pm.positions == []
        assert pm.realized_pnl == 0.0

    def test_open_position(self, pm, buy_fill):
        pm.apply_fill(buy_fill, "strat-1")
        assert len(pm.positions) == 1
        assert pm.positions[0].symbol == "BTC/USDT"
        assert pm.positions[0].quantity == 0.1
        assert pm.positions[0].avg_entry_price == 50000.0
        assert pm.cash < 100000.0

    def test_close_position(self, pm, buy_fill, sell_fill):
        pm.apply_fill(buy_fill, "strat-1")
        pm.apply_fill(sell_fill, "strat-1")
        assert len(pm.positions) == 0
        assert pm.realized_pnl != 0 or pm.total_commission > 0

    def test_partial_close(self, pm, buy_fill):
        pm.apply_fill(buy_fill, "strat-1")

        partial_sell = Fill(
            order_id="order-3",
            symbol="BTC/USDT",
            side=Side.SELL,
            price=51000.0,
            quantity=0.05,
            commission=2.5,
            exchange="binance",
            timestamp=datetime.now(timezone.utc),
        )
        pm.apply_fill(partial_sell, "strat-1")

        assert len(pm.positions) == 1
        assert pm.positions[0].quantity == 0.05

    def test_price_update(self, pm, buy_fill):
        pm.apply_fill(buy_fill, "strat-1")
        pm.update_prices({"BTC/USDT": 52000.0})
        pos = pm.positions[0]
        assert pos.current_price == 52000.0
        assert pos.unrealized_pnl > 0

    def test_portfolio_state(self, pm, buy_fill):
        pm.apply_fill(buy_fill, "strat-1")
        pm.update_prices({"BTC/USDT": 51000.0})
        state = pm.get_portfolio_state()

        assert state.total_equity > 0
        assert state.cash > 0
        assert state.positions_value > 0
        assert state.unrealized_pnl > 0  # Price went up
        assert state.timestamp is not None

    def test_multiple_positions(self, pm):
        fill1 = Fill(
            order_id="o1", symbol="BTC/USDT", side=Side.BUY,
            price=50000.0, quantity=0.1, commission=5.0,
            exchange="binance", timestamp=datetime.now(timezone.utc),
        )
        fill2 = Fill(
            order_id="o2", symbol="ETH/USDT", side=Side.BUY,
            price=3000.0, quantity=1.0, commission=3.0,
            exchange="binance", timestamp=datetime.now(timezone.utc),
        )
        pm.apply_fill(fill1, "strat-1")
        pm.apply_fill(fill2, "strat-1")

        assert len(pm.positions) == 2

    def test_get_positions_for_symbol(self, pm, buy_fill):
        pm.apply_fill(buy_fill, "strat-1")
        positions = pm.get_positions_for_symbol("BTC/USDT")
        assert len(positions) == 1

    def test_get_positions_for_strategy(self, pm, buy_fill):
        pm.apply_fill(buy_fill, "strat-1")
        positions = pm.get_positions_for_strategy("strat-1")
        assert len(positions) == 1

    def test_trade_history(self, pm, buy_fill, sell_fill):
        pm.apply_fill(buy_fill, "strat-1")
        pm.apply_fill(sell_fill, "strat-1")
        history = pm.get_trade_history()
        assert len(history) == 1
        assert history[0]["symbol"] == "BTC/USDT"

    def test_reset(self, pm, buy_fill):
        pm.apply_fill(buy_fill, "strat-1")
        pm.reset()
        assert pm.cash == 100000.0
        assert pm.positions == []
        assert pm.realized_pnl == 0.0

    def test_long_position_pnl(self, pm, buy_fill):
        pm.apply_fill(buy_fill, "strat-1")
        pm.update_prices({"BTC/USDT": 55000.0})
        state = pm.get_portfolio_state()
        # Profit: (55000 - 50000) * 0.1 = 500
        assert abs(state.unrealized_pnl - 500.0) < 1.0

    def test_short_position_pnl(self, pm, sell_fill):
        pm.apply_fill(sell_fill, "strat-1")
        pm.update_prices({"BTC/USDT": 45000.0})
        state = pm.get_portfolio_state()
        # Profit: (50000 - 45000) * 0.1 = 500
        assert abs(state.unrealized_pnl - 500.0) < 1.0

    def test_drawdown_tracking(self, pm, buy_fill):
        pm.apply_fill(buy_fill, "strat-1")
        # Price goes up
        pm.update_prices({"BTC/USDT": 55000.0})
        state1 = pm.get_portfolio_state()
        # Price goes down
        pm.update_prices({"BTC/USDT": 40000.0})
        state2 = pm.get_portfolio_state()
        assert state2.current_drawdown > 0
