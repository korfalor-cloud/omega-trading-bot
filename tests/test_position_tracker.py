"""Tests for position tracker."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from tradingbot.portfolio.position_tracker import PositionInfo, PositionTracker


class TestPositionTracker:
    @pytest.fixture
    def tracker(self):
        return PositionTracker(config={"initial_capital": 100000})

    def test_open_long(self, tracker):
        pos = tracker.update_position("BTC/USDT", "buy", 0.1, 50000, fee=5)
        assert pos.quantity == 0.1
        assert pos.side == "long"
        assert pos.average_entry == 50000

    def test_open_short(self, tracker):
        pos = tracker.update_position("BTC/USDT", "sell", 0.1, 50000, fee=5)
        assert pos.quantity == -0.1
        assert pos.side == "short"

    def test_add_to_position(self, tracker):
        tracker.update_position("BTC/USDT", "buy", 0.1, 50000)
        pos = tracker.update_position("BTC/USDT", "buy", 0.1, 52000)
        assert pos.quantity == 0.2
        assert pos.average_entry == 51000

    def test_reduce_position_realize_pnl(self, tracker):
        tracker.update_position("BTC/USDT", "buy", 0.2, 50000)
        pos = tracker.update_position("BTC/USDT", "sell", 0.1, 55000)
        assert pos.quantity == 0.1
        assert pos.realized_pnl == pytest.approx(500, abs=1)  # 0.1 * (55000-50000)

    def test_unrealized_pnl(self, tracker):
        tracker.update_position("BTC/USDT", "buy", 0.1, 50000)
        tracker.update_price("BTC/USDT", 55000)
        pos = tracker.get_position("BTC/USDT")
        assert pos.unrealized_pnl == pytest.approx(500, abs=1)

    def test_unrealized_pnl_short(self, tracker):
        tracker.update_position("BTC/USDT", "sell", 0.1, 50000)
        tracker.update_price("BTC/USDT", 45000)
        pos = tracker.get_position("BTC/USDT")
        assert pos.unrealized_pnl == pytest.approx(500, abs=1)

    def test_close_position(self, tracker):
        tracker.update_position("BTC/USDT", "buy", 0.1, 50000)
        pos = tracker.update_position("BTC/USDT", "sell", 0.1, 55000)
        assert pos.quantity == 0
        assert pos.side == "flat"
        assert pos.realized_pnl == pytest.approx(500, abs=1)

    def test_portfolio_value(self, tracker):
        tracker.update_position("BTC/USDT", "buy", 0.1, 50000)
        value = tracker.get_portfolio_value()
        assert value == 100000  # No P&L yet at entry

    def test_portfolio_summary(self, tracker):
        tracker.update_position("BTC/USDT", "buy", 0.1, 50000, fee=5)
        tracker.update_price("BTC/USDT", 55000)
        summary = tracker.get_portfolio_summary()
        assert summary["n_positions"] == 1
        assert "BTC/USDT" in summary["positions"]

    def test_get_all_positions(self, tracker):
        tracker.update_position("BTC/USDT", "buy", 0.1, 50000)
        tracker.update_position("ETH/USDT", "buy", 1.0, 3000)
        positions = tracker.get_all_positions()
        assert len(positions) == 2

    def test_equity_curve(self, tracker):
        tracker.record_equity()
        tracker.update_position("BTC/USDT", "buy", 0.1, 50000)
        tracker.record_equity()
        curve = tracker.get_equity_curve()
        assert len(curve) == 2

    def test_max_drawdown(self, tracker):
        tracker.record_equity()
        tracker.update_position("BTC/USDT", "buy", 0.1, 50000)
        tracker.update_price("BTC/USDT", 45000)
        tracker.record_equity()
        dd = tracker.get_max_drawdown()
        assert dd > 0

    def test_trade_log(self, tracker):
        tracker.update_position("BTC/USDT", "buy", 0.1, 50000)
        tracker.update_position("ETH/USDT", "buy", 1.0, 3000)
        log = tracker.get_trade_log()
        assert len(log) == 2
        assert log[0]["symbol"] == "BTC/USDT"

    def test_trade_log_by_symbol(self, tracker):
        tracker.update_position("BTC/USDT", "buy", 0.1, 50000)
        tracker.update_position("ETH/USDT", "buy", 1.0, 3000)
        log = tracker.get_trade_log("BTC/USDT")
        assert len(log) == 1

    def test_net_pnl(self, tracker):
        tracker.update_position("BTC/USDT", "buy", 0.1, 50000, fee=10)
        tracker.update_price("BTC/USDT", 55000)
        pos = tracker.get_position("BTC/USDT")
        assert pos.net_pnl == pytest.approx(490, abs=1)  # 500 unrealized - 10 fees
