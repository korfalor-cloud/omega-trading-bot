"""Tests for PositionMonitor — real-time position P&L tracking."""
from __future__ import annotations

import pytest

from tradingbot.monitoring.position_monitor import PositionMonitor, PositionPnL


class TestPositionMonitor:
    @pytest.fixture
    def monitor(self):
        return PositionMonitor({"initial_equity": 100000})

    def test_update_position_long_profit(self, monitor):
        pos = monitor.update_position("BTC/USDT", "buy", 50000.0, 1.0, 51000.0)
        assert isinstance(pos, PositionPnL)
        assert pos.unrealized_pnl == 1000.0
        assert pos.unrealized_pnl_pct == pytest.approx(0.02)

    def test_update_position_long_loss(self, monitor):
        pos = monitor.update_position("BTC/USDT", "buy", 50000.0, 1.0, 49000.0)
        assert pos.unrealized_pnl == -1000.0

    def test_update_position_short_profit(self, monitor):
        pos = monitor.update_position("BTC/USDT", "sell", 50000.0, 1.0, 49000.0)
        assert pos.unrealized_pnl == 1000.0

    def test_update_position_short_loss(self, monitor):
        pos = monitor.update_position("BTC/USDT", "sell", 50000.0, 1.0, 51000.0)
        assert pos.unrealized_pnl == -1000.0

    def test_close_position_long(self, monitor):
        monitor.update_position("BTC/USDT", "buy", 50000.0, 1.0, 51000.0)
        realized = monitor.close_position("BTC/USDT", exit_price=52000.0)
        assert realized == 2000.0
        assert len(monitor.get_all_positions()) == 0

    def test_close_position_short(self, monitor):
        monitor.update_position("BTC/USDT", "sell", 50000.0, 1.0, 49000.0)
        realized = monitor.close_position("BTC/USDT", exit_price=48000.0)
        assert realized == 2000.0

    def test_close_nonexistent_position(self, monitor):
        realized = monitor.close_position("ETH/USDT", exit_price=3000.0)
        assert realized == 0

    def test_max_favorable_and_adverse(self, monitor):
        monitor.update_position("BTC/USDT", "buy", 50000.0, 1.0, 52000.0)
        pos = monitor.update_position("BTC/USDT", "buy", 50000.0, 1.0, 48000.0)
        assert pos.max_favorable == 2000.0
        assert pos.max_adverse == -2000.0

    def test_get_all_positions(self, monitor):
        monitor.update_position("BTC/USDT", "buy", 50000.0, 1.0, 51000.0)
        monitor.update_position("ETH/USDT", "buy", 3000.0, 10.0, 3100.0)
        positions = monitor.get_all_positions()
        assert len(positions) == 2

    def test_get_total_unrealized(self, monitor):
        monitor.update_position("BTC/USDT", "buy", 50000.0, 1.0, 51000.0)
        monitor.update_position("ETH/USDT", "buy", 3000.0, 10.0, 3100.0)
        total = monitor.get_total_unrealized()
        assert total == 1000.0 + 1000.0

    def test_get_total_exposure(self, monitor):
        monitor.update_position("BTC/USDT", "buy", 50000.0, 1.0, 51000.0)
        exposure = monitor.get_total_exposure()
        assert exposure == 51000.0 * 1.0

    def test_get_leverage(self, monitor):
        monitor.update_position("BTC/USDT", "buy", 50000.0, 2.0, 50000.0)
        leverage = monitor.get_leverage()
        assert leverage == pytest.approx(1.0)

    def test_get_summary(self, monitor):
        monitor.update_position("BTC/USDT", "buy", 50000.0, 1.0, 51000.0)
        summary = monitor.get_summary()
        assert summary["n_positions"] == 1
        assert "BTC/USDT" in summary["symbols"]
        assert summary["total_unrealized"] == 1000.0
