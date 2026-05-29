"""Tests for risk dashboard."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from tradingbot.monitoring.dashboard import RiskDashboard, DashboardState


class TestRiskDashboard:
    @pytest.fixture
    def dashboard(self):
        return RiskDashboard()

    def test_update_equity(self, dashboard):
        dashboard.update_equity(100000)
        dashboard.update_equity(101000)
        curve = dashboard.get_equity_curve()
        assert len(curve) == 2

    def test_record_pnl(self, dashboard):
        dashboard.record_pnl(100, strategy_id="trend", symbol="BTC/USDT")
        dashboard.record_pnl(-50, strategy_id="trend", symbol="ETH/USDT")
        history = dashboard.get_pnl_history()
        assert len(history) == 2

    def test_get_state(self, dashboard):
        dashboard.update_equity(100000)
        dashboard.record_pnl(100, strategy_id="trend", symbol="BTC/USDT")
        state = dashboard.get_state(equity=100000, daily_pnl=100)
        assert isinstance(state, DashboardState)
        assert state.total_equity == 100000
        assert state.daily_pnl == 100

    def test_strategy_breakdown(self, dashboard):
        dashboard.record_pnl(100, strategy_id="trend")
        dashboard.record_pnl(50, strategy_id="trend")
        dashboard.record_pnl(-30, strategy_id="mean_rev")
        state = dashboard.get_state(equity=100000)
        assert state.strategy_breakdown["trend"] == 150
        assert state.strategy_breakdown["mean_rev"] == -30

    def test_symbol_breakdown(self, dashboard):
        dashboard.record_pnl(100, symbol="BTC/USDT")
        dashboard.record_pnl(-50, symbol="ETH/USDT")
        state = dashboard.get_state(equity=100000)
        assert state.symbol_breakdown["BTC/USDT"] == 100

    def test_strategy_ranking(self, dashboard):
        dashboard.record_pnl(100, strategy_id="trend")
        dashboard.record_pnl(200, strategy_id="momentum")
        dashboard.record_pnl(-30, strategy_id="mean_rev")
        ranking = dashboard.get_strategy_ranking()
        assert ranking[0][0] == "momentum"
        assert ranking[-1][0] == "mean_rev"

    def test_symbol_ranking(self, dashboard):
        dashboard.record_pnl(100, symbol="BTC/USDT")
        dashboard.record_pnl(200, symbol="SOL/USDT")
        ranking = dashboard.get_symbol_ranking()
        assert ranking[0][0] == "SOL/USDT"

    def test_leverage(self, dashboard):
        positions = {
            "BTC": {"notional": 30000},
            "ETH": {"notional": 20000},
        }
        state = dashboard.get_state(positions=positions, equity=100000)
        assert state.leverage == pytest.approx(0.5, abs=0.01)
        assert state.n_positions == 2

    def test_max_drawdown(self, dashboard):
        for eq in [100000, 105000, 95000, 100000, 110000]:
            dashboard.update_equity(eq)
        state = dashboard.get_state(equity=110000)
        assert state.max_drawdown > 0

    def test_win_rate(self, dashboard):
        for pnl in [100, -50, 200, -30, 150]:
            dashboard.record_pnl(pnl)
        state = dashboard.get_state(equity=100000)
        assert state.win_rate == pytest.approx(0.6, abs=0.01)

    def test_empty_dashboard(self, dashboard):
        state = dashboard.get_state(equity=100000)
        assert state.total_pnl == 0
        assert state.n_positions == 0
