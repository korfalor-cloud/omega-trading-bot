"""Tests for position sizing."""
from __future__ import annotations

import pytest

from tradingbot.risk.position_sizing import PositionSizer, SizingResult


class TestPositionSizer:
    @pytest.fixture
    def sizer(self):
        return PositionSizer(config={"max_position_pct": 0.10, "risk_per_trade": 0.01, "kelly_fraction": 0.5})

    def test_kelly_size(self, sizer):
        result = sizer.kelly_size(win_rate=0.6, avg_win=200, avg_loss=100, portfolio_value=100000, price=50000)
        assert result.position_size > 0
        assert result.method == "kelly"
        assert result.details["kelly_fraction"] == 0.5

    def test_kelly_no_edge(self, sizer):
        result = sizer.kelly_size(win_rate=0.4, avg_win=100, avg_loss=100, portfolio_value=100000, price=50000)
        assert result.position_size == 0

    def test_kelly_max_position_cap(self, sizer):
        result = sizer.kelly_size(win_rate=0.9, avg_win=1000, avg_loss=10, portfolio_value=100000, price=50000)
        assert result.risk_amount <= 100000 * 0.10

    def test_volatility_target(self, sizer):
        result = sizer.volatility_target(100000, 50000, current_vol=0.30, target_vol=0.15)
        assert result.position_size > 0
        assert result.method == "vol_target"

    def test_vol_target_high_vol(self, sizer):
        low_vol = sizer.volatility_target(100000, 50000, current_vol=0.10, target_vol=0.15)
        high_vol = sizer.volatility_target(100000, 50000, current_vol=0.50, target_vol=0.15)
        assert low_vol.position_size > high_vol.position_size

    def test_risk_per_trade(self, sizer):
        result = sizer.risk_per_trade_size(100000, 50000, stop_distance=5000)
        assert result.position_size > 0
        assert result.method == "risk_per_trade"
        # risk = 100000 * 0.01 = 1000. size = 1000 / 5000 = 0.2
        assert result.position_size == pytest.approx(0.2, abs=0.01)

    def test_risk_per_trade_cap(self, sizer):
        result = sizer.risk_per_trade_size(100000, 50000, stop_distance=1)
        # Without cap: 1000 / 1 = 1000 units = 50M, way over 10% cap
        assert result.position_size <= 100000 * 0.10 / 50000

    def test_atr_size(self, sizer):
        result = sizer.atr_size(100000, 50000, atr=1000, atr_multiplier=2.0)
        assert result.position_size > 0
        assert result.method == "risk_per_trade"

    def test_fixed_fractional(self, sizer):
        result = sizer.fixed_fractional(100000, 50000, fraction=0.05)
        assert result.position_size == pytest.approx(0.1, abs=0.01)
        assert result.method == "fixed_fractional"

    def test_zero_price(self, sizer):
        result = sizer.kelly_size(0.6, 200, 100, 100000, 0)
        assert result.position_size == 0

    def test_zero_stop(self, sizer):
        result = sizer.risk_per_trade_size(100000, 50000, 0)
        assert result.position_size == 0
