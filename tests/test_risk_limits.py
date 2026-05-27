"""Tests for risk limits engine."""
from __future__ import annotations

import pytest

from tradingbot.risk.limits import (
    LimitBreach,
    LimitCheck,
    LimitType,
    RiskLimit,
    RiskLimitsEngine,
)


class TestRiskLimitsEngine:
    @pytest.fixture
    def engine(self):
        return RiskLimitsEngine(config={
            "max_position_pct": 0.10,
            "max_daily_loss_pct": 0.03,
            "max_drawdown_pct": 0.15,
            "max_leverage": 3.0,
            "max_concentration_pct": 0.40,
        })

    def test_position_size_within_limit(self, engine):
        check = engine.check_position_size(5000, 100000)
        assert check.breach_level is None
        assert check.utilization == pytest.approx(0.5, abs=0.01)

    def test_position_size_warning(self, engine):
        check = engine.check_position_size(9000, 100000)
        assert check.breach_level == LimitBreach.WARNING

    def test_position_size_breach(self, engine):
        check = engine.check_position_size(12000, 100000)
        assert check.breach_level == LimitBreach.HARD_BREACH

    def test_daily_loss_within(self, engine):
        check = engine.check_daily_loss(-1000, 100000)
        assert check.breach_level is None

    def test_daily_loss_breach(self, engine):
        check = engine.check_daily_loss(-5000, 100000)
        assert check.breach_level == LimitBreach.HARD_BREACH

    def test_daily_gain_no_breach(self, engine):
        check = engine.check_daily_loss(5000, 100000)
        assert check.breach_level is None

    def test_drawdown_within(self, engine):
        check = engine.check_drawdown(95000, 100000)
        assert check.breach_level is None

    def test_drawdown_warning(self, engine):
        check = engine.check_drawdown(87000, 100000)
        assert check.breach_level == LimitBreach.WARNING

    def test_drawdown_breach(self, engine):
        check = engine.check_drawdown(83000, 100000)
        assert check.breach_level == LimitBreach.HARD_BREACH

    def test_leverage_within(self, engine):
        check = engine.check_leverage(200000, 100000)
        assert check.breach_level is None
        assert check.current_value == 2.0

    def test_leverage_breach(self, engine):
        check = engine.check_leverage(400000, 100000)
        assert check.breach_level == LimitBreach.HARD_BREACH

    def test_concentration_within(self, engine):
        checks = engine.check_concentration({"BTC": 30000}, 100000)
        assert checks[0].breach_level is None

    def test_concentration_breach(self, engine):
        checks = engine.check_concentration({"BTC": 50000}, 100000)
        assert checks[0].breach_level == LimitBreach.HARD_BREACH

    def test_check_all(self, engine):
        breaches = engine.check_all(
            portfolio_value=83000,
            peak_equity=100000,
            daily_pnl=-5000,
            total_exposure=300000,
            position_values={"BTC": 50000},
        )
        assert len(breaches) > 0

    def test_set_limit(self, engine):
        engine.set_limit(LimitType.DAILY_LOSS, 0.05)
        check = engine.check_daily_loss(-4000, 100000)
        assert check.breach_level is None  # 4% < 5%

    def test_circuit_breaker(self, engine):
        engine.check_drawdown(80000, 100000)  # Hard breach
        assert engine.is_circuit_breaker_active()

    def test_no_circuit_breaker(self, engine):
        engine.check_drawdown(95000, 100000)  # No breach
        assert not engine.is_circuit_breaker_active()

    def test_breach_history(self, engine):
        engine.check_drawdown(80000, 100000)
        engine.check_daily_loss(-5000, 100000)
        history = engine.get_breach_history()
        assert len(history) >= 2

    def test_disabled_limit(self, engine):
        engine.set_limit(LimitType.DRAWDOWN, 0.10)
        # Disable by setting threshold to 0
        engine._limits[LimitType.DRAWDOWN].enabled = False
        check = engine.check_drawdown(50000, 100000)
        assert check.breach_level is None
