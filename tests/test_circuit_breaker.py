"""Tests for circuit breaker — risk limits, state transitions, recovery."""
from __future__ import annotations

import pytest

from tradingbot.core.events import Event, EventBus
from tradingbot.core.types import PortfolioState, RiskAlert, Signal, Side
from tradingbot.risk.risk_manager import RiskManager
from tradingbot.risk.limits import (
    LimitBreach,
    LimitCheck,
    LimitType,
    RiskLimit,
    RiskLimitsEngine,
)
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# RiskManager circuit breaker
# ---------------------------------------------------------------------------

class TestRiskManagerCircuitBreaker:
    @pytest.fixture
    def risk_manager(self):
        bus = EventBus()
        return RiskManager({
            "max_position_pct": 0.05,
            "max_gross_exposure": 2.0,
            "max_drawdown_pct": 0.15,
            "daily_loss_limit_pct": 0.05,
            "max_leverage": 3.0,
            "circuit_breaker_dd_pct": 0.10,
        }, bus)

    def test_initial_state(self, risk_manager):
        assert risk_manager.is_circuit_breaker_active is False
        assert risk_manager.is_emergency_stop is False

    @pytest.mark.asyncio
    async def test_drawdown_triggers_circuit_breaker(self, risk_manager):
        portfolio = PortfolioState(
            timestamp=datetime.now(timezone.utc), total_equity=100000,
            cash=50000, positions_value=50000, unrealized_pnl=0, realized_pnl=0,
            current_drawdown=0.20,  # exceeds 15% limit
        )
        signal = Signal(strategy_id="s1", symbol="BTC/USDT", side=Side.BUY, strength=0.5, confidence=0.8)
        check = await risk_manager.pre_trade_check(signal, portfolio, 50000.0)
        assert check.approved is False
        assert risk_manager.is_circuit_breaker_active is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_subsequent_trades(self, risk_manager):
        risk_manager._circuit_breaker_active = True
        portfolio = PortfolioState(
            timestamp=datetime.now(timezone.utc), total_equity=100000,
            cash=100000, positions_value=0, unrealized_pnl=0, realized_pnl=0,
        )
        signal = Signal(strategy_id="s1", symbol="BTC/USDT", side=Side.BUY, strength=0.5, confidence=0.8)
        check = await risk_manager.pre_trade_check(signal, portfolio, 50000.0)
        assert check.approved is False
        assert "Circuit breaker" in check.reason

    @pytest.mark.asyncio
    async def test_reset_clears_circuit_breaker(self, risk_manager):
        risk_manager._circuit_breaker_active = True
        await risk_manager.reset()
        assert risk_manager.is_circuit_breaker_active is False

    @pytest.mark.asyncio
    async def test_emergency_stop_blocks_all(self, risk_manager):
        await risk_manager.emergency_stop()
        assert risk_manager.is_emergency_stop is True
        portfolio = PortfolioState(
            timestamp=datetime.now(timezone.utc), total_equity=100000,
            cash=100000, positions_value=0, unrealized_pnl=0, realized_pnl=0,
        )
        signal = Signal(strategy_id="s1", symbol="BTC/USDT", side=Side.BUY, strength=0.5, confidence=0.8)
        check = await risk_manager.pre_trade_check(signal, portfolio, 50000.0)
        assert check.approved is False
        assert "Emergency" in check.reason


# ---------------------------------------------------------------------------
# RiskLimitsEngine
# ---------------------------------------------------------------------------

class TestRiskLimitsEngine:
    @pytest.fixture
    def limits_engine(self):
        return RiskLimitsEngine({
            "max_position_pct": 0.10,
            "max_daily_loss_pct": 0.03,
            "max_drawdown_pct": 0.15,
            "max_leverage": 3.0,
            "max_concentration_pct": 0.40,
        })

    def test_initial_limits_set(self, limits_engine):
        assert LimitType.POSITION_SIZE in limits_engine._limits
        assert LimitType.DAILY_LOSS in limits_engine._limits
        assert LimitType.DRAWDOWN in limits_engine._limits
        assert LimitType.LEVERAGE in limits_engine._limits

    def test_position_size_within_limit(self, limits_engine):
        check = limits_engine.check_position_size(5000, 100000)
        assert check.breach_level is None
        assert check.utilization == pytest.approx(0.5)

    def test_position_size_hard_breach(self, limits_engine):
        check = limits_engine.check_position_size(15000, 100000)
        assert check.breach_level == LimitBreach.HARD_BREACH

    def test_position_size_warning(self, limits_engine):
        # 9% of 100k = 0.09, threshold is 0.10, utilization = 0.9 > 0.8 warning_pct
        check = limits_engine.check_position_size(9000, 100000)
        assert check.breach_level == LimitBreach.WARNING

    def test_daily_loss_within_limit(self, limits_engine):
        check = limits_engine.check_daily_loss(-1000, 100000)
        assert check.breach_level is None

    def test_daily_loss_hard_breach(self, limits_engine):
        check = limits_engine.check_daily_loss(-5000, 100000)
        assert check.breach_level == LimitBreach.HARD_BREACH

    def test_drawdown_within_limit(self, limits_engine):
        check = limits_engine.check_drawdown(90000, 100000)
        assert check.breach_level is None

    def test_drawdown_hard_breach(self, limits_engine):
        check = limits_engine.check_drawdown(80000, 100000)
        assert check.breach_level == LimitBreach.HARD_BREACH

    def test_leverage_within_limit(self, limits_engine):
        check = limits_engine.check_leverage(200000, 100000)
        assert check.breach_level is None

    def test_leverage_hard_breach(self, limits_engine):
        check = limits_engine.check_leverage(400000, 100000)
        assert check.breach_level == LimitBreach.HARD_BREACH

    def test_concentration_check(self, limits_engine):
        positions = {"BTC/USDT": 50000, "ETH/USDT": 10000}
        checks = limits_engine.check_concentration(positions, 100000)
        assert len(checks) == 2
        # BTC is 50% > 40% threshold -> hard breach
        btc_check = next(c for c in checks if "BTC" in c.message)
        assert btc_check.breach_level == LimitBreach.HARD_BREACH

    def test_check_all_returns_breaches_only(self, limits_engine):
        checks = limits_engine.check_all(
            portfolio_value=100000,
            peak_equity=100000,
            daily_pnl=-1000,
            total_exposure=150000,
            position_values={"BTC/USDT": 5000},
        )
        # All within limits, so no breaches
        assert len(checks) == 0

    def test_breach_history_recorded(self, limits_engine):
        limits_engine.check_position_size(15000, 100000)
        history = limits_engine.get_breach_history()
        assert len(history) == 1
        assert history[0].breach_level == LimitBreach.HARD_BREACH

    def test_circuit_breaker_from_breach_history(self, limits_engine):
        limits_engine.check_position_size(15000, 100000)
        assert limits_engine.is_circuit_breaker_active() is True

    def test_custom_limit_set(self, limits_engine):
        limits_engine.set_limit(LimitType.POSITION_SIZE, 0.20)
        check = limits_engine.check_position_size(15000, 100000)
        assert check.breach_level is None  # 15% < 20%
