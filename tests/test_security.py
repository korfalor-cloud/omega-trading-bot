"""Tests for security utilities — input validation, data sanitization, error handling."""
from __future__ import annotations

import math

import pytest

from tradingbot.core.enums import OrderState, OrderType, Side
from tradingbot.core.errors import (
    ConfigError,
    ExchangeError,
    InsufficientFundsError,
    OmegaError,
    OrderError,
    OrderRejectedError,
    RiskLimitExceededError,
)
from tradingbot.core.types import Order, Position, Signal
from tradingbot.risk.risk_manager import RiskManager
from tradingbot.core.events import EventBus


class TestInputValidation:
    """Test that domain objects enforce valid inputs."""

    def test_order_default_state(self):
        order = Order()
        assert order.state == OrderState.PENDING
        assert order.remaining_quantity == 0.0

    def test_order_notional_zero_price(self):
        order = Order(quantity=10.0, price=None)
        assert order.notional_value == 0.0

    def test_order_notional_with_price(self):
        order = Order(quantity=10.0, price=100.0)
        assert order.notional_value == pytest.approx(1000.0)

    def test_position_pnl_buy(self):
        pos = Position(symbol="BTC/USDT", strategy_id="s1", side=Side.BUY,
                       quantity=1.0, avg_entry_price=50000.0, current_price=55000.0)
        assert pos.pnl_pct == pytest.approx(0.1)

    def test_position_pnl_sell(self):
        pos = Position(symbol="BTC/USDT", strategy_id="s1", side=Side.SELL,
                       quantity=1.0, avg_entry_price=50000.0, current_price=45000.0)
        assert pos.pnl_pct == pytest.approx(0.1)

    def test_position_update_price_buy(self):
        pos = Position(symbol="BTC/USDT", strategy_id="s1", side=Side.BUY,
                       quantity=2.0, avg_entry_price=50000.0)
        pos.update_price(52000.0)
        assert pos.unrealized_pnl == pytest.approx(4000.0)

    def test_position_update_price_sell(self):
        pos = Position(symbol="BTC/USDT", strategy_id="s1", side=Side.SELL,
                       quantity=2.0, avg_entry_price=50000.0)
        pos.update_price(48000.0)
        assert pos.unrealized_pnl == pytest.approx(4000.0)


class TestErrorHierarchy:
    """Test that the error hierarchy is properly structured."""

    def test_omega_error_is_base(self):
        assert issubclass(ExchangeError, OmegaError)
        assert issubclass(OrderError, OmegaError)
        assert issubclass(ConfigError, OmegaError)

    def test_order_errors(self):
        assert issubclass(InsufficientFundsError, OrderError)
        assert issubclass(OrderRejectedError, OrderError)

    def test_risk_error(self):
        assert issubclass(RiskLimitExceededError, OmegaError)

    def test_catch_specific_error(self):
        with pytest.raises(ExchangeError):
            raise ExchangeError("connection failed")

    def test_catch_base_error(self):
        with pytest.raises(OmegaError):
            raise InsufficientFundsError("not enough funds")


class TestRiskManagerValidation:
    """Test RiskManager rejects invalid or risky operations."""

    @pytest.fixture
    def risk_manager(self):
        bus = EventBus()
        return RiskManager({
            "max_position_pct": 0.05,
            "max_gross_exposure": 2.0,
            "max_drawdown_pct": 0.15,
            "daily_loss_limit_pct": 0.05,
            "max_leverage": 3.0,
        }, bus)

    @pytest.mark.asyncio
    async def test_emergency_stop_blocks_all(self, risk_manager):
        from tradingbot.core.types import PortfolioState
        from datetime import datetime
        await risk_manager.emergency_stop()
        signal = Signal(strategy_id="s1", symbol="BTC/USDT", side=Side.BUY, strength=0.5, confidence=0.8)
        portfolio = PortfolioState(timestamp=datetime.utcnow(), total_equity=100000, cash=50000,
                                   positions_value=50000, unrealized_pnl=0, realized_pnl=0)
        check = await risk_manager.pre_trade_check(signal, portfolio, 50000.0)
        assert check.approved is False
        assert "Emergency" in check.reason

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks(self, risk_manager):
        from tradingbot.core.types import PortfolioState
        from datetime import datetime
        risk_manager._circuit_breaker_active = True
        signal = Signal(strategy_id="s1", symbol="BTC/USDT", side=Side.BUY, strength=0.5, confidence=0.8)
        portfolio = PortfolioState(timestamp=datetime.utcnow(), total_equity=100000, cash=50000,
                                   positions_value=50000, unrealized_pnl=0, realized_pnl=0)
        check = await risk_manager.pre_trade_check(signal, portfolio, 50000.0)
        assert check.approved is False
        assert "Circuit breaker" in check.reason

    @pytest.mark.asyncio
    async def test_drawdown_triggers_circuit_breaker(self, risk_manager):
        from tradingbot.core.types import PortfolioState
        from datetime import datetime
        signal = Signal(strategy_id="s1", symbol="BTC/USDT", side=Side.BUY, strength=0.5, confidence=0.8)
        portfolio = PortfolioState(timestamp=datetime.utcnow(), total_equity=100000, cash=50000,
                                   positions_value=50000, unrealized_pnl=0, realized_pnl=0,
                                   current_drawdown=0.20)
        check = await risk_manager.pre_trade_check(signal, portfolio, 50000.0)
        assert check.approved is False
        assert risk_manager.is_circuit_breaker_active

    @pytest.mark.asyncio
    async def test_reset_clears_state(self, risk_manager):
        risk_manager._circuit_breaker_active = True
        risk_manager._daily_pnl = -500.0
        await risk_manager.reset()
        assert risk_manager.is_circuit_breaker_active is False
        assert risk_manager._daily_pnl == 0.0

    @pytest.mark.asyncio
    async def test_valid_trade_approved(self, risk_manager):
        from tradingbot.core.types import PortfolioState
        from datetime import datetime
        signal = Signal(strategy_id="s1", symbol="BTC/USDT", side=Side.BUY, strength=0.5, confidence=0.8)
        portfolio = PortfolioState(timestamp=datetime.utcnow(), total_equity=100000, cash=100000,
                                   positions_value=0, unrealized_pnl=0, realized_pnl=0,
                                   gross_exposure=0)
        check = await risk_manager.pre_trade_check(signal, portfolio, 50000.0)
        assert check.approved is True
        assert check.max_allowed_quantity > 0


class TestSanitization:
    """Test that NaN/Inf values are handled gracefully."""

    def test_nan_in_order_price(self):
        order = Order(quantity=1.0, price=float("nan"))
        assert math.isnan(order.price)

    def test_position_zero_entry_pnl(self):
        pos = Position(symbol="X", strategy_id="s", side=Side.BUY, quantity=1.0,
                       avg_entry_price=0.0, current_price=100.0)
        assert pos.pnl_pct == 0.0

    def test_order_remaining_quantity(self):
        order = Order(quantity=10.0, filled_quantity=3.0)
        assert order.remaining_quantity == pytest.approx(7.0)

    def test_order_is_active_states(self):
        for state in (OrderState.PENDING, OrderState.SUBMITTED, OrderState.PARTIAL):
            order = Order(state=state)
            assert order.is_active is True

    def test_order_inactive_states(self):
        for state in (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED):
            order = Order(state=state)
            assert order.is_active is False
