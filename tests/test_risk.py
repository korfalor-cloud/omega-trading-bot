"""Tests for risk management."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from tradingbot.core.enums import Side
from tradingbot.core.events import Event, EventBus
from tradingbot.core.types import Fill, PortfolioState, Position, Signal
from tradingbot.risk.risk_manager import RiskManager


class TestRiskManager:
    @pytest.fixture
    def event_bus(self):
        return EventBus()

    @pytest.fixture
    def risk_manager(self, event_bus):
        return RiskManager({
            "max_position_pct": 0.05,
            "max_gross_exposure": 2.0,
            "max_drawdown_pct": 0.15,
            "daily_loss_limit_pct": 0.05,
            "max_leverage": 3.0,
            "circuit_breaker_dd_pct": 0.10,
        }, event_bus)

    @pytest.fixture
    def healthy_portfolio(self):
        return PortfolioState(
            timestamp=datetime.now(timezone.utc),
            total_equity=100000.0,
            cash=50000.0,
            positions_value=50000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            positions=[],
            current_drawdown=0.0,
            gross_exposure=50000.0,
            leverage=0.5,
        )

    @pytest.fixture
    def buy_signal(self):
        return Signal(
            strategy_id="strat-1",
            symbol="BTC/USDT",
            side=Side.BUY,
            strength=0.8,
            confidence=0.7,
        )

    @pytest.mark.asyncio
    async def test_healthy_portfolio_approved(self, risk_manager, healthy_portfolio, buy_signal):
        check = await risk_manager.pre_trade_check(buy_signal, healthy_portfolio, 50000.0)
        assert check.approved
        assert check.max_allowed_quantity > 0

    @pytest.mark.asyncio
    async def test_emergency_stop_blocks(self, risk_manager, healthy_portfolio, buy_signal):
        await risk_manager.emergency_stop()
        check = await risk_manager.pre_trade_check(buy_signal, healthy_portfolio, 50000.0)
        assert not check.approved
        assert "Emergency stop" in check.reason

    @pytest.mark.asyncio
    async def test_circuit_breaker_on_drawdown(self, risk_manager, buy_signal):
        # Portfolio with drawdown exceeding the 15% limit
        distressed = PortfolioState(
            timestamp=datetime.now(timezone.utc),
            total_equity=84000.0,
            cash=84000.0,
            positions_value=0.0,
            unrealized_pnl=-16000.0,
            realized_pnl=-16000.0,
            positions=[],
            current_drawdown=0.16,
            gross_exposure=0.0,
            leverage=0.0,
        )
        check = await risk_manager.pre_trade_check(buy_signal, distressed, 50000.0)
        assert not check.approved

    @pytest.mark.asyncio
    async def test_post_trade_update(self, risk_manager):
        fill = Fill(
            order_id="order-1",
            symbol="BTC/USDT",
            side=Side.BUY,
            price=50000.0,
            quantity=0.1,
            commission=5.0,
            exchange="binance",
            timestamp=datetime.now(timezone.utc),
        )
        await risk_manager.post_trade_update(fill)
        # Should not raise

    @pytest.mark.asyncio
    async def test_reset_clears_state(self, risk_manager):
        await risk_manager.emergency_stop()
        assert risk_manager.is_emergency_stop
        await risk_manager.reset()
        assert not risk_manager.is_circuit_breaker_active

    @pytest.mark.asyncio
    async def test_position_size_limit(self, risk_manager, buy_signal):
        small_portfolio = PortfolioState(
            timestamp=datetime.now(timezone.utc),
            total_equity=10000.0,
            cash=10000.0,
            positions_value=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            positions=[],
            current_drawdown=0.0,
            gross_exposure=0.0,
            leverage=0.0,
        )
        check = await risk_manager.pre_trade_check(buy_signal, small_portfolio, 50000.0)
        assert check.approved
        # Max position should be limited
        assert check.max_allowed_quantity <= 10000 * 0.05 * 0.7 / 50000  # confidence-adjusted
