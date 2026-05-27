"""Tests for portfolio rebalancer."""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from tradingbot.portfolio.rebalancer import (
    PortfolioRebalancer,
    RebalanceResult,
    RebalanceTrade,
)


class TestPortfolioRebalancer:
    @pytest.fixture
    def rebalancer(self):
        return PortfolioRebalancer(config={
            "drift_threshold": 0.05,
            "cost_per_trade": 0.001,
            "min_trade_value": 10.0,
        })

    def test_check_drift_no_rebalance_needed(self, rebalancer):
        current = {"BTC": 0.40, "ETH": 0.35, "SOL": 0.25}
        target = {"BTC": 0.42, "ETH": 0.34, "SOL": 0.24}
        result = rebalancer.check_drift(current, target)
        assert result.should_rebalance is False
        assert result.max_drift < 0.05

    def test_check_drift_rebalance_needed(self, rebalancer):
        current = {"BTC": 0.60, "ETH": 0.20, "SOL": 0.20}
        target = {"BTC": 0.40, "ETH": 0.35, "SOL": 0.25}
        result = rebalancer.check_drift(current, target)
        assert result.should_rebalance is True
        assert result.max_drift > 0.05
        assert len(result.trades) > 0

    def test_check_calendar_daily(self, rebalancer):
        rebalancer.rebalance_frequency = "daily"
        now = datetime(2024, 6, 1, tzinfo=timezone.utc)
        assert rebalancer.check_calendar(now) is True

        rebalancer.last_rebalance = now
        assert rebalancer.check_calendar(now + timedelta(hours=12)) is False
        assert rebalancer.check_calendar(now + timedelta(days=1, seconds=1)) is True

    def test_check_calendar_weekly(self, rebalancer):
        rebalancer.rebalance_frequency = "weekly"
        now = datetime(2024, 6, 1, tzinfo=timezone.utc)
        rebalancer.last_rebalance = now
        assert rebalancer.check_calendar(now + timedelta(days=6)) is False
        assert rebalancer.check_calendar(now + timedelta(days=7, seconds=1)) is True

    def test_check_calendar_monthly(self, rebalancer):
        rebalancer.rebalance_frequency = "monthly"
        now = datetime(2024, 6, 1, tzinfo=timezone.utc)
        rebalancer.last_rebalance = now
        assert rebalancer.check_calendar(now + timedelta(days=29)) is False
        assert rebalancer.check_calendar(now + timedelta(days=31)) is True

    def test_compute_trades(self, rebalancer):
        current = {"BTC": 0.60, "ETH": 0.40}
        target = {"BTC": 0.50, "ETH": 0.50}
        prices = {"BTC": 60000, "ETH": 3000}
        portfolio_value = 100000

        result = rebalancer.compute_trades(current, target, portfolio_value, prices)
        assert result.should_rebalance is True
        assert len(result.trades) == 2
        assert result.estimated_cost > 0

    def test_compute_trades_small_drift(self, rebalancer):
        # Very small portfolio so trade value < min_trade_value
        current = {"BTC": 0.501, "ETH": 0.499}
        target = {"BTC": 0.50, "ETH": 0.50}
        prices = {"BTC": 60000, "ETH": 3000}

        result = rebalancer.compute_trades(current, target, 100, prices)
        assert result.should_rebalance is False  # Trade value < min_trade_value

    def test_minimize_trades(self, rebalancer):
        trades = [
            RebalanceTrade(symbol="BTC", drift=0.01),
            RebalanceTrade(symbol="ETH", drift=0.10),
            RebalanceTrade(symbol="SOL", drift=0.05),
        ]
        limited = rebalancer.minimize_trades(trades, max_trades=2)
        assert len(limited) == 2
        # Should keep the largest drifts
        assert limited[0].symbol == "ETH"

    def test_tax_lot_optimize(self, rebalancer):
        lots = [
            {"lot_id": "1", "quantity": 0.5, "cost_basis": 50000},
            {"lot_id": "2", "quantity": 0.3, "cost_basis": 60000},
            {"lot_id": "3", "quantity": 0.2, "cost_basis": 40000},
        ]
        selected = rebalancer.tax_lot_optimize("BTC", 0.6, lots)
        assert len(selected) >= 2
        # Highest cost basis first
        assert selected[0]["lot_id"] == "2"

    def test_estimate_rebalance_impact(self, rebalancer):
        trades = [
            RebalanceTrade(symbol="BTC", quantity=0.1),
        ]
        prices = {"BTC": 60000}
        adv = {"BTC": 1000}
        impacts = rebalancer.estimate_rebalance_impact(trades, prices, adv)
        assert "BTC" in impacts
        assert impacts["BTC"] > 0

    def test_check_drift_new_asset(self, rebalancer):
        current = {"BTC": 1.0}
        target = {"BTC": 0.5, "ETH": 0.5}
        result = rebalancer.check_drift(current, target)
        assert result.should_rebalance is True

    def test_empty_portfolio(self, rebalancer):
        result = rebalancer.check_drift({}, {})
        assert result.should_rebalance is False
