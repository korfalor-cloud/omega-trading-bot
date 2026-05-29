"""Tests for PerformanceAttribution — P&L decomposition."""
from __future__ import annotations

from datetime import datetime

import pytest

from tradingbot.monitoring.performance_attribution import (
    PerformanceAttribution,
    AttributionResult,
)


class TestPerformanceAttribution:
    @pytest.fixture
    def attribution(self):
        a = PerformanceAttribution()
        a.add_trade({"strategy_id": "momentum", "symbol": "BTC/USDT", "pnl": 100.0, "timestamp": datetime(2024, 1, 15, 10, 30)})
        a.add_trade({"strategy_id": "momentum", "symbol": "ETH/USDT", "pnl": -20.0, "timestamp": datetime(2024, 1, 15, 14, 0)})
        a.add_trade({"strategy_id": "mean_reversion", "symbol": "BTC/USDT", "pnl": 50.0, "timestamp": datetime(2024, 1, 16, 9, 0)})
        a.add_trade({"strategy_id": "mean_reversion", "symbol": "ETH/USDT", "pnl": -10.0, "timestamp": datetime(2024, 1, 16, 11, 30)})
        return a

    def test_add_trade(self, attribution):
        assert len(attribution._trades) == 4

    def test_attribute_by_strategy(self, attribution):
        result = attribution.attribute_by_strategy()
        assert result["momentum"] == 80.0
        assert result["mean_reversion"] == 40.0

    def test_attribute_by_symbol(self, attribution):
        result = attribution.attribute_by_symbol()
        assert result["BTC/USDT"] == 150.0
        assert result["ETH/USDT"] == -30.0

    def test_attribute_by_factor(self, attribution):
        factor_loadings = {
            "momentum": {"trend": 0.7, "volatility": 0.3},
            "mean_reversion": {"trend": 0.3, "volatility": 0.7},
        }
        result = attribution.attribute_by_factor(factor_loadings)
        # momentum: 80 * 0.7 = 56 for trend, 80 * 0.3 = 24 for vol
        # mean_reversion: 40 * 0.3 = 12 for trend, 40 * 0.7 = 28 for vol
        assert result["trend"] == pytest.approx(68.0)
        assert result["volatility"] == pytest.approx(52.0)

    def test_attribute_by_time_hour(self, attribution):
        result = attribution.attribute_by_time(period="hour")
        assert "10:00" in result
        assert "14:00" in result
        assert result["10:00"] == 100.0

    def test_attribute_by_time_day(self, attribution):
        result = attribution.attribute_by_time(period="day")
        assert "Monday" in result

    def test_attribute_by_time_month(self, attribution):
        result = attribution.attribute_by_time(period="month")
        assert "2024-01" in result

    def test_get_full_attribution(self, attribution):
        result = attribution.get_full_attribution()
        assert isinstance(result, AttributionResult)
        assert result.total_pnl == 120.0
        assert "momentum" in result.strategy_attribution
        assert "BTC/USDT" in result.symbol_attribution

    def test_empty_attribution(self):
        a = PerformanceAttribution()
        assert a.attribute_by_strategy() == {}
        assert a.attribute_by_symbol() == {}
        result = a.get_full_attribution()
        assert result.total_pnl == 0

    def test_trades_without_timestamp(self):
        a = PerformanceAttribution()
        a.add_trade({"strategy_id": "s1", "symbol": "BTC/USDT", "pnl": 100})
        result = a.attribute_by_time(period="hour")
        assert result == {}
