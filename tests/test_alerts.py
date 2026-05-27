"""Tests for alerting system."""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from tradingbot.monitoring.alerts import (
    Alert,
    AlertEvent,
    AlertManager,
    AlertPriority,
    AlertStatus,
    AlertType,
)


class TestAlertManager:
    @pytest.fixture
    def manager(self):
        return AlertManager()

    def test_create_price_alert(self, manager):
        alert = manager.create_price_alert(
            name="BTC breakout",
            symbol="BTC/USDT",
            threshold=60000,
            condition="crosses_above",
        )
        assert alert.alert_type == AlertType.PRICE
        assert alert.symbol == "BTC/USDT"
        assert alert.threshold == 60000
        assert alert.status == AlertStatus.ACTIVE
        assert alert.id in manager._alerts

    def test_create_drawdown_alert(self, manager):
        alert = manager.create_drawdown_alert(
            name="Max DD",
            max_drawdown_pct=0.10,
        )
        assert alert.alert_type == AlertType.RISK
        assert alert.threshold == 0.10

    def test_create_pnl_alert(self, manager):
        alert = manager.create_pnl_alert(
            name="Daily loss",
            pnl_threshold=-500,
            condition="daily_loss",
        )
        assert alert.alert_type == AlertType.PERFORMANCE
        assert alert.threshold == -500

    def test_create_system_alert(self, manager):
        alert = manager.create_system_alert(
            name="High latency",
            metric="latency_ms",
            threshold=500,
        )
        assert alert.alert_type == AlertType.SYSTEM
        assert alert.metric == "latency_ms"

    def test_evaluate_price_crosses_above(self, manager):
        manager.create_price_alert(
            name="BTC up",
            symbol="BTC/USDT",
            threshold=60000,
            condition="crosses_above",
        )
        # Below threshold — no trigger
        events = manager.evaluate_price("BTC/USDT", 59000)
        assert len(events) == 0

        # Above threshold — trigger
        events = manager.evaluate_price("BTC/USDT", 61000)
        assert len(events) == 1
        assert events[0].current_value == 61000

    def test_evaluate_price_crosses_below(self, manager):
        manager.create_price_alert(
            name="BTC down",
            symbol="BTC/USDT",
            threshold=50000,
            condition="crosses_below",
        )
        events = manager.evaluate_price("BTC/USDT", 49000)
        assert len(events) == 1

    def test_cooldown_prevents_retrigger(self, manager):
        alert = manager.create_price_alert(
            name="BTC up",
            symbol="BTC/USDT",
            threshold=60000,
            condition="crosses_above",
        )
        alert.cooldown_seconds = 9999  # Long cooldown

        events = manager.evaluate_price("BTC/USDT", 61000)
        assert len(events) == 1

        # Second evaluation within cooldown — no trigger
        events = manager.evaluate_price("BTC/USDT", 62000)
        assert len(events) == 0

    def test_alert_expiry(self, manager):
        alert = manager.create_price_alert(
            name="Temp",
            symbol="BTC/USDT",
            threshold=60000,
            condition="crosses_above",
        )
        alert.expires_at = datetime.utcnow() - timedelta(hours=1)
        events = manager.evaluate_price("BTC/USDT", 61000)
        assert len(events) == 0
        assert alert.status == AlertStatus.EXPIRED

    def test_evaluate_metric(self, manager):
        manager.create_drawdown_alert(
            name="DD alert",
            max_drawdown_pct=0.05,
        )
        events = manager.evaluate_metric("max_drawdown", 0.03)
        assert len(events) == 0

        events = manager.evaluate_metric("max_drawdown", 0.08)
        assert len(events) == 1

    def test_acknowledge_alert(self, manager):
        alert = manager.create_price_alert(
            name="test",
            symbol="BTC/USDT",
            threshold=60000,
        )
        result = manager.acknowledge_alert(alert.id)
        assert result is True
        assert alert.status == AlertStatus.ACKNOWLEDGED

    def test_remove_alert(self, manager):
        alert = manager.create_price_alert(
            name="test",
            symbol="BTC/USDT",
            threshold=60000,
        )
        result = manager.remove_alert(alert.id)
        assert result is True
        assert alert.id not in manager._alerts

    def test_get_active_alerts(self, manager):
        a1 = manager.create_price_alert(name="a1", symbol="BTC/USDT", threshold=60000)
        a2 = manager.create_price_alert(name="a2", symbol="ETH/USDT", threshold=3000)
        manager.acknowledge_alert(a2.id)

        active = manager.get_active_alerts()
        assert len(active) == 1
        assert active[0].id == a1.id

    def test_get_status(self, manager):
        manager.create_price_alert(name="a1", symbol="BTC/USDT", threshold=60000)
        manager.create_price_alert(name="a2", symbol="ETH/USDT", threshold=3000)
        status = manager.get_status()
        assert status["total_alerts"] == 2
        assert status["status_counts"]["ACTIVE"] == 2

    def test_get_triggered_history(self, manager):
        manager.create_price_alert(
            name="test",
            symbol="BTC/USDT",
            threshold=60000,
            condition="crosses_above",
        )
        manager.evaluate_price("BTC/USDT", 61000)
        history = manager.get_triggered_history()
        assert len(history) == 1

    def test_wrong_symbol_no_trigger(self, manager):
        manager.create_price_alert(
            name="BTC",
            symbol="BTC/USDT",
            threshold=60000,
            condition="crosses_above",
        )
        events = manager.evaluate_price("ETH/USDT", 61000)
        assert len(events) == 0

    def test_acknowledge_nonexistent(self, manager):
        result = manager.acknowledge_alert("nonexistent")
        assert result is False
