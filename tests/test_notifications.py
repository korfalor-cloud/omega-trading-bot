"""Tests for NotificationManager — multi-channel alerts."""
from __future__ import annotations

import pytest

from tradingbot.monitoring.notifications import (
    Channel,
    Notification,
    NotificationManager,
)


class TestNotificationManager:
    @pytest.fixture
    def manager(self):
        return NotificationManager()

    @pytest.fixture
    def enabled_manager(self):
        return NotificationManager({
            "email_enabled": True,
            "slack_enabled": True,
            "discord_enabled": True,
            "telegram_enabled": True,
            "webhooks": {"slack": "https://hooks.slack.com/test", "discord": "https://discord.com/webhooks/test"},
        })

    def test_send_disabled_channel(self, manager):
        notif = Notification(title="Test", body="Body", channel=Channel.SLACK)
        result = manager.send(notif)
        assert result is False

    def test_send_enabled_telegram(self, enabled_manager):
        notif = Notification(title="Test", body="Body", channel=Channel.TELEGRAM)
        result = enabled_manager.send(notif)
        assert result is True

    def test_send_enabled_email(self, enabled_manager):
        notif = Notification(title="Test", body="Body", channel=Channel.EMAIL)
        result = enabled_manager.send(notif)
        assert result is True

    def test_send_slack_with_webhook(self, enabled_manager):
        notif = Notification(title="Test", body="Body", channel=Channel.SLACK)
        result = enabled_manager.send(notif)
        assert result is True

    def test_send_slack_without_webhook(self):
        mgr = NotificationManager({"slack_enabled": True})
        notif = Notification(title="Test", body="Body", channel=Channel.SLACK)
        result = mgr.send(notif)
        assert result is False

    def test_send_discord_with_webhook(self, enabled_manager):
        notif = Notification(title="Test", body="Body", channel=Channel.DISCORD)
        result = enabled_manager.send(notif)
        assert result is True

    def test_history_recorded(self, manager):
        notif = Notification(title="Test", body="Body", channel=Channel.TELEGRAM)
        manager.send(notif)
        history = manager.get_history()
        assert len(history) == 1
        assert history[0].title == "Test"

    def test_history_filter_by_channel(self, enabled_manager):
        enabled_manager.send(Notification(title="T1", body="B1", channel=Channel.TELEGRAM))
        enabled_manager.send(Notification(title="T2", body="B2", channel=Channel.EMAIL))
        telegram_only = enabled_manager.get_history(channel=Channel.TELEGRAM)
        assert len(telegram_only) == 1
        assert telegram_only[0].title == "T1"

    def test_history_limit(self, manager):
        for i in range(10):
            manager.send(Notification(title=f"T{i}", body="B", channel=Channel.TELEGRAM))
        limited = manager.get_history(limit=3)
        assert len(limited) == 3

    def test_send_trade_alert(self, enabled_manager):
        result = enabled_manager.send_trade_alert("BTC/USDT", "buy", 50000.0, pnl=100.0)
        assert result is True
        history = enabled_manager.get_history()
        assert "Trade" in history[-1].title

    def test_send_risk_alert(self, enabled_manager):
        result = enabled_manager.send_risk_alert("Drawdown exceeded 5%")
        assert result is True
        history = enabled_manager.get_history()
        assert history[-1].priority == "high"

    def test_send_evolution_alert(self, enabled_manager):
        result = enabled_manager.send_evolution_alert(generation=10, fitness=0.85)
        assert result is True
        history = enabled_manager.get_history()
        assert "Evolution" in history[-1].title

    def test_notification_default_values(self):
        notif = Notification()
        assert notif.title == ""
        assert notif.channel == Channel.TELEGRAM
        assert notif.priority == "normal"
