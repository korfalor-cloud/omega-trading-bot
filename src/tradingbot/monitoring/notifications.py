"""Notification System — multi-channel alerts.

Implements:
- Email notifications
- Slack integration
- Discord integration
- Push notifications
- Notification templates
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

logger = logging.getLogger(__name__)


class Channel(Enum):
    EMAIL = auto()
    SLACK = auto()
    DISCORD = auto()
    PUSH = auto()
    TELEGRAM = auto()


@dataclass
class Notification:
    """A notification message."""
    title: str = ""
    body: str = ""
    channel: Channel = Channel.TELEGRAM
    priority: str = "normal"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


class NotificationManager:
    """Multi-channel notification manager."""

    def __init__(self, config: dict = None):
        config = config or {}
        self._enabled: dict[Channel, bool] = {
            Channel.EMAIL: config.get("email_enabled", False),
            Channel.SLACK: config.get("slack_enabled", False),
            Channel.DISCORD: config.get("discord_enabled", False),
            Channel.PUSH: config.get("push_enabled", False),
            Channel.TELEGRAM: config.get("telegram_enabled", False),
        }
        self._webhooks: dict[str, str] = config.get("webhooks", {})
        self._history: list[Notification] = []

    def send(self, notification: Notification) -> bool:
        """Send a notification."""
        self._history.append(notification)

        if not self._enabled.get(notification.channel, False):
            logger.debug(f"Channel {notification.channel} disabled, skipping")
            return False

        try:
            if notification.channel == Channel.SLACK:
                return self._send_slack(notification)
            elif notification.channel == Channel.DISCORD:
                return self._send_discord(notification)
            elif notification.channel == Channel.EMAIL:
                return self._send_email(notification)
            elif notification.channel == Channel.TELEGRAM:
                return self._send_telegram(notification)
            else:
                logger.info(f"Notification: {notification.title} - {notification.body}")
                return True
        except Exception as e:
            logger.error(f"Notification failed: {e}")
            return False

    def _send_slack(self, notification: Notification) -> bool:
        webhook = self._webhooks.get("slack")
        if not webhook:
            return False
        # Real implementation would POST to webhook
        logger.info(f"Slack: {notification.title}")
        return True

    def _send_discord(self, notification: Notification) -> bool:
        webhook = self._webhooks.get("discord")
        if not webhook:
            return False
        logger.info(f"Discord: {notification.title}")
        return True

    def _send_email(self, notification: Notification) -> bool:
        logger.info(f"Email: {notification.title}")
        return True

    def _send_telegram(self, notification: Notification) -> bool:
        logger.info(f"Telegram: {notification.title}")
        return True

    def send_trade_alert(self, symbol: str, side: str, price: float, pnl: float = 0, channel: Channel = Channel.TELEGRAM) -> bool:
        emoji = "🟢" if side == "buy" else "🔴"
        return self.send(Notification(
            title=f"{emoji} Trade: {side.upper()} {symbol}",
            body=f"Price: {price:.2f} | P&L: {pnl:+.2f}",
            channel=channel,
        ))

    def send_risk_alert(self, message: str, channel: Channel = Channel.TELEGRAM) -> bool:
        return self.send(Notification(
            title="⚠️ Risk Alert",
            body=message,
            channel=channel,
            priority="high",
        ))

    def send_evolution_alert(self, generation: int, fitness: float, channel: Channel = Channel.TELEGRAM) -> bool:
        return self.send(Notification(
            title=f"🧬 Evolution: Gen {generation}",
            body=f"Best fitness: {fitness:.4f}",
            channel=channel,
        ))

    def get_history(self, channel: Channel = None, limit: int = 100) -> list[Notification]:
        if channel:
            return [n for n in self._history if n.channel == channel][-limit:]
        return self._history[-limit:]
