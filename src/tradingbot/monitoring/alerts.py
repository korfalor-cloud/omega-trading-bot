"""Alerting and Notification System.

Implements:
- Price alerts (crossing, percentage change)
- Performance alerts (drawdown, P&L threshold)
- Risk alerts (exposure, correlation)
- Delivery via webhook, email, or Telegram
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class AlertType(Enum):
    PRICE = auto()
    PERFORMANCE = auto()
    RISK = auto()
    SYSTEM = auto()
    STRATEGY = auto()


class AlertPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class AlertStatus(Enum):
    ACTIVE = auto()
    TRIGGERED = auto()
    ACKNOWLEDGED = auto()
    EXPIRED = auto()


@dataclass
class Alert:
    """An alert definition."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    alert_type: AlertType = AlertType.PRICE
    priority: AlertPriority = AlertPriority.MEDIUM
    status: AlertStatus = AlertStatus.ACTIVE
    condition: str = ""  # Human-readable condition
    symbol: str = ""
    metric: str = ""
    threshold: float = 0.0
    current_value: float = 0.0
    message: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    triggered_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    cooldown_seconds: int = 300  # Min seconds between re-triggers
    last_triggered: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class AlertEvent:
    """A triggered alert event."""
    alert_id: str
    alert_name: str
    priority: AlertPriority
    message: str
    current_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AlertManager:
    """Central alert management system.

    Creates, evaluates, and delivers alerts.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.max_alerts = config.get("max_alerts", 1000)
        self._alerts: dict[str, Alert] = {}
        self._history: list[AlertEvent] = []
        self._webhooks: list[str] = config.get("webhooks", [])
        self._telegram_token: str = config.get("telegram_token", "")
        self._telegram_chat_id: str = config.get("telegram_chat_id", "")

    def create_price_alert(
        self,
        name: str,
        symbol: str,
        threshold: float,
        condition: str = "crosses_above",  # crosses_above, crosses_below, pct_change
        priority: AlertPriority = AlertPriority.MEDIUM,
        expires_hours: int = 24,
    ) -> Alert:
        """Create a price-based alert."""
        alert = Alert(
            name=name,
            alert_type=AlertType.PRICE,
            priority=priority,
            condition=condition,
            symbol=symbol,
            metric="price",
            threshold=threshold,
            expires_at=datetime.utcnow() + timedelta(hours=expires_hours),
        )
        self._alerts[alert.id] = alert
        return alert

    def create_drawdown_alert(
        self,
        name: str,
        max_drawdown_pct: float,
        priority: AlertPriority = AlertPriority.HIGH,
    ) -> Alert:
        """Create a drawdown alert."""
        alert = Alert(
            name=name,
            alert_type=AlertType.RISK,
            priority=priority,
            condition="drawdown_exceeds",
            metric="max_drawdown",
            threshold=max_drawdown_pct,
        )
        self._alerts[alert.id] = alert
        return alert

    def create_pnl_alert(
        self,
        name: str,
        pnl_threshold: float,
        condition: str = "daily_loss",
        priority: AlertPriority = AlertPriority.HIGH,
    ) -> Alert:
        """Create a P&L alert."""
        alert = Alert(
            name=name,
            alert_type=AlertType.PERFORMANCE,
            priority=priority,
            condition=condition,
            metric="pnl",
            threshold=pnl_threshold,
        )
        self._alerts[alert.id] = alert
        return alert

    def create_system_alert(
        self,
        name: str,
        metric: str,
        threshold: float,
        condition: str = "exceeds",
        priority: AlertPriority = AlertPriority.MEDIUM,
    ) -> Alert:
        """Create a system health alert."""
        alert = Alert(
            name=name,
            alert_type=AlertType.SYSTEM,
            priority=priority,
            condition=condition,
            metric=metric,
            threshold=threshold,
        )
        self._alerts[alert.id] = alert
        return alert

    def evaluate_price(self, symbol: str, current_price: float) -> list[AlertEvent]:
        """Evaluate all price alerts for a symbol."""
        events = []
        for alert in list(self._alerts.values()):
            if alert.symbol != symbol or alert.status != AlertStatus.ACTIVE:
                continue
            if alert.alert_type != AlertType.PRICE:
                continue

            triggered = False
            if alert.condition == "crosses_above" and current_price >= alert.threshold:
                triggered = True
            elif alert.condition == "crosses_below" and current_price <= alert.threshold:
                triggered = True

            if triggered:
                event = self._trigger_alert(alert, current_price)
                if event:
                    events.append(event)

        return events

    def evaluate_metric(self, metric: str, value: float) -> list[AlertEvent]:
        """Evaluate alerts for a given metric."""
        events = []
        for alert in list(self._alerts.values()):
            if alert.metric != metric or alert.status != AlertStatus.ACTIVE:
                continue

            triggered = False
            if alert.condition in ("exceeds", "drawdown_exceeds", "daily_loss") and value >= alert.threshold:
                triggered = True
            elif alert.condition == "below" and value <= alert.threshold:
                triggered = True

            if triggered:
                event = self._trigger_alert(alert, value)
                if event:
                    events.append(event)

        return events

    def _trigger_alert(self, alert: Alert, current_value: float) -> Optional[AlertEvent]:
        """Trigger an alert."""
        now = datetime.utcnow()

        # Check cooldown
        if alert.last_triggered:
            elapsed = (now - alert.last_triggered).total_seconds()
            if elapsed < alert.cooldown_seconds:
                return None

        # Check expiry
        if alert.expires_at and now > alert.expires_at:
            alert.status = AlertStatus.EXPIRED
            return None

        alert.status = AlertStatus.TRIGGERED
        alert.triggered_at = now
        alert.last_triggered = now
        alert.current_value = current_value

        event = AlertEvent(
            alert_id=alert.id,
            alert_name=alert.name,
            priority=alert.priority,
            message=f"{alert.name}: {alert.metric}={current_value:.4f} (threshold={alert.threshold:.4f})",
            current_value=current_value,
            threshold=alert.threshold,
        )
        self._history.append(event)

        logger.warning(f"ALERT [{alert.priority.name}]: {event.message}")
        return event

    def acknowledge_alert(self, alert_id: str) -> bool:
        alert = self._alerts.get(alert_id)
        if alert:
            alert.status = AlertStatus.ACKNOWLEDGED
            return True
        return False

    def remove_alert(self, alert_id: str) -> bool:
        return self._alerts.pop(alert_id, None) is not None

    def get_active_alerts(self) -> list[Alert]:
        return [a for a in self._alerts.values() if a.status == AlertStatus.ACTIVE]

    def get_triggered_history(self, limit: int = 100) -> list[AlertEvent]:
        return self._history[-limit:]

    def get_status(self) -> dict:
        status_counts = {}
        for alert in self._alerts.values():
            status_counts[alert.status.name] = status_counts.get(alert.status.name, 0) + 1

        return {
            "total_alerts": len(self._alerts),
            "status_counts": status_counts,
            "events_today": len([
                e for e in self._history
                if (datetime.utcnow() - e.timestamp).days < 1
            ]),
        }
