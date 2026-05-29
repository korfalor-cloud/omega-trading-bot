"""Event Bus — decoupled pub/sub for system events.

Implements:
- Publish/subscribe pattern
- Event types: signal, trade, risk_alert, regime_change, evolution
- Async handlers
- Event history
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)


class EventType(Enum):
    SIGNAL = auto()
    TRADE = auto()
    RISK_ALERT = auto()
    REGIME_CHANGE = auto()
    EVOLUTION = auto()
    STRATEGY_STARTED = auto()
    STRATEGY_STOPPED = auto()
    CONFIG_CHANGED = auto()
    SYSTEM_ERROR = auto()


@dataclass
class Event:
    """A system event."""
    type: EventType = EventType.SIGNAL
    data: dict = field(default_factory=dict)
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


class EventBus:
    """Decoupled event pub/sub system."""

    def __init__(self):
        self._handlers: dict[EventType, list[callable]] = defaultdict(list)
        self._global_handlers: list[callable] = []
        self._history: list[Event] = []
        self._max_history = 1000

    def subscribe(self, event_type: EventType, handler: callable) -> None:
        """Subscribe to a specific event type."""
        self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: callable) -> None:
        """Subscribe to all events."""
        self._global_handlers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: callable) -> None:
        """Unsubscribe from an event type."""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Type-specific handlers
        for handler in self._handlers.get(event.type, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

        # Global handlers
        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Global event handler error: {e}")

    def emit(self, event_type: EventType, data: dict = None, source: str = "") -> None:
        """Convenience method to publish an event."""
        self.publish(Event(type=event_type, data=data or {}, source=source))

    def get_history(self, event_type: EventType = None, limit: int = 100) -> list[Event]:
        """Get event history."""
        if event_type:
            events = [e for e in self._history if e.type == event_type]
        else:
            events = list(self._history)
        return events[-limit:]

    def clear_history(self) -> None:
        self._history.clear()

    def handler_count(self, event_type: EventType = None) -> int:
        if event_type:
            return len(self._handlers.get(event_type, []))
        return sum(len(h) for h in self._handlers.values()) + len(self._global_handlers)
