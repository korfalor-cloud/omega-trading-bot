from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

Subscriber = Callable[[Any], Coroutine[Any, Any, None]]


class Event:
    """Event topic constants."""
    # Market data
    BAR_CLOSED = "bar.closed"
    TICK_RECEIVED = "tick.received"
    ORDER_BOOK_UPDATE = "orderbook.update"
    # Trading
    SIGNAL_GENERATED = "signal.generated"
    ORDER_SUBMITTED = "order.submitted"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REJECTED = "order.rejected"
    POSITION_UPDATED = "position.updated"
    POSITION_CLOSED = "position.closed"
    # Risk
    RISK_ALERT = "risk.alert"
    REGIME_CHANGED = "regime.changed"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker.triggered"
    # Evolution
    STRATEGY_CREATED = "strategy.created"
    STRATEGY_PROMOTED = "strategy.promoted"
    STRATEGY_RETIRED = "strategy.retired"
    EVOLUTION_CYCLE_COMPLETE = "evolution.cycle_complete"
    GENERATION_COMPLETE = "evolution.generation_complete"
    # World model
    WORLD_MODEL_UPDATED = "world_model.updated"
    CAUSAL_GRAPH_UPDATED = "causal_graph.updated"
    SCENARIO_GENERATED = "scenario.generated"
    # Consciousness
    GOAL_SET = "consciousness.goal_set"
    REFLECTION_COMPLETE = "consciousness.reflection_complete"
    UNCERTAINTY_HIGH = "consciousness.uncertainty_high"
    # System
    STRATEGY_STARTED = "strategy.started"
    STRATEGY_STOPPED = "strategy.stopped"
    MODEL_PREDICTION = "model.prediction"
    DATA_GAP_DETECTED = "data.gap_detected"
    CONNECTION_LOST = "connection.lost"
    CONNECTION_RESTORED = "connection.restored"
    HEARTBEAT = "system.heartbeat"
    SHUTDOWN = "system.shutdown"


class EventBus:
    """Async pub/sub event bus with topic routing and priority handlers."""

    def __init__(self, max_queue_size: int = 100_000):
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._priority_subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._stats: dict[str, int] = defaultdict(int)

    def subscribe(self, topic: str, handler: Subscriber, priority: bool = False) -> None:
        if priority:
            self._priority_subscribers[topic].append(handler)
        else:
            self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Subscriber) -> None:
        try:
            self._subscribers[topic].remove(handler)
        except ValueError:
            try:
                self._priority_subscribers[topic].remove(handler)
            except ValueError:
                pass

    async def publish(self, topic: str, payload: Any) -> None:
        self._stats[topic] += 1
        try:
            self._queue.put_nowait((topic, payload))
        except asyncio.QueueFull:
            logger.error(f"EventBus queue full! Dropping event: {topic}")

    async def run(self) -> None:
        self._running = True
        while self._running:
            try:
                topic, payload = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            for handler in self._priority_subscribers.get(topic, []):
                try:
                    await handler(payload)
                except Exception:
                    logger.exception(f"Priority handler error on topic={topic}")

            for handler in self._subscribers.get(topic, []):
                try:
                    await handler(payload)
                except Exception:
                    logger.exception(f"Handler error on topic={topic}")

    async def stop(self) -> None:
        self._running = False

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()
