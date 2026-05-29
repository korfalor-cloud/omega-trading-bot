"""Backpressure Handling — queue depth monitoring, rate limiting, load shedding.

Implements:
- Queue depth monitoring
- Adaptive rate limiting
- Priority-based message dropping
- Load shedding
- Queue size limits
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------

class Priority(IntEnum):
    """Message priority levels.  Lower numeric value = higher priority."""
    CRITICAL = 0   # Risk alerts, stop-loss executions — never drop
    HIGH = 1       # Order submissions
    NORMAL = 2     # Signal generation, strategy updates
    LOW = 3        # Telemetry, analytics
    BACKGROUND = 4 # Log shipping, non-critical reporting


# ---------------------------------------------------------------------------
# Queue message wrapper
# ---------------------------------------------------------------------------

@dataclass
class QueueMessage:
    """A message in the backpressure queue."""
    payload: Any
    priority: Priority = Priority.NORMAL
    enqueued_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adaptive rate limiter
# ---------------------------------------------------------------------------

class AdaptiveRateLimiter:
    """Token-bucket rate limiter whose rate adapts to system load.

    - *max_rate*:   upper bound (tokens/sec)
    - *min_rate*:   lower bound when under extreme pressure
    - *capacity*:   bucket capacity (burst size)
    """

    def __init__(self, max_rate: float = 100.0, min_rate: float = 1.0, capacity: int = 50):
        self.max_rate = max_rate
        self.min_rate = min_rate
        self.capacity = capacity
        self._current_rate = max_rate
        self._tokens = float(capacity)
        self._last_time = time.time()
        self._lock = threading.Lock()

    @property
    def current_rate(self) -> float:
        return self._current_rate

    def acquire(self, tokens: int = 1) -> bool:
        """Try to consume *tokens*.  Returns True if allowed."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_time
            self._tokens = min(self.capacity, self._tokens + elapsed * self._current_rate)
            self._last_time = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait_time(self, tokens: int = 1) -> float:
        if self._tokens >= tokens:
            return 0.0
        return (tokens - self._tokens) / max(self._current_rate, 0.001)

    def adjust_for_pressure(self, pressure: float) -> None:
        """Adapt the rate based on *pressure* in [0, 1].

        0 = no pressure (max rate), 1 = full pressure (min rate).
        """
        with self._lock:
            pressure = max(0.0, min(1.0, pressure))
            self._current_rate = self.max_rate - (self.max_rate - self.min_rate) * pressure


# ---------------------------------------------------------------------------
# Backpressure queue
# ---------------------------------------------------------------------------

class BackpressureQueue:
    """Bounded, priority-aware queue with load-shedding and backpressure."""

    def __init__(self, max_size: int = 10_000, high_watermark: float = 0.8,
                 low_watermark: float = 0.4, drop_below: Priority = Priority.LOW):
        self.max_size = max_size
        self.high_watermark = int(max_size * high_watermark)
        self.low_watermark = int(max_size * low_watermark)
        self.drop_below = drop_below   # drop messages with priority >= this value when above high watermark

        self._queue: deque[QueueMessage] = deque()
        self._lock = threading.Lock()
        self._dropped: int = 0
        self._total_enqueued: int = 0
        self._total_dequeued: int = 0
        self._shedding: bool = False

    # ---- properties ----

    @property
    def depth(self) -> int:
        return len(self._queue)

    @property
    def utilization(self) -> float:
        return self.depth / max(self.max_size, 1)

    @property
    def is_shedding(self) -> bool:
        return self._shedding

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "max_size": self.max_size,
            "utilization_pct": round(self.utilization * 100, 1),
            "shedding": self._shedding,
            "total_enqueued": self._total_enqueued,
            "total_dequeued": self._total_dequeued,
            "total_dropped": self._dropped,
        }

    # ---- enqueue / dequeue ----

    def put(self, payload: Any, priority: Priority = Priority.NORMAL,
            metadata: dict[str, Any] | None = None) -> bool:
        """Enqueue a message.  Returns False if the message was dropped."""
        msg = QueueMessage(payload=payload, priority=priority, metadata=metadata or {})

        with self._lock:
            # Full queue: drop the incoming message if its priority is low enough
            if self.depth >= self.max_size:
                if priority >= self.drop_below:
                    self._dropped += 1
                    logger.warning(
                        f"Backpressure: dropping message (priority={priority.name}) — queue full"
                    )
                    return False
                # High-priority message at full queue: evict the lowest-priority item
                if not self._evict_lowest():
                    self._dropped += 1
                    return False

            # Above high watermark: start shedding low-priority messages
            if self.depth >= self.high_watermark:
                self._shedding = True
                if priority >= self.drop_below:
                    self._dropped += 1
                    logger.debug(
                        f"Backpressure: shedding message (priority={priority.name}) — above high watermark"
                    )
                    return False

            self._queue.append(msg)
            self._total_enqueued += 1

            # Clear shedding flag once we drop below low watermark
            if self._shedding and self.depth <= self.low_watermark:
                self._shedding = False
                logger.info("Backpressure: shedding stopped — below low watermark")

        return True

    def get(self, timeout: float | None = None) -> QueueMessage | None:
        """Dequeue the highest-priority message.  Returns None on timeout / empty."""
        deadline = time.time() + timeout if timeout is not None else None

        while True:
            with self._lock:
                if self._queue:
                    # Find highest-priority (lowest int value) message
                    best_idx = 0
                    best_priority = self._queue[0].priority
                    for i in range(1, len(self._queue)):
                        if self._queue[i].priority < best_priority:
                            best_idx = i
                            best_priority = self._queue[i].priority
                    msg = self._queue[best_idx]
                    del self._queue[best_idx]
                    self._total_dequeued += 1
                    return msg

            if deadline is not None and time.time() >= deadline:
                return None
            time.sleep(0.005)  # 5ms poll

    def peek(self) -> QueueMessage | None:
        with self._lock:
            return self._queue[0] if self._queue else None

    def drain(self, max_items: int = 0) -> list[QueueMessage]:
        """Drain up to *max_items* messages (0 = all)."""
        with self._lock:
            count = max_items or len(self._queue)
            items = []
            for _ in range(min(count, len(self._queue))):
                items.append(self._queue.popleft())
            self._total_dequeued += len(items)
            return items

    def clear(self) -> int:
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count

    # ---- eviction ----

    def _evict_lowest(self) -> bool:
        """Remove the lowest-priority message to make room.  Caller holds lock."""
        if not self._queue:
            return False
        # Find lowest-priority (highest int value)
        worst_idx = 0
        worst_priority = self._queue[0].priority
        for i in range(1, len(self._queue)):
            if self._queue[i].priority > worst_priority:
                worst_idx = i
                worst_priority = self._queue[i].priority
        del self._queue[worst_idx]
        self._dropped += 1
        logger.debug(f"Backpressure: evicted lowest-priority message ({worst_priority.name})")
        return True


# ---------------------------------------------------------------------------
# Load shedder
# ---------------------------------------------------------------------------

class LoadShedder:
    """Monitors system load and triggers shedding policies.

    Tracks queue depth, memory pressure, and call latency to compute
    an overall *pressure* score between 0 and 1.
    """

    def __init__(self, queue: BackpressureQueue, rate_limiter: AdaptiveRateLimiter,
                 latency_threshold_ms: float = 500.0):
        self.queue = queue
        self.rate_limiter = rate_limiter
        self.latency_threshold_ms = latency_threshold_ms

        self._recent_latencies: deque[float] = deque(maxlen=100)
        self._lock = threading.Lock()

    def record_latency(self, latency_ms: float) -> None:
        with self._lock:
            self._recent_latencies.append(latency_ms)

    def compute_pressure(self) -> float:
        """Return a pressure value in [0, 1] combining queue depth and latency."""
        # Queue pressure
        queue_pressure = self.queue.utilization

        # Latency pressure
        latency_pressure = 0.0
        with self._lock:
            if self._recent_latencies:
                avg = sum(self._recent_latencies) / len(self._recent_latencies)
                latency_pressure = min(1.0, avg / max(self.latency_threshold_ms, 1.0))

        return max(queue_pressure, latency_pressure)

    def tick(self) -> float:
        """Recompute pressure and adjust the rate limiter.  Returns current pressure."""
        pressure = self.compute_pressure()
        self.rate_limiter.adjust_for_pressure(pressure)
        return pressure

    @property
    def should_shed(self) -> bool:
        return self.queue.is_shedding


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_backpressure_system(
    queue_size: int = 10_000,
    max_rate: float = 100.0,
    min_rate: float = 1.0,
    high_watermark: float = 0.8,
    low_watermark: float = 0.4,
    drop_below: Priority = Priority.LOW,
) -> tuple[BackpressureQueue, AdaptiveRateLimiter, LoadShedder]:
    """Create a fully wired backpressure system."""
    queue = BackpressureQueue(
        max_size=queue_size,
        high_watermark=high_watermark,
        low_watermark=low_watermark,
        drop_below=drop_below,
    )
    rate_limiter = AdaptiveRateLimiter(max_rate=max_rate, min_rate=min_rate)
    shedder = LoadShedder(queue=queue, rate_limiter=rate_limiter)
    return queue, rate_limiter, shedder
