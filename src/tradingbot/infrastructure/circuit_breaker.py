"""Circuit Breaker — protect downstream services from cascading failures.

Implements:
- States: closed, open, half-open
- Failure counting
- Automatic recovery
- Configurable thresholds
- Per-service circuit breakers
- Fallback actions
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    CLOSED = "closed"          # Normal operation, requests pass through
    OPEN = "open"              # Requests are rejected; waiting for recovery
    HALF_OPEN = "half_open"    # Probe requests allowed; testing recovery


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CircuitBreakerConfig:
    """Tunable thresholds for a single circuit breaker."""
    failure_threshold: int = 5          # consecutive failures to open
    recovery_timeout: float = 30.0      # seconds before half-open probe
    half_open_max_calls: int = 3        # successful calls to close again
    success_threshold: int = 2          # consecutive successes in half-open to close
    window_seconds: float = 60.0        # rolling window for failure counting
    exclude_exceptions: tuple[type[Exception], ...] = ()  # do NOT count as failures


# ---------------------------------------------------------------------------
# Result / errors
# ---------------------------------------------------------------------------

class CircuitBreakerError(Exception):
    """Raised when the circuit is open and a call is rejected."""

    def __init__(self, name: str, state: CircuitState):
        self.name = name
        self.state = state
        super().__init__(f"Circuit breaker '{name}' is {state.value}")


@dataclass
class CallResult:
    """Outcome of a call through the circuit breaker."""
    success: bool
    value: Any = None
    error: Exception | None = None
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """A single circuit breaker guarding one service / endpoint."""

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None,
                 fallback: Callable[..., Any] | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._fallback = fallback

        self._state = CircuitState.CLOSED
        self._failure_times: list[float] = []
        self._success_count: int = 0
        self._half_open_calls: int = 0
        self._last_failure_time: float = 0.0
        self._opened_at: float = 0.0
        self._lock = threading.Lock()

        # Stats
        self._total_calls: int = 0
        self._total_failures: int = 0
        self._total_rejected: int = 0
        self._consecutive_failures: int = 0

    # ---- properties ----

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition()
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            self._prune_old_failures()
            return len(self._failure_times)

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_rejected": self._total_rejected,
            "consecutive_failures": self._consecutive_failures,
            "failure_count_in_window": self.failure_count,
        }

    # ---- state transitions ----

    def _maybe_transition(self) -> None:
        """Internal: check if time-based transitions should fire (caller holds lock)."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._opened_at
            if elapsed >= self.config.recovery_timeout:
                self._transition(CircuitState.HALF_OPEN)

    def _transition(self, new_state: CircuitState) -> None:
        old = self._state
        self._state = new_state
        logger.info(f"Circuit breaker '{self.name}': {old.value} -> {new_state.value}")

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0

        if new_state == CircuitState.CLOSED:
            self._failure_times.clear()
            self._success_count = 0
            self._consecutive_failures = 0

    def _prune_old_failures(self) -> None:
        cutoff = time.time() - self.config.window_seconds
        self._failure_times = [t for t in self._failure_times if t > cutoff]

    # ---- call guard ----

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *fn* through the circuit breaker, returning its result."""
        with self._lock:
            self._maybe_transition()
            state = self._state

            if state == CircuitState.OPEN:
                self._total_rejected += 1
                if self._fallback is not None:
                    logger.debug(f"Circuit '{self.name}' open — using fallback")
                    return self._fallback(*args, **kwargs)
                raise CircuitBreakerError(self.name, state)

            if state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._total_rejected += 1
                    raise CircuitBreakerError(self.name, state)
                self._half_open_calls += 1

        # Execute outside the lock
        self._total_calls += 1
        start = time.time()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            duration = (time.time() - start) * 1000
            self._record_failure(exc, duration)
            raise
        else:
            duration = (time.time() - start) * 1000
            self._record_success(duration)
            return result

    @contextmanager
    def guard(self):
        """Context-manager variant.  Yields nothing; re-raises exceptions."""
        with self._lock:
            self._maybe_transition()
            state = self._state

            if state == CircuitState.OPEN:
                self._total_rejected += 1
                if self._fallback is not None:
                    logger.debug(f"Circuit '{self.name}' open — using fallback")
                    yield
                    return
                raise CircuitBreakerError(self.name, state)

            if state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._total_rejected += 1
                    raise CircuitBreakerError(self.name, state)
                self._half_open_calls += 1

        self._total_calls += 1
        start = time.time()
        try:
            yield
        except Exception as exc:
            duration = (time.time() - start) * 1000
            self._record_failure(exc, duration)
            raise
        else:
            duration = (time.time() - start) * 1000
            self._record_success(duration)

    # ---- recording ----

    def _record_success(self, duration_ms: float) -> None:
        with self._lock:
            self._consecutive_failures = 0
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition(CircuitState.CLOSED)

    def _record_failure(self, exc: Exception, duration_ms: float) -> None:
        if self.config.exclude_exceptions and isinstance(exc, self.config.exclude_exceptions):
            return

        with self._lock:
            now = time.time()
            self._failure_times.append(now)
            self._prune_old_failures()
            self._last_failure_time = now
            self._total_failures += 1
            self._consecutive_failures += 1

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._opened_at = now
                self._transition(CircuitState.OPEN)
                return

            if self._state == CircuitState.CLOSED:
                if self._consecutive_failures >= self.config.failure_threshold:
                    self._opened_at = now
                    self._transition(CircuitState.OPEN)

    # ---- manual control ----

    def reset(self) -> None:
        """Manually close the breaker and clear counters."""
        with self._lock:
            self._transition(CircuitState.CLOSED)

    def trip(self) -> None:
        """Manually open the breaker."""
        with self._lock:
            self._opened_at = time.time()
            self._transition(CircuitState.OPEN)


# ---------------------------------------------------------------------------
# Registry of per-service breakers
# ---------------------------------------------------------------------------

class CircuitBreakerRegistry:
    """Manages named circuit breakers for multiple services."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get_or_create(self, name: str, config: CircuitBreakerConfig | None = None,
                      fallback: Callable[..., Any] | None = None) -> CircuitBreaker:
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config, fallback)
            return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        return self._breakers.get(name)

    def remove(self, name: str) -> None:
        with self._lock:
            self._breakers.pop(name, None)

    def all_stats(self) -> list[dict[str, Any]]:
        return [b.stats for b in self._breakers.values()]

    def reset_all(self) -> None:
        for b in self._breakers.values():
            b.reset()


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------

REGISTRY = CircuitBreakerRegistry()


def get_circuit_breaker(name: str, config: CircuitBreakerConfig | None = None,
                        fallback: Callable[..., Any] | None = None) -> CircuitBreaker:
    """Get or create a named circuit breaker from the default registry."""
    return REGISTRY.get_or_create(name, config, fallback)
