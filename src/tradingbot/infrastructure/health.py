"""Health Check System — liveness, readiness, dependency checks.

Implements:
- Component health checks (exchange, database, strategies)
- Liveness probe (/health/live)
- Readiness probe (/health/ready)
- Dependency checks
- Health status aggregation
- Auto-recovery actions
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Optional
import json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# ---------------------------------------------------------------------------
# Per-component result
# ---------------------------------------------------------------------------

@dataclass
class ComponentHealth:
    """Result of a single component health check."""
    name: str
    status: HealthStatus = HealthStatus.HEALTHY
    message: str = ""
    latency_ms: float = 0.0
    last_check: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "last_check": self.last_check,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Check function signature
# ---------------------------------------------------------------------------

# A health check callable takes no args and returns a ComponentHealth or raises.
HealthCheckFn = Callable[[], ComponentHealth]


# ---------------------------------------------------------------------------
# Health checker registry
# ---------------------------------------------------------------------------

class HealthChecker:
    """Aggregates component health checks and exposes probe endpoints."""

    def __init__(self, name: str = "omega"):
        self.name = name
        self._checks: dict[str, HealthCheckFn] = {}
        self._results: dict[str, ComponentHealth] = {}
        self._recovery_actions: dict[str, Callable[[], None]] = {}
        self._lock = threading.Lock()
        self._background_thread: threading.Thread | None = None
        self._running = False
        self._check_interval: float = 30.0

    # ---- registration ----

    def register_check(self, name: str, check_fn: HealthCheckFn,
                       recovery_fn: Callable[[], None] | None = None) -> None:
        """Register a named health check and optional auto-recovery action."""
        with self._lock:
            self._checks[name] = check_fn
            if recovery_fn is not None:
                self._recovery_actions[name] = recovery_fn
            self._results[name] = ComponentHealth(name=name, message="not yet checked")

    def remove_check(self, name: str) -> None:
        with self._lock:
            self._checks.pop(name, None)
            self._results.pop(name, None)
            self._recovery_actions.pop(name, None)

    # ---- execution ----

    def check_component(self, name: str) -> ComponentHealth:
        """Run a single named component check."""
        check_fn = self._checks.get(name)
        if check_fn is None:
            return ComponentHealth(name=name, status=HealthStatus.UNHEALTHY, message="unknown component")

        start = time.time()
        try:
            result = check_fn()
            result.latency_ms = (time.time() - start) * 1000
            result.last_check = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            result = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(exc),
                latency_ms=(time.time() - start) * 1000,
                last_check=datetime.now(timezone.utc).isoformat(),
            )
            logger.error(f"Health check '{name}' failed: {exc}")

        with self._lock:
            self._results[name] = result

        # Trigger recovery if unhealthy
        if result.status == HealthStatus.UNHEALTHY:
            self._try_recovery(name)

        return result

    def check_all(self) -> dict[str, ComponentHealth]:
        """Run every registered health check."""
        with self._lock:
            names = list(self._checks.keys())
        results = {}
        for name in names:
            results[name] = self.check_component(name)
        return results

    def _try_recovery(self, name: str) -> None:
        action = self._recovery_actions.get(name)
        if action is None:
            return
        try:
            logger.info(f"Running auto-recovery for '{name}'")
            action()
        except Exception as exc:
            logger.error(f"Auto-recovery for '{name}' failed: {exc}")

    # ---- aggregation ----

    def get_status(self) -> HealthStatus:
        """Aggregate all component statuses into one."""
        with self._lock:
            results = list(self._results.values())

        if not results:
            return HealthStatus.HEALTHY

        statuses = [r.status for r in results]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        return HealthStatus.DEGRADED

    def is_live(self) -> bool:
        """Liveness: the process is running and can handle requests."""
        # The process being alive and able to respond is sufficient.
        return True

    def is_ready(self) -> bool:
        """Readiness: all components are healthy or degraded (not unhealthy)."""
        status = self.get_status()
        return status != HealthStatus.UNHEALTHY

    # ---- JSON payload ----

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            components = [r.to_dict() for r in self._results.values()]
        return {
            "service": self.name,
            "status": self.get_status().value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": components,
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    # ---- background checking ----

    def start_background_checks(self, interval: float = 30.0) -> None:
        """Periodically run all checks in a daemon thread."""
        self._check_interval = interval
        self._running = True
        self._background_thread = threading.Thread(target=self._background_loop, daemon=True)
        self._background_thread.start()
        logger.info(f"Background health checks started (interval={interval}s)")

    def stop_background_checks(self) -> None:
        self._running = False
        if self._background_thread:
            self._background_thread.join(timeout=5)
            logger.info("Background health checks stopped")

    def _background_loop(self) -> None:
        while self._running:
            try:
                self.check_all()
            except Exception as exc:
                logger.error(f"Background health check error: {exc}")
            time.sleep(self._check_interval)


# ---------------------------------------------------------------------------
# Common health check factories
# ---------------------------------------------------------------------------

def exchange_check_fn(exchange_name: str, ping_fn: Callable[[], bool]) -> HealthCheckFn:
    """Create a health check for an exchange."""
    def _check() -> ComponentHealth:
        ok = ping_fn()
        if ok:
            return ComponentHealth(name=f"exchange:{exchange_name}", status=HealthStatus.HEALTHY, message="reachable")
        return ComponentHealth(name=f"exchange:{exchange_name}", status=HealthStatus.UNHEALTHY, message="unreachable")
    return _check


def database_check_fn(db_path: str) -> HealthCheckFn:
    """Create a health check for a SQLite database."""
    import sqlite3

    def _check() -> ComponentHealth:
        try:
            conn = sqlite3.connect(db_path, timeout=5)
            conn.execute("SELECT 1")
            conn.close()
            return ComponentHealth(name="database", status=HealthStatus.HEALTHY, message="connected")
        except Exception as exc:
            return ComponentHealth(name="database", status=HealthStatus.UNHEALTHY, message=str(exc))
    return _check


def strategy_check_fn(strategy_id: str, is_active_fn: Callable[[], bool]) -> HealthCheckFn:
    """Create a health check for a strategy."""
    def _check() -> ComponentHealth:
        active = is_active_fn()
        if active:
            return ComponentHealth(name=f"strategy:{strategy_id}", status=HealthStatus.HEALTHY, message="active")
        return ComponentHealth(name=f"strategy:{strategy_id}", status=HealthStatus.DEGRADED, message="inactive")
    return _check


# ---------------------------------------------------------------------------
# HTTP probe server
# ---------------------------------------------------------------------------

class _HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for liveness and readiness probes."""

    checker: HealthChecker = None  # type: ignore[assignment]

    def do_GET(self) -> None:
        if self.path == "/health/live":
            self._respond(200, {"status": "alive"})
        elif self.path == "/health/ready":
            if self.checker.is_ready():
                self._respond(200, self.checker.to_dict())
            else:
                self._respond(503, self.checker.to_dict())
        elif self.path == "/health":
            self._respond(200, self.checker.to_dict())
        else:
            self.send_error(404)

    def _respond(self, code: int, body: dict) -> None:
        payload = json.dumps(body, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        logger.debug(format, *args)


class HealthServer:
    """Standalone HTTP server for health probes."""

    def __init__(self, checker: HealthChecker, host: str = "0.0.0.0", port: int = 8080):
        self.checker = checker
        self.host = host
        self.port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        handler = type("Handler", (_HealthHandler,), {"checker": self.checker})
        self._server = HTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(f"Health server started on {self.host}:{self.port}")

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            logger.info("Health server stopped")
