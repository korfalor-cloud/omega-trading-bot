"""Prometheus-Compatible Metrics — counters, gauges, histograms, summaries.

Implements:
- Counter (trades, orders, errors)
- Gauge (positions, equity, drawdown)
- Histogram (latency, slippage)
- Summary (P&L distribution)
- Metrics registry
- Export endpoint (/metrics)
- Labels (strategy, symbol, exchange)
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """Sanitize a metric name to Prometheus-compatible format."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _format_labels(labels: dict[str, str]) -> str:
    """Format labels dict into Prometheus label string."""
    if not labels:
        return ""
    parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
    return "{" + ",".join(parts) + "}"


# ---------------------------------------------------------------------------
# Metric types
# ---------------------------------------------------------------------------

class Counter:
    """Monotonically increasing counter."""

    def __init__(self, name: str, documentation: str = "", label_names: list[str] | None = None):
        self.name = _sanitize(name)
        self.documentation = documentation
        self.label_names = label_names or []
        self._values: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0, **labels: str) -> None:
        key = self._label_key(labels)
        with self._lock:
            self._values[key] += value

    def labels(self, **labels: str) -> _BoundCounter:
        return _BoundCounter(self, labels)

    def get(self, **labels: str) -> float:
        key = self._label_key(labels)
        return self._values.get(key, 0.0)

    def collect(self) -> list[str]:
        lines = [
            f"# HELP {self.name} {self.documentation}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            for key, value in self._values.items():
                lines.append(f"{self.name}{key} {value}")
        return lines

    def _label_key(self, labels: dict[str, str]) -> str:
        if not self.label_names:
            return ""
        filtered = {k: labels.get(k, "") for k in self.label_names}
        return _format_labels(filtered)


class _BoundCounter:
    """Counter pre-bound to specific label values."""

    def __init__(self, counter: Counter, labels: dict[str, str]):
        self._counter = counter
        self._labels = labels

    def inc(self, value: float = 1.0) -> None:
        self._counter.inc(value, **self._labels)

    @property
    def value(self) -> float:
        return self._counter.get(**self._labels)


class Gauge:
    """Metric that can go up and down."""

    def __init__(self, name: str, documentation: str = "", label_names: list[str] | None = None):
        self.name = _sanitize(name)
        self.documentation = documentation
        self.label_names = label_names or []
        self._values: dict[str, float] = defaultdict(float)
        self._timestamps: dict[str, float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, **labels: str) -> None:
        key = self._label_key(labels)
        with self._lock:
            self._values[key] = value
            self._timestamps[key] = time.time()

    def inc(self, value: float = 1.0, **labels: str) -> None:
        key = self._label_key(labels)
        with self._lock:
            self._values[key] += value

    def dec(self, value: float = 1.0, **labels: str) -> None:
        key = self._label_key(labels)
        with self._lock:
            self._values[key] -= value

    def labels(self, **labels: str) -> _BoundGauge:
        return _BoundGauge(self, labels)

    def get(self, **labels: str) -> float:
        key = self._label_key(labels)
        return self._values.get(key, 0.0)

    def collect(self) -> list[str]:
        lines = [
            f"# HELP {self.name} {self.documentation}",
            f"# TYPE {self.name} gauge",
        ]
        with self._lock:
            for key, value in self._values.items():
                lines.append(f"{self.name}{key} {value}")
        return lines

    def _label_key(self, labels: dict[str, str]) -> str:
        if not self.label_names:
            return ""
        filtered = {k: labels.get(k, "") for k in self.label_names}
        return _format_labels(filtered)


class _BoundGauge:
    """Gauge pre-bound to specific label values."""

    def __init__(self, gauge: Gauge, labels: dict[str, str]):
        self._gauge = gauge
        self._labels = labels

    def set(self, value: float) -> None:
        self._gauge.set(value, **self._labels)

    def inc(self, value: float = 1.0) -> None:
        self._gauge.inc(value, **self._labels)

    def dec(self, value: float = 1.0) -> None:
        self._gauge.dec(value, **self._labels)

    @property
    def value(self) -> float:
        return self._gauge.get(**self._labels)


class Histogram:
    """Observation histogram with configurable buckets."""

    DEFAULT_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(self, name: str, documentation: str = "", label_names: list[str] | None = None,
                 buckets: tuple[float, ...] | None = None):
        self.name = _sanitize(name)
        self.documentation = documentation
        self.label_names = label_names or []
        self.buckets = buckets or self.DEFAULT_BUCKETS
        # Per label-set: {"le_0.1": count, ..., "_sum": sum, "_count": count}
        self._data: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: str) -> None:
        key = self._label_key(labels)
        with self._lock:
            data = self._data[key]
            data["_sum"] += value
            data["_count"] += 1
            for bucket in self.buckets:
                if value <= bucket:
                    data[f"le_{bucket}"] += 1
            data["le_+Inf"] += 1

    def labels(self, **labels: str) -> _BoundHistogram:
        return _BoundHistogram(self, labels)

    def collect(self) -> list[str]:
        lines = [
            f"# HELP {self.name} {self.documentation}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            for key, data in self._data.items():
                for bucket in self.buckets:
                    count = data.get(f"le_{bucket}", 0)
                    bucket_label = key.rstrip("}") + f',le="{bucket}"}}' if key else f'{{le="{bucket}"}}'
                    if key:
                        base = key[:-1]  # remove trailing }
                        bucket_label = f'{base},le="{bucket}"}}'
                    else:
                        bucket_label = f'{{le="{bucket}"}}'
                    lines.append(f"{self.name}_bucket{bucket_label} {count}")
                inf_label = key.rstrip("}") + ',le="+Inf"}' if key else '{le="+Inf"}'
                if key:
                    base = key[:-1]
                    inf_label = f'{base},le="+Inf"}}'
                else:
                    inf_label = '{le="+Inf"}'
                lines.append(f"{self.name}_bucket{inf_label} {data.get('le_+Inf', 0)}")
                lines.append(f"{self.name}_sum{key} {data.get('_sum', 0)}")
                lines.append(f"{self.name}_count{key} {data.get('_count', 0)}")
        return lines

    def _label_key(self, labels: dict[str, str]) -> str:
        if not self.label_names:
            return ""
        filtered = {k: labels.get(k, "") for k in self.label_names}
        return _format_labels(filtered)


class _BoundHistogram:
    """Histogram pre-bound to specific label values."""

    def __init__(self, histogram: Histogram, labels: dict[str, str]):
        self._histogram = histogram
        self._labels = labels

    def observe(self, value: float) -> None:
        self._histogram.observe(value, **self._labels)


class Summary:
    """Tracks count, sum and quantiles for a distribution."""

    def __init__(self, name: str, documentation: str = "", label_names: list[str] | None = None,
                 max_age_seconds: float = 600.0, age_buckets: int = 5):
        self.name = _sanitize(name)
        self.documentation = documentation
        self.label_names = label_names or []
        self.max_age_seconds = max_age_seconds
        self.age_buckets = age_buckets
        # Per label-set: list of (timestamp, value) observations
        self._observations: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: str) -> None:
        key = self._label_key(labels)
        now = time.time()
        with self._lock:
            obs = self._observations[key]
            obs.append((now, value))
            # Evict old observations
            cutoff = now - self.max_age_seconds
            self._observations[key] = [(t, v) for t, v in obs if t >= cutoff]

    def labels(self, **labels: str) -> _BoundSummary:
        return _BoundSummary(self, labels)

    def _get_quantile(self, key: str, quantile: float) -> float:
        obs = self._observations.get(key, [])
        if not obs:
            return 0.0
        values = sorted(v for _, v in obs)
        idx = int(quantile * (len(values) - 1))
        return values[idx]

    def collect(self) -> list[str]:
        quantiles = (0.5, 0.9, 0.95, 0.99)
        lines = [
            f"# HELP {self.name} {self.documentation}",
            f"# TYPE {self.name} summary",
        ]
        with self._lock:
            for key, obs in self._observations.items():
                for q in quantiles:
                    value = self._get_quantile(key, q)
                    if key:
                        base = key[:-1]
                        q_label = f'{base},quantile="{q}"}}'
                    else:
                        q_label = f'{{quantile="{q}"}}'
                    lines.append(f"{self.name}{q_label} {value}")
                total = sum(v for _, v in obs)
                count = len(obs)
                lines.append(f"{self.name}_sum{key} {total}")
                lines.append(f"{self.name}_count{key} {count}")
        return lines

    def _label_key(self, labels: dict[str, str]) -> str:
        if not self.label_names:
            return ""
        filtered = {k: labels.get(k, "") for k in self.label_names}
        return _format_labels(filtered)


class _BoundSummary:
    """Summary pre-bound to specific label values."""

    def __init__(self, summary: Summary, labels: dict[str, str]):
        self._summary = summary
        self._labels = labels

    def observe(self, value: float) -> None:
        self._summary.observe(value, **self._labels)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class MetricsRegistry:
    """Central registry for all metrics, handles export."""

    def __init__(self):
        self._metrics: list[Counter | Gauge | Histogram | Summary] = []
        self._by_name: dict[str, Counter | Gauge | Histogram | Summary] = {}
        self._lock = threading.Lock()

    def register(self, metric: Counter | Gauge | Histogram | Summary) -> None:
        with self._lock:
            if metric.name in self._by_name:
                logger.warning(f"Metric '{metric.name}' already registered, skipping")
                return
            self._metrics.append(metric)
            self._by_name[metric.name] = metric

    def counter(self, name: str, documentation: str = "", label_names: list[str] | None = None) -> Counter:
        c = Counter(name, documentation, label_names)
        self.register(c)
        return c

    def gauge(self, name: str, documentation: str = "", label_names: list[str] | None = None) -> Gauge:
        g = Gauge(name, documentation, label_names)
        self.register(g)
        return g

    def histogram(self, name: str, documentation: str = "", label_names: list[str] | None = None,
                  buckets: tuple[float, ...] | None = None) -> Histogram:
        h = Histogram(name, documentation, label_names, buckets)
        self.register(h)
        return h

    def summary(self, name: str, documentation: str = "", label_names: list[str] | None = None) -> Summary:
        s = Summary(name, documentation, label_names)
        self.register(s)
        return s

    def get(self, name: str) -> Optional[Counter | Gauge | Histogram | Summary]:
        return self._by_name.get(name)

    def collect_all(self) -> str:
        """Export all metrics in Prometheus text format."""
        parts: list[str] = []
        with self._lock:
            for metric in self._metrics:
                parts.append("\n".join(metric.collect()))
        return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Default registry + trading metrics
# ---------------------------------------------------------------------------

REGISTRY = MetricsRegistry()

# Counters
TRADES_TOTAL = REGISTRY.counter(
    "omega_trades_total", "Total trades executed",
    label_names=["strategy", "symbol", "side"],
)
ORDERS_TOTAL = REGISTRY.counter(
    "omega_orders_total", "Total orders placed",
    label_names=["strategy", "symbol", "exchange", "type"],
)
ERRORS_TOTAL = REGISTRY.counter(
    "omega_errors_total", "Total errors",
    label_names=["component", "error_type"],
)

# Gauges
OPEN_POSITIONS = REGISTRY.gauge(
    "omega_open_positions", "Current open positions",
    label_names=["strategy", "symbol"],
)
EQUITY = REGISTRY.gauge(
    "omega_equity_usd", "Current account equity in USD",
)
DRAWDOWN_PCT = REGISTRY.gauge(
    "omega_drawdown_pct", "Current drawdown as a fraction",
)

# Histograms
ORDER_LATENCY = REGISTRY.histogram(
    "omega_order_latency_seconds", "Order execution latency",
    label_names=["exchange"],
)
SLIPPAGE_BPS = REGISTRY.histogram(
    "omega_slippage_bps", "Trade slippage in basis points",
    label_names=["strategy", "symbol"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0),
)

# Summary
PNL_DISTRIBUTION = REGISTRY.summary(
    "omega_trade_pnl_usd", "Distribution of per-trade P&L in USD",
    label_names=["strategy"],
)


# ---------------------------------------------------------------------------
# /metrics HTTP endpoint
# ---------------------------------------------------------------------------

class _MetricsHandler(BaseHTTPRequestHandler):
    """Lightweight HTTP handler that serves Prometheus metrics."""

    registry: MetricsRegistry = REGISTRY

    def do_GET(self) -> None:
        if self.path == "/metrics":
            body = self.registry.collect_all().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, format: str, *args) -> None:
        # Silence default stderr logging; use our logger instead.
        logger.debug(format, *args)


class MetricsServer:
    """Standalone Prometheus metrics HTTP server."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9090, registry: MetricsRegistry | None = None):
        self.host = host
        self.port = port
        self.registry = registry or REGISTRY
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        handler = type("Handler", (_MetricsHandler,), {"registry": self.registry})
        self._server = HTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(f"Metrics server started on {self.host}:{self.port}/metrics")

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            logger.info("Metrics server stopped")
