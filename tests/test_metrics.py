"""Tests for metrics — EngineMetrics, PipelineMetrics, structured logging."""
from __future__ import annotations

import json
import time
import logging

import pytest

from tradingbot.engine.async_engine import EngineMetrics
from tradingbot.engine.order_pipeline import PipelineMetrics
from tradingbot.infrastructure.logger import JSONFormatter, StructuredLogger, setup_logging


class TestEngineMetrics:
    def test_default_counters_zero(self):
        m = EngineMetrics()
        assert m.bars_processed == 0
        assert m.ticks_processed == 0
        assert m.signals_generated == 0
        assert m.orders_submitted == 0
        assert m.orders_filled == 0
        assert m.orders_rejected == 0
        assert m.orders_cancelled == 0
        assert m.risk_checks_passed == 0
        assert m.risk_checks_failed == 0

    def test_default_gauges_zero(self):
        m = EngineMetrics()
        assert m.active_strategies == 0
        assert m.open_positions == 0
        assert m.event_queue_depth == 0
        assert m.active_data_streams == 0

    def test_snapshot_keys(self):
        m = EngineMetrics()
        snap = m.snapshot()
        expected_keys = [
            "uptime_seconds", "bars_processed", "ticks_processed",
            "signals_generated", "orders_submitted", "orders_filled",
            "orders_rejected", "orders_cancelled", "active_strategies",
        ]
        for key in expected_keys:
            assert key in snap, f"Missing key: {key}"

    def test_uptime_zero_without_start(self):
        m = EngineMetrics()
        assert m.uptime_seconds == 0.0

    def test_uptime_positive_with_start(self):
        m = EngineMetrics()
        m.engine_start_time = time.monotonic() - 5.0
        assert m.uptime_seconds >= 4.0

    def test_snapshot_values_are_numeric(self):
        m = EngineMetrics()
        m.bars_processed = 100
        m.orders_filled = 50
        snap = m.snapshot()
        assert isinstance(snap["bars_processed"], int)
        assert snap["bars_processed"] == 100
        assert snap["orders_filled"] == 50


class TestPipelineMetrics:
    def test_defaults(self):
        m = PipelineMetrics()
        assert m.signals_received == 0
        assert m.orders_created == 0
        assert m.risk_passed == 0
        assert m.risk_failed == 0
        assert m.orders_submitted == 0
        assert m.orders_filled == 0

    def test_snapshot_structure(self):
        m = PipelineMetrics()
        snap = m.snapshot()
        assert len(snap) == 10
        for key in snap:
            assert isinstance(snap[key], int)

    def test_increment_and_snapshot(self):
        m = PipelineMetrics()
        m.signals_received += 1
        m.orders_created += 1
        m.risk_passed += 1
        snap = m.snapshot()
        assert snap["signals_received"] == 1
        assert snap["orders_created"] == 1
        assert snap["risk_passed"] == 1


class TestJSONFormatter:
    def test_produces_valid_json(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test message", args=(), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["message"] == "test message"
        assert "timestamp" in data
        assert "logger" in data

    def test_includes_extra_fields(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="trade executed", args=(), exc_info=None,
        )
        record.strategy_id = "strat_1"
        record.symbol = "BTC/USDT"
        record.pnl = 150.0
        output = formatter.format(record)
        data = json.loads(output)
        assert data["strategy_id"] == "strat_1"
        assert data["symbol"] == "BTC/USDT"
        assert data["pnl"] == 150.0

    def test_exception_included(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="", lineno=0,
                msg="error occurred", args=(), exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        data = json.loads(output)
        assert "exception" in data
        assert "ValueError" in data["exception"]


class TestStructuredLogger:
    def test_creates_logger(self):
        logger = StructuredLogger("test_metrics")
        assert logger.logger is not None
        assert logger.logger.name == "test_metrics"

    def test_setup_logging(self):
        logger = setup_logging()
        assert logger.logger.name == "omega"

    def test_log_methods_exist(self):
        logger = StructuredLogger("test_methods")
        # All these methods should be callable
        assert callable(logger.trade)
        assert callable(logger.signal)
        assert callable(logger.risk)
        assert callable(logger.error)
        assert callable(logger.info)
        assert callable(logger.debug)


class TestMetricsIntegration:
    def test_engine_and_pipeline_metrics_independent(self):
        em = EngineMetrics()
        pm = PipelineMetrics()
        em.bars_processed = 100
        pm.signals_received = 50
        assert em.bars_processed != pm.signals_received

    def test_metrics_snapshot_serializable(self):
        m = EngineMetrics()
        m.bars_processed = 42
        m.orders_filled = 10
        snap = m.snapshot()
        # Should be JSON-serializable
        json_str = json.dumps(snap)
        restored = json.loads(json_str)
        assert restored["bars_processed"] == 42
