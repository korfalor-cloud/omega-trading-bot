"""Structured Logging — JSON logs with context.

Implements:
- JSON-formatted log output
- Structured fields (strategy, symbol, trade_id)
- Log levels with filtering
- File and console output
- Log rotation
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields
        if hasattr(record, "strategy_id"):
            log_entry["strategy_id"] = record.strategy_id
        if hasattr(record, "symbol"):
            log_entry["symbol"] = record.symbol
        if hasattr(record, "trade_id"):
            log_entry["trade_id"] = record.trade_id
        if hasattr(record, "pnl"):
            log_entry["pnl"] = record.pnl
        if hasattr(record, "signal_type"):
            log_entry["signal_type"] = record.signal_type

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class StructuredLogger:
    """Structured logging with context."""

    def __init__(self, name: str, config: dict = None):
        config = config or {}
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.get("level", "INFO")))

        # Console handler
        if config.get("console", True):
            console = logging.StreamHandler(sys.stdout)
            if config.get("json", False):
                console.setFormatter(JSONFormatter())
            else:
                console.setFormatter(logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
                ))
            self.logger.addHandler(console)

        # File handler
        log_file = config.get("file")
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            from logging.handlers import RotatingFileHandler
            handler = RotatingFileHandler(
                log_file,
                maxBytes=config.get("max_bytes", 10 * 1024 * 1024),
                backupCount=config.get("backup_count", 5),
            )
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)

    def trade(self, msg: str, strategy_id: str = "", symbol: str = "", trade_id: str = "", pnl: float = 0, **kwargs):
        extra = {"strategy_id": strategy_id, "symbol": symbol, "trade_id": trade_id, "pnl": pnl, **kwargs}
        self.logger.info(msg, extra=extra)

    def signal(self, msg: str, strategy_id: str = "", symbol: str = "", signal_type: str = "", **kwargs):
        extra = {"strategy_id": strategy_id, "symbol": symbol, "signal_type": signal_type, **kwargs}
        self.logger.info(msg, extra=extra)

    def risk(self, msg: str, **kwargs):
        self.logger.warning(msg, extra=kwargs)

    def error(self, msg: str, **kwargs):
        self.logger.error(msg, extra=kwargs)

    def info(self, msg: str, **kwargs):
        self.logger.info(msg, extra=kwargs)

    def debug(self, msg: str, **kwargs):
        self.logger.debug(msg, extra=kwargs)


def setup_logging(config: dict = None) -> StructuredLogger:
    """Setup project-wide logging."""
    config = config or {}
    return StructuredLogger("omega", config)
