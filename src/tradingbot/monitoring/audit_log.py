"""Audit Log — Full action audit trail for compliance and debugging.

Implements:
- Trade logging (entries, exits, modifications)
- Config change logging (before/after values)
- Strategy lifecycle logging (create, start, pause, stop)
- Query and filter capabilities (by type, time, strategy, symbol)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of auditable events."""
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    TRADE_MODIFY = "trade_modify"
    CONFIG_CHANGE = "config_change"
    STRATEGY_CREATED = "strategy_created"
    STRATEGY_STARTED = "strategy_started"
    STRATEGY_PAUSED = "strategy_paused"
    STRATEGY_STOPPED = "strategy_stopped"
    STRATEGY_DELETED = "strategy_deleted"
    RISK_BREACH = "risk_breach"
    MANUAL_ACTION = "manual_action"
    SYSTEM_EVENT = "system_event"


@dataclass
class AuditEntry:
    """A single audit log entry."""
    timestamp: float = 0.0
    event_type: str = ""
    strategy_id: str = ""
    symbol: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    user: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "details": self.details,
            "user": self.user,
            "message": self.message,
        }


class AuditLog:
    """Append-only audit log with query and filter capabilities.

    Usage:
        log = AuditLog()
        log.log_trade("strat_1", "BTC/USDT", side="buy", qty=0.5, price=30000)
        log.log_config_change("strat_1", "risk_per_trade", 0.01, 0.02)
        entries = log.query(strategy_id="strat_1", event_type="trade_entry")
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.max_entries = config.get("max_entries", 100_000)
        self._entries: list[AuditEntry] = []
        self._index_by_strategy: dict[str, list[int]] = {}
        self._index_by_type: dict[str, list[int]] = {}
        self._index_by_symbol: dict[str, list[int]] = {}

    def _append(self, entry: AuditEntry) -> None:
        """Add entry to log and update indices."""
        idx = len(self._entries)
        self._entries.append(entry)

        # Update indices
        if entry.strategy_id:
            self._index_by_strategy.setdefault(entry.strategy_id, []).append(idx)
        self._index_by_type.setdefault(entry.event_type, []).append(idx)
        if entry.symbol:
            self._index_by_symbol.setdefault(entry.symbol, []).append(idx)

        # Evict oldest if over capacity
        if len(self._entries) > self.max_entries:
            self._evict(len(self._entries) - self.max_entries)

    def _evict(self, n: int) -> None:
        """Remove the oldest n entries and rebuild indices."""
        self._entries = self._entries[n:]
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        self._index_by_strategy.clear()
        self._index_by_type.clear()
        self._index_by_symbol.clear()
        for idx, entry in enumerate(self._entries):
            if entry.strategy_id:
                self._index_by_strategy.setdefault(entry.strategy_id, []).append(idx)
            self._index_by_type.setdefault(entry.event_type, []).append(idx)
            if entry.symbol:
                self._index_by_symbol.setdefault(entry.symbol, []).append(idx)

    # ── Trade Logging ─────────────────────────────────────────────

    def log_trade_entry(
        self,
        strategy_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_id: str = "",
        **kwargs: Any,
    ) -> AuditEntry:
        """Log a trade entry event."""
        entry = AuditEntry(
            timestamp=time.time(),
            event_type=AuditEventType.TRADE_ENTRY.value,
            strategy_id=strategy_id,
            symbol=symbol,
            details={
                "side": side,
                "quantity": quantity,
                "price": price,
                "order_id": order_id,
                **kwargs,
            },
            message=f"Trade entry: {side} {quantity} {symbol} @ {price}",
        )
        self._append(entry)
        logger.debug("Audit: trade entry logged for %s", strategy_id)
        return entry

    def log_trade_exit(
        self,
        strategy_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: float = 0.0,
        order_id: str = "",
        **kwargs: Any,
    ) -> AuditEntry:
        """Log a trade exit event."""
        entry = AuditEntry(
            timestamp=time.time(),
            event_type=AuditEventType.TRADE_EXIT.value,
            strategy_id=strategy_id,
            symbol=symbol,
            details={
                "side": side,
                "quantity": quantity,
                "price": price,
                "pnl": pnl,
                "order_id": order_id,
                **kwargs,
            },
            message=f"Trade exit: {side} {quantity} {symbol} @ {price} (P&L: {pnl:.2f})",
        )
        self._append(entry)
        return entry

    def log_trade_modify(
        self,
        strategy_id: str,
        symbol: str,
        order_id: str,
        field_name: str,
        old_value: Any,
        new_value: Any,
    ) -> AuditEntry:
        """Log a trade/order modification."""
        entry = AuditEntry(
            timestamp=time.time(),
            event_type=AuditEventType.TRADE_MODIFY.value,
            strategy_id=strategy_id,
            symbol=symbol,
            details={
                "order_id": order_id,
                "field": field_name,
                "old_value": old_value,
                "new_value": new_value,
            },
            message=f"Order {order_id} modified: {field_name} {old_value} -> {new_value}",
        )
        self._append(entry)
        return entry

    # ── Config Change Logging ─────────────────────────────────────

    def log_config_change(
        self,
        strategy_id: str,
        parameter: str,
        old_value: Any,
        new_value: Any,
        user: str = "",
    ) -> AuditEntry:
        """Log a configuration change."""
        entry = AuditEntry(
            timestamp=time.time(),
            event_type=AuditEventType.CONFIG_CHANGE.value,
            strategy_id=strategy_id,
            details={
                "parameter": parameter,
                "old_value": old_value,
                "new_value": new_value,
            },
            user=user,
            message=f"Config change: {parameter} {old_value} -> {new_value}",
        )
        self._append(entry)
        return entry

    # ── Strategy Lifecycle Logging ────────────────────────────────

    def log_lifecycle(
        self,
        strategy_id: str,
        event_type: AuditEventType,
        reason: str = "",
        user: str = "",
        **kwargs: Any,
    ) -> AuditEntry:
        """Log a strategy lifecycle event."""
        entry = AuditEntry(
            timestamp=time.time(),
            event_type=event_type.value,
            strategy_id=strategy_id,
            details={"reason": reason, **kwargs},
            user=user,
            message=f"Strategy {event_type.value}: {strategy_id} ({reason})",
        )
        self._append(entry)
        return entry

    # ── System Events ─────────────────────────────────────────────

    def log_risk_breach(
        self,
        strategy_id: str,
        breach_type: str,
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log a risk limit breach."""
        entry = AuditEntry(
            timestamp=time.time(),
            event_type=AuditEventType.RISK_BREACH.value,
            strategy_id=strategy_id,
            details=details or {"breach_type": breach_type},
            message=f"Risk breach: {breach_type}",
        )
        self._append(entry)
        return entry

    def log_manual_action(
        self,
        action: str,
        user: str = "",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log a manual user action."""
        entry = AuditEntry(
            timestamp=time.time(),
            event_type=AuditEventType.MANUAL_ACTION.value,
            details=details or {},
            user=user,
            message=f"Manual action: {action}",
        )
        self._append(entry)
        return entry

    def log_system_event(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log a system-level event."""
        entry = AuditEntry(
            timestamp=time.time(),
            event_type=AuditEventType.SYSTEM_EVENT.value,
            details=details or {},
            message=message,
        )
        self._append(entry)
        return entry

    # ── Query / Filter ────────────────────────────────────────────

    def query(
        self,
        strategy_id: str | None = None,
        event_type: str | None = None,
        symbol: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit entries with optional filters.

        Uses indices for strategy_id, event_type, and symbol for
        efficient lookup, then applies time filters.
        """
        # Start with the narrowest indexed set
        candidate_indices: set[int] | None = None

        if strategy_id is not None:
            ids = set(self._index_by_strategy.get(strategy_id, []))
            candidate_indices = ids if candidate_indices is None else candidate_indices & ids

        if event_type is not None:
            ids = set(self._index_by_type.get(event_type, []))
            candidate_indices = ids if candidate_indices is None else candidate_indices & ids

        if symbol is not None:
            ids = set(self._index_by_symbol.get(symbol, []))
            candidate_indices = ids if candidate_indices is None else candidate_indices & ids

        # Fall back to all entries if no index filter
        if candidate_indices is None:
            candidate_indices = set(range(len(self._entries)))

        # Apply time filters
        results: list[AuditEntry] = []
        for idx in sorted(candidate_indices, reverse=True):
            if len(results) >= limit:
                break
            entry = self._entries[idx]
            if start_time is not None and entry.timestamp < start_time:
                continue
            if end_time is not None and entry.timestamp > end_time:
                continue
            results.append(entry)

        return results

    def count(
        self,
        strategy_id: str | None = None,
        event_type: str | None = None,
    ) -> int:
        """Count entries matching filters."""
        if strategy_id and event_type:
            s = set(self._index_by_strategy.get(strategy_id, []))
            t = set(self._index_by_type.get(event_type, []))
            return len(s & t)
        if strategy_id:
            return len(self._index_by_strategy.get(strategy_id, []))
        if event_type:
            return len(self._index_by_type.get(event_type, []))
        return len(self._entries)

    # ── Export ─────────────────────────────────────────────────────

    def export_json(self, filepath: str, limit: int = 0) -> int:
        """Export audit log as JSON. Returns number of entries written."""
        entries = self._entries[-limit:] if limit > 0 else self._entries
        data = [e.to_dict() for e in entries]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Exported %d audit entries to %s", len(data), filepath)
        return len(data)

    def get_stats(self) -> dict:
        """Get audit log statistics."""
        type_counts = {t.value: 0 for t in AuditEventType}
        for entry in self._entries:
            if entry.event_type in type_counts:
                type_counts[entry.event_type] += 1

        return {
            "total_entries": len(self._entries),
            "by_type": {k: v for k, v in type_counts.items() if v > 0},
            "n_strategies": len(self._index_by_strategy),
            "n_symbols": len(self._index_by_symbol),
        }
