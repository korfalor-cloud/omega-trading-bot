"""Strategy Lifecycle Management — Create, run, pause, stop strategies.

Implements:
- Strategy creation / modification / deletion
- State tracking (idle, running, paused, stopped)
- Performance monitoring per strategy
- Auto-stop on poor performance (drawdown, loss streak)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class StrategyState(Enum):
    """Strategy lifecycle states."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class StrategyConfig:
    """Configuration for a managed strategy."""
    strategy_id: str = ""
    name: str = ""
    strategy_type: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    max_drawdown_pct: float = 20.0
    max_loss_streak: int = 10
    created_at: float = 0.0


@dataclass
class StrategyRecord:
    """Internal record tracking a strategy's lifecycle."""
    config: StrategyConfig = field(default_factory=StrategyConfig)
    state: StrategyState = StrategyState.IDLE
    started_at: float = 0.0
    stopped_at: float = 0.0
    pnl: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_streak: int = 0
    max_loss_streak: int = 0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    max_drawdown_pct: float = 0.0
    pause_reason: str = ""
    stop_reason: str = ""
    state_history: list[tuple[float, str, str]] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.win_count / self.trade_count if self.trade_count > 0 else 0.0

    @property
    def uptime_seconds(self) -> float:
        if self.state == StrategyState.RUNNING:
            return time.time() - self.started_at
        if self.stopped_at > 0 and self.started_at > 0:
            return self.stopped_at - self.started_at
        return 0.0


class StrategyLifecycleManager:
    """Manage strategy creation, state transitions, and auto-stop rules.

    Usage:
        manager = StrategyLifecycleManager()
        manager.create_strategy(config)
        manager.start_strategy("my_strat")
        manager.record_trade("my_strat", pnl=150.0)
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.check_interval = config.get("check_interval", 60)
        self._strategies: dict[str, StrategyRecord] = {}
        self._callbacks: dict[str, list] = {}

    def create_strategy(self, config: StrategyConfig) -> StrategyRecord:
        """Register a new strategy in idle state."""
        if config.strategy_id in self._strategies:
            raise ValueError(f"Strategy {config.strategy_id} already exists")

        if not config.created_at:
            config.created_at = time.time()

        record = StrategyRecord(config=config, state=StrategyState.IDLE)
        record.state_history.append((time.time(), "", StrategyState.IDLE.value))
        self._strategies[config.strategy_id] = record
        logger.info("Strategy created: %s (%s)", config.strategy_id, config.name)
        self._emit(config.strategy_id, "created", StrategyState.IDLE)
        return record

    def delete_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy (must be stopped or idle)."""
        record = self._strategies.get(strategy_id)
        if record is None:
            logger.warning("Strategy not found: %s", strategy_id)
            return False

        if record.state == StrategyState.RUNNING:
            logger.warning("Cannot delete running strategy %s; stop it first", strategy_id)
            return False

        del self._strategies[strategy_id]
        logger.info("Strategy deleted: %s", strategy_id)
        self._emit(strategy_id, "deleted", StrategyState.STOPPED)
        return True

    def start_strategy(self, strategy_id: str) -> bool:
        """Transition a strategy to running state."""
        record = self._strategies.get(strategy_id)
        if record is None:
            logger.warning("Strategy not found: %s", strategy_id)
            return False

        if record.state == StrategyState.RUNNING:
            logger.debug("Strategy %s already running", strategy_id)
            return True

        record.state = StrategyState.RUNNING
        record.started_at = time.time()
        record.state_history.append((time.time(), "", StrategyState.RUNNING.value))
        logger.info("Strategy started: %s", strategy_id)
        self._emit(strategy_id, "started", StrategyState.RUNNING)
        return True

    def pause_strategy(self, strategy_id: str, reason: str = "") -> bool:
        """Pause a running strategy."""
        record = self._strategies.get(strategy_id)
        if record is None:
            return False

        if record.state != StrategyState.RUNNING:
            logger.warning("Cannot pause strategy %s in state %s", strategy_id, record.state)
            return False

        record.state = StrategyState.PAUSED
        record.pause_reason = reason
        record.state_history.append((time.time(), reason, StrategyState.PAUSED.value))
        logger.info("Strategy paused: %s (reason: %s)", strategy_id, reason)
        self._emit(strategy_id, "paused", StrategyState.PAUSED)
        return True

    def resume_strategy(self, strategy_id: str) -> bool:
        """Resume a paused strategy."""
        record = self._strategies.get(strategy_id)
        if record is None:
            return False

        if record.state != StrategyState.PAUSED:
            logger.warning("Cannot resume strategy %s in state %s", strategy_id, record.state)
            return False

        record.state = StrategyState.RUNNING
        record.pause_reason = ""
        record.state_history.append((time.time(), "", StrategyState.RUNNING.value))
        logger.info("Strategy resumed: %s", strategy_id)
        self._emit(strategy_id, "resumed", StrategyState.RUNNING)
        return True

    def stop_strategy(self, strategy_id: str, reason: str = "") -> bool:
        """Stop a strategy permanently."""
        record = self._strategies.get(strategy_id)
        if record is None:
            return False

        record.state = StrategyState.STOPPED
        record.stopped_at = time.time()
        record.stop_reason = reason
        record.state_history.append((time.time(), reason, StrategyState.STOPPED.value))
        logger.info("Strategy stopped: %s (reason: %s)", strategy_id, reason)
        self._emit(strategy_id, "stopped", StrategyState.STOPPED)
        return True

    # ── Trade Recording & Monitoring ──────────────────────────────

    def record_trade(self, strategy_id: str, pnl: float, equity: float | None = None) -> None:
        """Record a completed trade and check auto-stop rules."""
        record = self._strategies.get(strategy_id)
        if record is None:
            return

        record.pnl += pnl
        record.trade_count += 1

        if pnl > 0:
            record.win_count += 1
            record.loss_streak = 0
        else:
            record.loss_streak += 1
            record.max_loss_streak = max(record.max_loss_streak, record.loss_streak)

        if equity is not None:
            record.current_equity = equity
            if equity > record.peak_equity:
                record.peak_equity = equity

            # Drawdown check
            if record.peak_equity > 0:
                dd = (record.peak_equity - equity) / record.peak_equity * 100
                record.max_drawdown_pct = max(record.max_drawdown_pct, dd)

        # Auto-stop checks
        self._check_auto_stop(strategy_id)

    def _check_auto_stop(self, strategy_id: str) -> None:
        """Check whether auto-stop rules are triggered."""
        record = self._strategies.get(strategy_id)
        if record is None or record.state != StrategyState.RUNNING:
            return

        cfg = record.config

        # Max drawdown
        if record.max_drawdown_pct > cfg.max_drawdown_pct:
            self.stop_strategy(
                strategy_id,
                reason=f"Auto-stop: drawdown {record.max_drawdown_pct:.1f}% exceeds limit {cfg.max_drawdown_pct:.1f}%",
            )
            return

        # Loss streak
        if record.loss_streak >= cfg.max_loss_streak:
            self.stop_strategy(
                strategy_id,
                reason=f"Auto-stop: loss streak {record.loss_streak} >= limit {cfg.max_loss_streak}",
            )
            return

    # ── Query ─────────────────────────────────────────────────────

    def get_strategy(self, strategy_id: str) -> StrategyRecord | None:
        return self._strategies.get(strategy_id)

    def list_strategies(self, state: StrategyState | None = None) -> list[StrategyRecord]:
        """List strategies, optionally filtered by state."""
        records = list(self._strategies.values())
        if state is not None:
            records = [r for r in records if r.state == state]
        return records

    def get_running_strategies(self) -> list[StrategyRecord]:
        return self.list_strategies(StrategyState.RUNNING)

    def modify_strategy(self, strategy_id: str, **kwargs: Any) -> bool:
        """Modify mutable config fields on a strategy."""
        record = self._strategies.get(strategy_id)
        if record is None:
            return False

        cfg = record.config
        for key, val in kwargs.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)
            else:
                cfg.params[key] = val

        logger.info("Strategy modified: %s -> %s", strategy_id, kwargs)
        return True

    # ── Callbacks ─────────────────────────────────────────────────

    def on_state_change(self, strategy_id: str, callback) -> None:
        """Register a callback for state changes on a strategy."""
        if strategy_id not in self._callbacks:
            self._callbacks[strategy_id] = []
        self._callbacks[strategy_id].append(callback)

    def _emit(self, strategy_id: str, event: str, state: StrategyState) -> None:
        for cb in self._callbacks.get(strategy_id, []):
            try:
                cb(strategy_id, event, state)
            except Exception as e:
                logger.error("Callback error for %s: %s", strategy_id, e)

    # ── Reporting ─────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Get a summary of all managed strategies."""
        states = {s.value: 0 for s in StrategyState}
        for r in self._strategies.values():
            states[r.state.value] += 1

        return {
            "total_strategies": len(self._strategies),
            "by_state": states,
            "total_pnl": sum(r.pnl for r in self._strategies.values()),
            "total_trades": sum(r.trade_count for r in self._strategies.values()),
        }

    def get_strategy_report(self, strategy_id: str) -> dict:
        """Get detailed report for a single strategy."""
        record = self._strategies.get(strategy_id)
        if record is None:
            return {}

        return {
            "strategy_id": strategy_id,
            "name": record.config.name,
            "state": record.state.value,
            "pnl": record.pnl,
            "trade_count": record.trade_count,
            "win_rate": record.win_rate,
            "loss_streak": record.loss_streak,
            "max_loss_streak": record.max_loss_streak,
            "max_drawdown_pct": record.max_drawdown_pct,
            "uptime_seconds": record.uptime_seconds,
            "pause_reason": record.pause_reason,
            "stop_reason": record.stop_reason,
            "state_changes": len(record.state_history),
        }
