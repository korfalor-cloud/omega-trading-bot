"""Graceful Shutdown — signal handling, order tracking, state persistence.

Implements:
- Signal handling (SIGTERM, SIGINT)
- Active order tracking
- Position reconciliation
- State persistence
- Timeout enforcement
- Cleanup callbacks
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shutdown phases
# ---------------------------------------------------------------------------

@dataclass
class ShutdownPhase:
    """Tracks progress through a shutdown phase."""
    name: str
    started_at: float = 0.0
    completed_at: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class ShutdownState:
    """Snapshot of system state at shutdown time."""
    open_orders: list[dict[str, Any]] = field(default_factory=list)
    open_positions: list[dict[str, Any]] = field(default_factory=list)
    equity: float = 0.0
    pending_signals: list[dict[str, Any]] = field(default_factory=list)
    strategy_states: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "open_orders": self.open_orders,
            "open_positions": self.open_positions,
            "equity": self.equity,
            "pending_signals": self.pending_signals,
            "strategy_states": self.strategy_states,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Order / position providers (callables the system registers)
# ---------------------------------------------------------------------------

@dataclass
class OrderTracker:
    """Provides access to currently active orders."""
    get_open_orders: Callable[[], list[dict[str, Any]]]
    cancel_order: Callable[[str], bool] | None = None
    get_order_status: Callable[[str], str] | None = None


@dataclass
class PositionReconciler:
    """Provides access to current positions and can reconcile them."""
    get_positions: Callable[[], list[dict[str, Any]]]
    close_position: Callable[[str], bool] | None = None
    get_unrealized_pnl: Callable[[], float] | None = None


# ---------------------------------------------------------------------------
# Cleanup callback
# ---------------------------------------------------------------------------

@dataclass
class CleanupCallback:
    """A registered cleanup action to run during shutdown."""
    name: str
    fn: Callable[[], None]
    timeout: float = 10.0
    phase: str = "cleanup"  # "orders", "positions", "state", "cleanup"


# ---------------------------------------------------------------------------
# Graceful shutdown manager
# ---------------------------------------------------------------------------

class GracefulShutdown:
    """Coordinates a clean, phased shutdown of the trading system."""

    def __init__(self, timeout: float = 60.0, state_dir: str = "data/shutdown"):
        self.timeout = timeout
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self._order_tracker: OrderTracker | None = None
        self._position_reconciler: PositionReconciler | None = None
        self._callbacks: list[CleanupCallback] = []
        self._phases: list[ShutdownPhase] = []
        self._shutting_down = False
        self._shutdown_complete = threading.Event()
        self._lock = threading.Lock()

        # Install signal handlers
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Register atexit as a last-resort safety net
        atexit.register(self._atexit_handler)

    # ---- registration ----

    def register_order_tracker(self, tracker: OrderTracker) -> None:
        self._order_tracker = tracker

    def register_position_reconciler(self, reconciler: PositionReconciler) -> None:
        self._position_reconciler = reconciler

    def register_cleanup(self, name: str, fn: Callable[[], None],
                         timeout: float = 10.0, phase: str = "cleanup") -> None:
        """Register a cleanup callback.  *phase* controls execution order:
        "orders" -> "positions" -> "state" -> "cleanup"
        """
        self._callbacks.append(CleanupCallback(name=name, fn=fn, timeout=timeout, phase=phase))
        self._callbacks.sort(key=lambda c: {"orders": 0, "positions": 1, "state": 2, "cleanup": 3}.get(c.phase, 9))

    # ---- queries ----

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    @property
    def shutdown_complete(self) -> bool:
        return self._shutdown_complete.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until shutdown completes.  Returns True if shutdown finished."""
        return self._shutdown_complete.wait(timeout=timeout)

    # ---- signal handling ----

    def _signal_handler(self, signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown")
        # Run shutdown in a separate thread so the signal handler returns promptly
        thread = threading.Thread(target=self.shutdown, daemon=True)
        thread.start()

    def _atexit_handler(self) -> None:
        if not self._shutting_down:
            logger.warning("atexit triggered without prior shutdown — running emergency shutdown")
            self.shutdown()

    # ---- main shutdown sequence ----

    def shutdown(self) -> bool:
        """Execute the full graceful shutdown sequence.  Returns True on success."""
        with self._lock:
            if self._shutting_down:
                return False
            self._shutting_down = True

        logger.info("=== Graceful shutdown started ===")
        deadline = time.time() + self.timeout
        success = True

        # Phase 1: cancel open orders
        if not self._phase_cancel_orders(deadline):
            success = False

        # Phase 2: reconcile positions
        if not self._phase_reconcile_positions(deadline):
            success = False

        # Phase 3: persist state
        if not self._phase_persist_state(deadline):
            success = False

        # Phase 4: run cleanup callbacks
        if not self._phase_run_cleanups("cleanup", deadline):
            success = False

        self._shutdown_complete.set()

        if success:
            logger.info("=== Graceful shutdown completed successfully ===")
        else:
            logger.warning("=== Graceful shutdown completed with errors ===")

        # Restore signal handlers
        signal.signal(signal.SIGTERM, self._original_sigterm)
        signal.signal(signal.SIGINT, self._original_sigint)

        return success

    # ---- phases ----

    def _start_phase(self, name: str) -> ShutdownPhase:
        phase = ShutdownPhase(name=name, started_at=time.time())
        with self._lock:
            self._phases.append(phase)
        logger.info(f"Shutdown phase: {name}")
        return phase

    def _finish_phase(self, phase: ShutdownPhase, success: bool = True, error: str = "") -> None:
        phase.completed_at = time.time()
        phase.success = success
        phase.error = error
        elapsed = phase.completed_at - phase.started_at
        status = "OK" if success else "FAILED"
        logger.info(f"Shutdown phase '{phase.name}': {status} ({elapsed:.1f}s)")

    def _check_deadline(self, deadline: float, phase_name: str) -> bool:
        if time.time() >= deadline:
            logger.error(f"Shutdown timeout during phase '{phase_name}'")
            return False
        return True

    def _phase_cancel_orders(self, deadline: float) -> bool:
        phase = self._start_phase("cancel_orders")
        if self._order_tracker is None:
            self._finish_phase(phase, success=True)
            return True

        try:
            orders = self._order_tracker.get_open_orders()
            if not orders:
                logger.info("No open orders to cancel")
                self._finish_phase(phase, success=True)
                return True

            logger.info(f"Cancelling {len(orders)} open order(s)")
            cancelled = 0
            for order in orders:
                if not self._check_deadline(deadline, phase.name):
                    self._finish_phase(phase, success=False, error="timeout")
                    return False
                order_id = order.get("id", "")
                if self._order_tracker.cancel_order:
                    try:
                        ok = self._order_tracker.cancel_order(order_id)
                        if ok:
                            cancelled += 1
                    except Exception as exc:
                        logger.error(f"Failed to cancel order {order_id}: {exc}")
            logger.info(f"Cancelled {cancelled}/{len(orders)} orders")
            self._finish_phase(phase, success=True)
            return True
        except Exception as exc:
            self._finish_phase(phase, success=False, error=str(exc))
            return False

    def _phase_reconcile_positions(self, deadline: float) -> bool:
        phase = self._start_phase("reconcile_positions")
        if self._position_reconciler is None:
            self._finish_phase(phase, success=True)
            return True

        try:
            positions = self._position_reconciler.get_positions()
            if not positions:
                logger.info("No open positions")
                self._finish_phase(phase, success=True)
                return True

            logger.info(f"Reconciling {len(positions)} open position(s)")
            closed = 0
            for pos in positions:
                if not self._check_deadline(deadline, phase.name):
                    self._finish_phase(phase, success=False, error="timeout")
                    return False
                pos_id = pos.get("id", pos.get("symbol", ""))
                if self._position_reconciler.close_position:
                    try:
                        ok = self._position_reconciler.close_position(pos_id)
                        if ok:
                            closed += 1
                    except Exception as exc:
                        logger.error(f"Failed to close position {pos_id}: {exc}")
            logger.info(f"Closed {closed}/{len(positions)} positions")
            self._finish_phase(phase, success=True)
            return True
        except Exception as exc:
            self._finish_phase(phase, success=False, error=str(exc))
            return False

    def _phase_persist_state(self, deadline: float) -> bool:
        phase = self._start_phase("persist_state")
        try:
            state = self._collect_state()
            path = self.state_dir / "shutdown_state.json"
            path.write_text(json.dumps(state.to_dict(), indent=2, default=str))
            logger.info(f"State persisted to {path}")
            self._finish_phase(phase, success=True)
            return True
        except Exception as exc:
            self._finish_phase(phase, success=False, error=str(exc))
            return False

    def _phase_run_cleanups(self, phase_name: str, deadline: float) -> bool:
        phase = self._start_phase(phase_name)
        for cb in self._callbacks:
            if cb.phase != phase_name:
                continue
            if not self._check_deadline(deadline, phase.name):
                self._finish_phase(phase, success=False, error="timeout")
                return False
            logger.info(f"Running cleanup: {cb.name}")
            try:
                # Run with a per-callback timeout using a thread
                result = self._run_with_timeout(cb.fn, cb.timeout)
                if not result:
                    logger.warning(f"Cleanup '{cb.name}' timed out after {cb.timeout}s")
            except Exception as exc:
                logger.error(f"Cleanup '{cb.name}' failed: {exc}")

        self._finish_phase(phase, success=True)
        return True

    # ---- helpers ----

    def _collect_state(self) -> ShutdownState:
        state = ShutdownState()
        if self._order_tracker:
            try:
                state.open_orders = self._order_tracker.get_open_orders()
            except Exception:
                pass
        if self._position_reconciler:
            try:
                state.open_positions = self._position_reconciler.get_positions()
            except Exception:
                pass
            if self._position_reconciler.get_unrealized_pnl:
                try:
                    state.equity = self._position_reconciler.get_unrealized_pnl()
                except Exception:
                    pass
        return state

    @staticmethod
    def _run_with_timeout(fn: Callable[[], None], timeout: float) -> bool:
        """Run *fn* in a thread with a timeout.  Returns True if it completed."""
        done = threading.Event()
        def wrapper():
            fn()
            done.set()
        t = threading.Thread(target=wrapper, daemon=True)
        t.start()
        t.join(timeout=timeout)
        return done.is_set()

    # ---- persistence helpers ----

    def load_last_shutdown_state(self) -> ShutdownState | None:
        """Load the most recent persisted shutdown state, if available."""
        path = self.state_dir / "shutdown_state.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            state = ShutdownState(
                open_orders=data.get("open_orders", []),
                open_positions=data.get("open_positions", []),
                equity=data.get("equity", 0.0),
                pending_signals=data.get("pending_signals", []),
                strategy_states=data.get("strategy_states", {}),
                metadata=data.get("metadata", {}),
                timestamp=data.get("timestamp", ""),
            )
            return state
        except Exception as exc:
            logger.error(f"Failed to load shutdown state: {exc}")
            return None

    def get_phases(self) -> list[dict[str, Any]]:
        return [
            {
                "name": p.name,
                "started_at": p.started_at,
                "completed_at": p.completed_at,
                "duration_ms": round((p.completed_at - p.started_at) * 1000, 1) if p.completed_at else 0,
                "success": p.success,
                "error": p.error,
            }
            for p in self._phases
        ]
