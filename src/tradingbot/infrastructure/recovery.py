"""Error Recovery & Crash Handling — state persistence, idempotent operations, graceful shutdown.

Implements:
- State persistence before operations
- Idempotent order operations (deduplication)
- Crash detection and recovery
- State reconstruction from exchange
- Graceful shutdown handling
- Signal handlers (SIGTERM, SIGINT)
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data models
# ---------------------------------------------------------------------------

class OperationStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ShutdownPhase(Enum):
    RUNNING = auto()
    DRAINING = auto()      # stop accepting new work
    FLUSHING = auto()      # persist in-flight state
    STOPPED = auto()


@dataclass
class OperationRecord:
    """Record of a tracked operation (e.g. an order submission)."""
    op_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    idempotency_key: str = ""
    op_type: str = ""  # "place_order", "cancel_order", "modify_order", …
    payload: dict = field(default_factory=dict)
    status: OperationStatus = OperationStatus.PENDING
    result: Optional[dict] = None
    error: str = ""
    attempts: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float = 0.0


@dataclass
class StateSnapshot:
    """A point-in-time snapshot of system state."""
    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)
    component: str = ""  # "portfolio", "orders", "positions", etc.
    state: dict = field(default_factory=dict)
    checksum: str = ""  # SHA-256 of the JSON-serialised state


# ---------------------------------------------------------------------------
# Crash log / state persistence store
# ---------------------------------------------------------------------------

class CrashStore:
    """SQLite-backed persistence for operation records and state snapshots.

    Survives process crashes so that recovery can inspect what was in-flight.
    """

    def __init__(self, db_path: str = "data/recovery.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()
        logger.info("CrashStore initialised at %s", db_path)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS operations (
                    op_id TEXT PRIMARY KEY,
                    idempotency_key TEXT,
                    op_type TEXT,
                    payload TEXT,
                    status TEXT,
                    result TEXT,
                    error TEXT,
                    attempts INTEGER DEFAULT 0,
                    created_at REAL,
                    updated_at REAL,
                    completed_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_ops_idem ON operations(idempotency_key);
                CREATE INDEX IF NOT EXISTS idx_ops_status ON operations(status);

                CREATE TABLE IF NOT EXISTS state_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    component TEXT,
                    state TEXT,
                    checksum TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_snap_component ON state_snapshots(component);
                CREATE INDEX IF NOT EXISTS idx_snap_ts ON state_snapshots(timestamp);

                CREATE TABLE IF NOT EXISTS shutdown_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event TEXT,
                    timestamp REAL,
                    detail TEXT
                );
            """)

    # -- Operations ---------------------------------------------------------

    def save_operation(self, op: OperationRecord) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO operations
                (op_id, idempotency_key, op_type, payload, status, result,
                 error, attempts, created_at, updated_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                op.op_id, op.idempotency_key, op.op_type,
                json.dumps(op.payload), op.status.value,
                json.dumps(op.result) if op.result else None,
                op.error, op.attempts,
                op.created_at, op.updated_at, op.completed_at,
            ))

    def get_operation(self, op_id: str) -> Optional[OperationRecord]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM operations WHERE op_id = ?", (op_id,)).fetchone()
            return self._row_to_op(row) if row else None

    def get_by_idempotency_key(self, key: str) -> Optional[OperationRecord]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM operations WHERE idempotency_key = ?", (key,)
            ).fetchone()
            return self._row_to_op(row) if row else None

    def get_incomplete_operations(self) -> list[OperationRecord]:
        """Return operations that were interrupted (pending or in_progress)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM operations WHERE status IN ('pending', 'in_progress') ORDER BY created_at"
            ).fetchall()
            return [self._row_to_op(r) for r in rows]

    def get_operations_by_type(self, op_type: str, limit: int = 100) -> list[OperationRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM operations WHERE op_type = ? ORDER BY created_at DESC LIMIT ?",
                (op_type, limit),
            ).fetchall()
            return [self._row_to_op(r) for r in rows]

    @staticmethod
    def _row_to_op(row: sqlite3.Row) -> OperationRecord:
        return OperationRecord(
            op_id=row["op_id"],
            idempotency_key=row["idempotency_key"] or "",
            op_type=row["op_type"],
            payload=json.loads(row["payload"]) if row["payload"] else {},
            status=OperationStatus(row["status"]),
            result=json.loads(row["result"]) if row["result"] else None,
            error=row["error"] or "",
            attempts=row["attempts"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
        )

    # -- State snapshots ----------------------------------------------------

    def save_snapshot(self, snapshot: StateSnapshot) -> None:
        import hashlib
        state_json = json.dumps(snapshot.state, sort_keys=True)
        snapshot.checksum = hashlib.sha256(state_json.encode()).hexdigest()
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO state_snapshots
                (snapshot_id, timestamp, component, state, checksum)
                VALUES (?, ?, ?, ?, ?)
            """, (
                snapshot.snapshot_id, snapshot.timestamp,
                snapshot.component, state_json, snapshot.checksum,
            ))

    def get_latest_snapshot(self, component: str) -> Optional[StateSnapshot]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM state_snapshots WHERE component = ? ORDER BY timestamp DESC LIMIT 1",
                (component,),
            ).fetchone()
            if row:
                return StateSnapshot(
                    snapshot_id=row["snapshot_id"],
                    timestamp=row["timestamp"],
                    component=row["component"],
                    state=json.loads(row["state"]),
                    checksum=row["checksum"],
                )
        return None

    def get_snapshot_history(self, component: str, limit: int = 10) -> list[StateSnapshot]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM state_snapshots WHERE component = ? ORDER BY timestamp DESC LIMIT ?",
                (component, limit),
            ).fetchall()
            return [
                StateSnapshot(
                    snapshot_id=r["snapshot_id"],
                    timestamp=r["timestamp"],
                    component=r["component"],
                    state=json.loads(r["state"]),
                    checksum=r["checksum"],
                )
                for r in rows
            ]

    # -- Shutdown log -------------------------------------------------------

    def log_shutdown_event(self, event: str, detail: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO shutdown_log (event, timestamp, detail) VALUES (?, ?, ?)",
                (event, time.time(), detail),
            )

    def get_last_shutdown(self) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM shutdown_log ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None


# ---------------------------------------------------------------------------
# Idempotent operation executor
# ---------------------------------------------------------------------------

class IdempotentExecutor:
    """Execute operations with idempotency guarantees.

    Each operation is identified by an idempotency key.  If an operation
    with the same key has already completed, the cached result is returned.
    """

    def __init__(self, store: CrashStore):
        self._store = store

    def execute(
        self,
        op_type: str,
        idempotency_key: str,
        payload: dict,
        handler: Callable[[dict], dict],
        max_retries: int = 3,
    ) -> OperationRecord:
        """Execute *handler(payload)* with idempotency.

        Returns the OperationRecord with the result.
        """
        # Check for existing completed operation
        existing = self._store.get_by_idempotency_key(idempotency_key)
        if existing and existing.status == OperationStatus.COMPLETED:
            logger.info("Idempotent hit: %s (key=%s)", op_type, idempotency_key[:16])
            return existing

        # Create or reuse the record
        op = existing or OperationRecord(
            idempotency_key=idempotency_key,
            op_type=op_type,
            payload=payload,
        )
        op.status = OperationStatus.IN_PROGRESS
        op.attempts += 1
        op.updated_at = time.time()
        self._store.save_operation(op)

        # Execute with retries
        last_error: str = ""
        for attempt in range(max_retries):
            try:
                result = handler(payload)
                op.status = OperationStatus.COMPLETED
                op.result = result
                op.completed_at = time.time()
                op.updated_at = time.time()
                self._store.save_operation(op)
                logger.info("Operation completed: %s (%s)", op_type, op.op_id[:12])
                return op
            except Exception as exc:
                last_error = str(exc)
                op.error = last_error
                op.attempts = attempt + 1
                op.updated_at = time.time()
                self._store.save_operation(op)
                logger.warning(
                    "Operation %s attempt %d/%d failed: %s",
                    op_type, attempt + 1, max_retries, last_error,
                )
                if attempt < max_retries - 1:
                    time.sleep(min(2 ** attempt, 10))

        op.status = OperationStatus.FAILED
        op.updated_at = time.time()
        self._store.save_operation(op)
        logger.error("Operation %s failed after %d attempts: %s", op_type, max_retries, last_error)
        return op


# ---------------------------------------------------------------------------
# State manager (snapshot + reconstruction)
# ---------------------------------------------------------------------------

class StateManager:
    """Manage system state snapshots for crash recovery.

    Call ``snapshot()`` before risky operations and ``reconstruct()``
    after a crash to restore the last known good state.
    """

    def __init__(self, store: CrashStore):
        self._store = store
        self._live_state: dict[str, dict] = {}  # component -> state

    def snapshot(self, component: str, state: dict) -> StateSnapshot:
        """Persist a snapshot of *component*'s current state."""
        snap = StateSnapshot(component=component, state=state)
        self._store.save_snapshot(snap)
        self._live_state[component] = state
        logger.debug("State snapshot saved: %s (%s)", component, snap.snapshot_id[:12])
        return snap

    def reconstruct(self, component: str) -> Optional[dict]:
        """Load the latest snapshot for *component*."""
        snap = self._store.get_latest_snapshot(component)
        if snap:
            self._live_state[component] = snap.state
            logger.info(
                "State reconstructed: %s (snapshot=%s, age=%.0fs)",
                component,
                snap.snapshot_id[:12],
                time.time() - snap.timestamp,
            )
            return snap.state
        logger.warning("No snapshot found for component: %s", component)
        return None

    def reconstruct_all(self) -> dict[str, dict]:
        """Reconstruct all components that have snapshots.

        Returns a dict mapping component names to their restored state.
        """
        restored: dict[str, dict] = {}
        with self._store._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT component FROM state_snapshots"
            ).fetchall()
            for row in rows:
                component = row["component"]
                state = self.reconstruct(component)
                if state is not None:
                    restored[component] = state
        logger.info("Reconstructed %d components", len(restored))
        return restored

    def get_live_state(self, component: str) -> Optional[dict]:
        return self._live_state.get(component)

    def update_live(self, component: str, state: dict) -> None:
        """Update in-memory live state without persisting."""
        self._live_state[component] = state


# ---------------------------------------------------------------------------
# Crash detector
# ---------------------------------------------------------------------------

class CrashDetector:
    """Detect unclean shutdowns and trigger recovery."""

    def __init__(self, store: CrashStore):
        self._store = store
        self._running = False

    def mark_startup(self) -> bool:
        """Call at application start. Returns True if the previous run crashed."""
        last = self._store.get_last_shutdown()
        crashed = False
        if last and last["event"] != "clean_shutdown":
            logger.warning(
                "Unclean shutdown detected (event=%s, %.0f seconds ago)",
                last["event"],
                time.time() - last["timestamp"],
            )
            crashed = True
        self._store.log_shutdown_event("startup")
        self._running = True
        return crashed

    def mark_clean_shutdown(self) -> None:
        self._store.log_shutdown_event("clean_shutdown")
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running


# ---------------------------------------------------------------------------
# Graceful shutdown manager
# ---------------------------------------------------------------------------

ShutdownCallback = Callable[[], None]


class ShutdownManager:
    """Coordinate graceful shutdown across all components.

    Registers SIGTERM/SIGINT handlers, drains in-flight work, flushes
    state, and invokes registered callbacks in LIFO order.
    """

    def __init__(self, crash_store: Optional[CrashStore] = None):
        self._phase = ShutdownPhase.RUNNING
        self._callbacks: list[ShutdownCallback] = []
        self._lock = threading.Lock()
        self._crash_store = crash_store
        self._original_handlers: dict[int, Any] = {}
        self._drain_timeout: float = 30.0  # seconds to wait for in-flight work

    @property
    def phase(self) -> ShutdownPhase:
        return self._phase

    @property
    def is_shutting_down(self) -> bool:
        return self._phase != ShutdownPhase.RUNNING

    # -- Registration -------------------------------------------------------

    def register(self, callback: ShutdownCallback) -> None:
        """Register a callback to be invoked during shutdown (LIFO order)."""
        self._callbacks.append(callback)
        logger.debug("Shutdown callback registered: %s", callback.__qualname__)

    def install_signal_handlers(self) -> None:
        """Install SIGTERM and SIGINT signal handlers."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)
        # Also register atexit as a safety net
        atexit.register(self._atexit_handler)
        logger.info("Shutdown signal handlers installed (SIGTERM, SIGINT)")

    def uninstall_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        self._original_handlers.clear()

    # -- Shutdown execution -------------------------------------------------

    def shutdown(self, reason: str = "manual", timeout: Optional[float] = None) -> None:
        """Initiate graceful shutdown.

        This is safe to call from a signal handler or from application code.
        """
        with self._lock:
            if self._phase != ShutdownPhase.RUNNING:
                logger.warning("Shutdown already in progress (phase=%s)", self._phase.name)
                return

            logger.info("Graceful shutdown initiated: %s", reason)
            self._phase = ShutdownPhase.DRAINING

            if self._crash_store:
                self._crash_store.log_shutdown_event("shutdown_start", reason)

        # Phase 1: Drain — stop accepting new work
        drain_deadline = time.time() + (timeout or self._drain_timeout)

        # Phase 2: Flush — run all registered callbacks (LIFO)
        self._phase = ShutdownPhase.FLUSHING
        for cb in reversed(self._callbacks):
            try:
                logger.info("Running shutdown callback: %s", cb.__qualname__)
                cb()
            except Exception as exc:
                logger.error("Shutdown callback failed (%s): %s", cb.__qualname__, exc)

        # Phase 3: Stopped
        self._phase = ShutdownPhase.STOPPED

        if self._crash_store:
            self._crash_store.log_shutdown_event("clean_shutdown", reason)

        logger.info("Graceful shutdown complete")

    # -- Signal handlers (private) ------------------------------------------

    def _handle_signal(self, signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received signal %s", sig_name)
        self.shutdown(reason=f"signal:{sig_name}")

    def _atexit_handler(self) -> None:
        if self._phase == ShutdownPhase.RUNNING:
            self.shutdown(reason="atexit")


# ---------------------------------------------------------------------------
# Order state reconstructor
# ---------------------------------------------------------------------------

class OrderReconstructor:
    """Reconstruct order state from an exchange after a crash.

    The caller provides an ``exchange_adapter`` that implements the
    ``fetch_open_orders()`` and ``fetch_order(order_id)`` methods.
    """

    def __init__(self, state_manager: StateManager, crash_store: CrashStore):
        self._state_mgr = state_manager
        self._crash_store = crash_store

    def reconcile(
        self,
        exchange_adapter: Any,
        symbol: Optional[str] = None,
    ) -> dict[str, Any]:
        """Compare persisted state with exchange and return a reconciliation report.

        The exchange adapter must expose:
          - ``fetch_open_orders(symbol=None) -> list[dict]``
          - ``fetch_order(order_id) -> dict``
        """
        report: dict[str, Any] = {
            "orphaned_on_exchange": [],  # orders on exchange but not in our state
            "stale_in_state": [],        # orders in our state but not on exchange
            "status_mismatches": [],     # orders whose status differs
            "recovered": [],
            "timestamp": time.time(),
        }

        # Get our last known state
        our_orders: dict[str, dict] = {}
        snap = self._crash_store.get_latest_snapshot("orders")
        if snap:
            our_orders = {o.get("order_id", ""): o for o in snap.state.get("orders", [])}

        # Get exchange state
        try:
            exchange_orders_raw = exchange_adapter.fetch_open_orders(symbol=symbol)
            exchange_orders = {o.get("order_id", o.get("id", "")): o for o in exchange_orders_raw}
        except Exception as exc:
            logger.error("Failed to fetch exchange orders: %s", exc)
            return report

        # Find orphaned orders (on exchange, not in our records)
        for oid, ex_order in exchange_orders.items():
            if oid not in our_orders:
                report["orphaned_on_exchange"].append(ex_order)
                logger.warning("Orphaned order on exchange: %s", oid)

        # Find stale orders (in our records, not on exchange)
        for oid, our_order in our_orders.items():
            if oid not in exchange_orders and our_order.get("status") in ("open", "partial"):
                report["stale_in_state"].append(our_order)
                logger.warning("Stale order in state: %s", oid)

        # Status mismatches
        for oid in set(our_orders) & set(exchange_orders):
            our_status = our_orders[oid].get("status", "")
            ex_status = exchange_orders[oid].get("status", "")
            if our_status != ex_status:
                report["status_mismatches"].append({
                    "order_id": oid,
                    "our_status": our_status,
                    "exchange_status": ex_status,
                })

        report["recovered"] = list(exchange_orders.values())
        logger.info(
            "Order reconciliation complete: orphaned=%d, stale=%d, mismatches=%d",
            len(report["orphaned_on_exchange"]),
            len(report["stale_in_state"]),
            len(report["status_mismatches"]),
        )

        # Update our snapshot with the canonical exchange state
        self._state_mgr.snapshot("orders", {"orders": list(exchange_orders.values())})
        return report


# ---------------------------------------------------------------------------
# Recovery orchestrator
# ---------------------------------------------------------------------------

class RecoveryManager:
    """Top-level orchestrator that ties crash detection, state reconstruction,
    and idempotent execution together.

    Usage::

        recovery = RecoveryManager(db_path="data/recovery.db")
        recovery.startup()  # returns True if a crash was detected

        # During normal operation:
        recovery.snapshot_state("portfolio", portfolio_state)
        result = recovery.execute_idempotent("place_order", key, payload, handler)

        # On shutdown:
        recovery.shutdown()
    """

    def __init__(self, db_path: str = "data/recovery.db"):
        self._store = CrashStore(db_path)
        self._crash_detector = CrashDetector(self._store)
        self._state_manager = StateManager(self._store)
        self._executor = IdempotentExecutor(self._store)
        self._shutdown_manager = ShutdownManager(crash_store=self._store)
        self._order_reconstructor = OrderReconstructor(self._state_manager, self._store)

    # -- Properties ---------------------------------------------------------

    @property
    def store(self) -> CrashStore:
        return self._store

    @property
    def state_manager(self) -> StateManager:
        return self._state_manager

    @property
    def shutdown_manager(self) -> ShutdownManager:
        return self._shutdown_manager

    @property
    def order_reconstructor(self) -> OrderReconstructor:
        return self._order_reconstructor

    # -- Lifecycle ----------------------------------------------------------

    def startup(self) -> bool:
        """Initialise recovery, install signal handlers, check for crash.

        Returns True if an unclean shutdown was detected (caller should
        trigger state reconstruction).
        """
        self._shutdown_manager.install_signal_handlers()
        crashed = self._crash_detector.mark_startup()

        if crashed:
            incomplete = self._store.get_incomplete_operations()
            if incomplete:
                logger.warning(
                    "Found %d incomplete operations after crash",
                    len(incomplete),
                )
                for op in incomplete:
                    logger.info(
                        "  - %s [%s] status=%s attempts=%d",
                        op.op_type, op.op_id[:12], op.status.value, op.attempts,
                    )
        return crashed

    def shutdown(self) -> None:
        """Initiate graceful shutdown."""
        self._shutdown_manager.shutdown(reason="application_exit")

    # -- State snapshots ----------------------------------------------------

    def snapshot_state(self, component: str, state: dict) -> StateSnapshot:
        """Persist a state snapshot (call before risky operations)."""
        return self._state_manager.snapshot(component, state)

    def reconstruct_state(self, component: str) -> Optional[dict]:
        """Reconstruct state from the latest snapshot."""
        return self._state_manager.reconstruct(component)

    def reconstruct_all_states(self) -> dict[str, dict]:
        """Reconstruct all components."""
        return self._state_manager.reconstruct_all()

    # -- Idempotent execution -----------------------------------------------

    def execute_idempotent(
        self,
        op_type: str,
        idempotency_key: str,
        payload: dict,
        handler: Callable[[dict], dict],
        max_retries: int = 3,
    ) -> OperationRecord:
        """Execute an operation with idempotency and crash-safe retries."""
        return self._executor.execute(op_type, idempotency_key, payload, handler, max_retries)

    # -- Order reconciliation -----------------------------------------------

    def reconcile_orders(self, exchange_adapter: Any, symbol: Optional[str] = None) -> dict:
        """Reconcile order state with the exchange after a crash."""
        return self._order_reconstructor.reconcile(exchange_adapter, symbol)

    # -- Shutdown callback registration -------------------------------------

    def on_shutdown(self, callback: ShutdownCallback) -> None:
        """Register a callback for graceful shutdown."""
        self._shutdown_manager.register(callback)
