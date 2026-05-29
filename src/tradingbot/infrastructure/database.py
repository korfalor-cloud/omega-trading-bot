"""Database Layer — persistence for trades, signals, configs.

Implements:
- SQLite-based persistent storage (no external DB needed)
- Trade journal persistence
- Signal history persistence
- Strategy state persistence
- Config persistence
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Database:
    """SQLite-based persistent storage."""

    def __init__(self, db_path: str = "data/omega.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

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
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    strategy_id TEXT,
                    symbol TEXT,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    fees REAL,
                    entry_time TEXT,
                    exit_time TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS signals (
                    id TEXT PRIMARY KEY,
                    strategy_id TEXT,
                    symbol TEXT,
                    side TEXT,
                    strength REAL,
                    confidence REAL,
                    signal_type TEXT,
                    timeframe TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS strategy_state (
                    strategy_id TEXT PRIMARY KEY,
                    genome TEXT,
                    status TEXT,
                    pnl REAL DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    sharpe REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    equity REAL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS evolution_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation INTEGER,
                    best_fitness REAL,
                    avg_fitness REAL,
                    population_size INTEGER,
                    diversity REAL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots(timestamp);
            """)

    # ── Trades ───────────────────────────────────────────────────

    def save_trade(self, trade: dict) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trades
                (id, strategy_id, symbol, side, entry_price, exit_price,
                 quantity, pnl, pnl_pct, fees, entry_time, exit_time, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.get("id", ""), trade.get("strategy_id", ""),
                trade.get("symbol", ""), trade.get("side", ""),
                trade.get("entry_price", 0), trade.get("exit_price", 0),
                trade.get("quantity", 0), trade.get("pnl", 0),
                trade.get("pnl_pct", 0), trade.get("fees", 0),
                str(trade.get("entry_time", "")), str(trade.get("exit_time", "")),
                json.dumps(trade.get("metadata", {})),
            ))

    def get_trades(self, strategy_id: str = "", symbol: str = "", limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            if strategy_id:
                query += " AND strategy_id = ?"
                params.append(strategy_id)
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    # ── Signals ──────────────────────────────────────────────────

    def save_signal(self, signal: dict) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO signals
                (id, strategy_id, symbol, side, strength, confidence,
                 signal_type, timeframe, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.get("id", ""), signal.get("strategy_id", ""),
                signal.get("symbol", ""), signal.get("side", ""),
                signal.get("strength", 0), signal.get("confidence", 0),
                signal.get("signal_type", ""), signal.get("timeframe", ""),
                json.dumps(signal.get("metadata", {})),
            ))

    def get_signals(self, strategy_id: str = "", limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            query = "SELECT * FROM signals WHERE 1=1"
            params = []
            if strategy_id:
                query += " AND strategy_id = ?"
                params.append(strategy_id)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    # ── Strategy State ───────────────────────────────────────────

    def save_strategy_state(self, state: dict) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO strategy_state
                (strategy_id, genome, status, pnl, total_trades, sharpe, max_drawdown, win_rate, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                state.get("strategy_id", ""), json.dumps(state.get("genome", {})),
                state.get("status", "idle"), state.get("pnl", 0),
                state.get("total_trades", 0), state.get("sharpe", 0),
                state.get("max_drawdown", 0), state.get("win_rate", 0),
                datetime.utcnow().isoformat(),
            ))

    def get_strategy_state(self, strategy_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM strategy_state WHERE strategy_id = ?", (strategy_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_all_strategy_states(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM strategy_state").fetchall()
            return [dict(r) for r in rows]

    # ── Equity ───────────────────────────────────────────────────

    def save_equity(self, equity: float) -> None:
        with self._conn() as conn:
            conn.execute("INSERT INTO equity_snapshots (equity) VALUES (?)", (equity,))

    def get_equity_history(self, limit: int = 1000) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM equity_snapshots ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Config ───────────────────────────────────────────────────

    def set_config(self, key: str, value: any) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO config (key, value, updated_at) VALUES (?, ?, ?)
            """, (key, json.dumps(value), datetime.utcnow().isoformat()))

    def get_config(self, key: str, default: any = None) -> any:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
            if row:
                try:
                    return json.loads(row["value"])
                except (json.JSONDecodeError, TypeError):
                    return row["value"]
            return default

    # ── Evolution Log ────────────────────────────────────────────

    def log_evolution(self, generation: int, best_fitness: float, avg_fitness: float, pop_size: int, diversity: float) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO evolution_log (generation, best_fitness, avg_fitness, population_size, diversity)
                VALUES (?, ?, ?, ?, ?)
            """, (generation, best_fitness, avg_fitness, pop_size, diversity))

    def get_evolution_history(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM evolution_log ORDER BY generation DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Stats ────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._conn() as conn:
            trades = conn.execute("SELECT COUNT(*) as n FROM trades").fetchone()["n"]
            signals = conn.execute("SELECT COUNT(*) as n FROM signals").fetchone()["n"]
            strategies = conn.execute("SELECT COUNT(*) as n FROM strategy_state").fetchone()["n"]
            equity_points = conn.execute("SELECT COUNT(*) as n FROM equity_snapshots").fetchone()["n"]
            return {
                "total_trades": trades,
                "total_signals": signals,
                "total_strategies": strategies,
                "equity_points": equity_points,
            }
