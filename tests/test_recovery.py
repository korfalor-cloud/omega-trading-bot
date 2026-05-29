"""Tests for error recovery — event bus resilience, state persistence, error handling."""
from __future__ import annotations

import asyncio
import tempfile
import os
from pathlib import Path

import pytest

from tradingbot.core.events import Event, EventBus
from tradingbot.core.errors import ExchangeError, OrderError
from tradingbot.infrastructure.event_bus import EventBus as SyncEventBus, EventType
from tradingbot.infrastructure.config_manager import ConfigManager
from tradingbot.infrastructure.database import Database
from tradingbot.exchanges.base_connector import ConnectionState


class TestSyncEventBusRecovery:
    """Test that synchronous EventBus continues operating after handler errors."""

    def test_handler_error_does_not_crash_bus(self):
        bus = SyncEventBus()
        received = []

        def bad_handler(event):
            raise ValueError("handler crash")

        def good_handler(event):
            received.append(event)

        bus.subscribe(EventType.SIGNAL, bad_handler)
        bus.subscribe(EventType.SIGNAL, good_handler)
        bus.emit(EventType.SIGNAL, {"ok": True})
        assert len(received) == 1

    def test_global_handler_survives_error(self):
        bus = SyncEventBus()
        received = []

        def bad_global(event):
            raise RuntimeError("global crash")

        def good_global(event):
            received.append(event)

        bus.subscribe_all(bad_global)
        bus.subscribe_all(good_global)
        bus.emit(EventType.TRADE, {"x": 1})
        assert len(received) == 1


class TestAsyncEventBusRecovery:
    """Test that async EventBus continues operating after handler errors."""

    @pytest.mark.asyncio
    async def test_async_handler_error_isolated(self):
        bus = EventBus()
        results = []

        async def bad_handler(payload):
            raise RuntimeError("async crash")

        async def good_handler(payload):
            results.append(payload)

        bus.subscribe(Event.BAR_CLOSED, bad_handler)
        bus.subscribe(Event.BAR_CLOSED, good_handler)

        task = asyncio.create_task(bus.run())
        await bus.publish(Event.BAR_CLOSED, {"price": 100})
        await asyncio.sleep(0.2)
        await bus.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_event_bus_stats_survive_errors(self):
        bus = EventBus()

        async def bad(payload):
            raise ValueError("fail")

        bus.subscribe(Event.TICK_RECEIVED, bad)

        task = asyncio.create_task(bus.run())
        await bus.publish(Event.TICK_RECEIVED, {"x": 1})
        await asyncio.sleep(0.1)
        assert bus.stats.get(Event.TICK_RECEIVED) == 1
        await bus.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class TestConfigPersistence:
    """Test that configuration state persists across crashes."""

    @pytest.fixture
    def config_path(self):
        path = tempfile.mktemp(suffix=".yaml")
        yield path
        for ext in [".yaml", ".json"]:
            p = Path(path).with_suffix(ext)
            if p.exists():
                os.unlink(p)

    def test_config_survives_restart(self, config_path):
        cm1 = ConfigManager(config_path)
        cm1.set("trading", "mode", "live")
        cm1.set("exchange", "api_key", "recovery_key")

        cm2 = ConfigManager(config_path)
        assert cm2.get("trading", "mode") == "live"
        assert cm2.get("exchange", "api_key") == "recovery_key"

    def test_config_change_callback_registered(self, config_path):
        cm = ConfigManager(config_path)
        changes = []
        cm.on_change(lambda cfg: changes.append(cfg))
        assert len(cm._callbacks) == 1


class TestDatabasePersistence:
    """Test that database state persists across sessions."""

    @pytest.fixture
    def db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        yield Database(db_path=path)
        os.unlink(path)

    def test_trade_persists(self, db):
        db.save_trade({
            "id": "recovery_t1", "strategy_id": "s1", "symbol": "BTC/USDT",
            "side": "buy", "pnl": 200,
        })
        trades = db.get_trades(strategy_id="s1")
        assert any(t["id"] == "recovery_t1" for t in trades)

    def test_equity_persists(self, db):
        db.save_equity(100000)
        db.save_equity(105000)
        history = db.get_equity_history()
        assert len(history) >= 2

    def test_strategy_state_persists(self, db):
        db.save_strategy_state({
            "strategy_id": "recover_strat",
            "status": "running",
            "pnl": 1500,
            "sharpe": 1.8,
        })
        state = db.get_strategy_state("recover_strat")
        assert state["pnl"] == 1500


class TestConnectionStateRecovery:
    """Test connection state transitions for recovery scenarios."""

    def test_states_are_distinct(self):
        states = list(ConnectionState)
        assert len(states) == len(set(states))

    def test_disconnected_is_not_connected(self):
        assert ConnectionState.DISCONNECTED != ConnectionState.CONNECTED

    def test_reconnecting_is_unique(self):
        assert ConnectionState.RECONNECTING != ConnectionState.CONNECTED
        assert ConnectionState.RECONNECTING != ConnectionState.DISCONNECTED

    def test_closed_is_final(self):
        closed = ConnectionState.CLOSED
        assert closed is not None
        assert closed != ConnectionState.DISCONNECTED
