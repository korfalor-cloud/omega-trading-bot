"""Tests for infrastructure components."""
from __future__ import annotations

import pytest
import json
import tempfile
import os
from pathlib import Path

from tradingbot.infrastructure.database import Database
from tradingbot.infrastructure.config_manager import ConfigManager, DEFAULT_CONFIG
from tradingbot.infrastructure.event_bus import EventBus, Event, EventType
from tradingbot.infrastructure.rate_limiter import RateLimiter, TokenBucket
from tradingbot.infrastructure.report_generator import ReportGenerator, ReportData


class TestDatabase:
    @pytest.fixture
    def db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        yield Database(db_path=path)
        os.unlink(path)

    def test_save_trade(self, db):
        db.save_trade({
            "id": "t1", "strategy_id": "s1", "symbol": "BTC/USDT",
            "side": "buy", "entry_price": 50000, "exit_price": 51000,
            "quantity": 0.1, "pnl": 100, "pnl_pct": 0.02, "fees": 5,
        })
        trades = db.get_trades(strategy_id="s1")
        assert len(trades) == 1
        assert trades[0]["pnl"] == 100

    def test_save_signal(self, db):
        db.save_signal({
            "id": "sig1", "strategy_id": "s1", "symbol": "BTC/USDT",
            "side": "buy", "strength": 0.8, "confidence": 0.7,
            "signal_type": "entry", "timeframe": "1h",
        })
        signals = db.get_signals(strategy_id="s1")
        assert len(signals) == 1

    def test_strategy_state(self, db):
        db.save_strategy_state({
            "strategy_id": "s1", "genome": {"rsi": 14}, "status": "running",
            "pnl": 500, "total_trades": 10, "sharpe": 1.5,
        })
        state = db.get_strategy_state("s1")
        assert state["pnl"] == 500
        assert state["sharpe"] == 1.5

    def test_equity_snapshots(self, db):
        db.save_equity(100000)
        db.save_equity(101000)
        history = db.get_equity_history()
        assert len(history) == 2

    def test_config(self, db):
        db.set_config("trading.mode", "live")
        assert db.get_config("trading.mode") == "live"

    def test_evolution_log(self, db):
        db.log_evolution(1, 0.8, 0.5, 50, 0.3)
        db.log_evolution(2, 0.9, 0.6, 50, 0.25)
        history = db.get_evolution_history()
        assert len(history) == 2
        assert history[0]["generation"] == 2

    def test_stats(self, db):
        db.save_trade({"id": "t1", "strategy_id": "s1", "symbol": "BTC/USDT", "side": "buy", "pnl": 100})
        db.save_equity(100000)
        stats = db.get_stats()
        assert stats["total_trades"] == 1
        assert stats["equity_points"] == 1

    def test_filter_trades(self, db):
        db.save_trade({"id": "t1", "strategy_id": "s1", "symbol": "BTC/USDT", "side": "buy", "pnl": 100})
        db.save_trade({"id": "t2", "strategy_id": "s2", "symbol": "ETH/USDT", "side": "sell", "pnl": -50})
        btc_trades = db.get_trades(symbol="BTC/USDT")
        assert len(btc_trades) == 1


class TestConfigManager:
    @pytest.fixture
    def config_path(self):
        path = tempfile.mktemp(suffix=".yaml")
        yield path
        # Cleanup
        for ext in [".yaml", ".json"]:
            p = Path(path).with_suffix(ext)
            if p.exists():
                os.unlink(p)

    def test_default_config(self, config_path):
        cm = ConfigManager(config_path)
        assert cm.get("system", "name") == "Omega Trading Bot"

    def test_get_set(self, config_path):
        cm = ConfigManager(config_path)
        cm.set("trading", "mode", "live")
        assert cm.get("trading", "mode") == "live"

    def test_get_section(self, config_path):
        cm = ConfigManager(config_path)
        trading = cm.get("trading")
        assert "mode" in trading

    def test_get_all(self, config_path):
        cm = ConfigManager(config_path)
        all_config = cm.get_all()
        assert "system" in all_config
        assert "trading" in all_config

    def test_generate_default(self, config_path):
        cm = ConfigManager(config_path)
        cm.generate_default()
        assert cm.get("system", "name") == "Omega Trading Bot"


class TestEventBus:
    @pytest.fixture
    def bus(self):
        return EventBus()

    def test_subscribe_publish(self, bus):
        received = []
        bus.subscribe(EventType.SIGNAL, lambda e: received.append(e))
        bus.emit(EventType.SIGNAL, {"test": True})
        assert len(received) == 1
        assert received[0].data["test"] is True

    def test_subscribe_all(self, bus):
        received = []
        bus.subscribe_all(lambda e: received.append(e))
        bus.emit(EventType.SIGNAL)
        bus.emit(EventType.TRADE)
        assert len(received) == 2

    def test_unsubscribe(self, bus):
        received = []
        handler = lambda e: received.append(e)
        bus.subscribe(EventType.SIGNAL, handler)
        bus.unsubscribe(EventType.SIGNAL, handler)
        bus.emit(EventType.SIGNAL)
        assert len(received) == 0

    def test_history(self, bus):
        bus.emit(EventType.SIGNAL, {"a": 1})
        bus.emit(EventType.TRADE, {"b": 2})
        history = bus.get_history()
        assert len(history) == 2

    def test_history_filter(self, bus):
        bus.emit(EventType.SIGNAL)
        bus.emit(EventType.TRADE)
        signals = bus.get_history(EventType.SIGNAL)
        assert len(signals) == 1

    def test_handler_count(self, bus):
        bus.subscribe(EventType.SIGNAL, lambda e: None)
        bus.subscribe(EventType.SIGNAL, lambda e: None)
        assert bus.handler_count(EventType.SIGNAL) == 2


class TestRateLimiter:
    @pytest.fixture
    def limiter(self):
        return RateLimiter()

    def test_acquire(self, limiter):
        assert limiter.acquire("test") is True

    def test_is_available(self, limiter):
        assert limiter.is_available("test") is True

    def test_backoff(self, limiter):
        limiter.backoff("test", 10)
        assert limiter.is_available("test") is False

    def test_set_limit(self, limiter):
        limiter.set_limit("api", requests=10, window=60)
        for _ in range(10):
            assert limiter.acquire("api") is True


class TestReportGenerator:
    @pytest.fixture
    def generator(self):
        return ReportGenerator()

    def test_generate_html(self, generator):
        import numpy as np
        data = ReportData(
            equity_curve=np.array([100000, 101000, 99000, 102000, 105000]),
            returns=np.array([0.01, -0.02, 0.03, 0.03]),
            trades=[{"pnl": 100}, {"pnl": -200}, {"pnl": 300}],
            strategy_name="Test Strategy",
        )
        html = generator.generate_html(data)
        assert "Omega Trading Bot" in html
        assert "Test Strategy" in html
        assert "Sharpe" in html

    def test_compute_metrics(self, generator):
        import numpy as np
        data = ReportData(
            equity_curve=np.array([100000, 105000, 95000, 110000]),
            returns=np.array([0.05, -0.095, 0.158]),
            trades=[{"pnl": 5000}, {"pnl": -10000}, {"pnl": 15000}],
        )
        metrics = generator._compute_metrics(data)
        assert metrics["total_trades"] == 3
        assert metrics["total_return"] > 0

    def test_save_report(self, generator):
        import numpy as np
        data = ReportData(
            equity_curve=np.array([100000, 101000, 102000]),
            returns=np.array([0.01, 0.01]),
            strategy_name="Test",
        )
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            generator.save_report(data, path)
            assert Path(path).exists()
            content = Path(path).read_text()
            assert "Omega" in content
        finally:
            os.unlink(path)
