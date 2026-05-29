"""Tests for AsyncTradingEngine — startup, shutdown, strategy execution, metrics."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tradingbot.config import OmegaConfig
from tradingbot.core.enums import Side, Timeframe
from tradingbot.core.events import Event, EventBus
from tradingbot.core.types import Fill, OHLCVBar, PortfolioState, Position, Signal
from tradingbot.engine.async_engine import AsyncTradingEngine, EngineMetrics


class TestEngineMetrics:
    """Test the EngineMetrics dataclass."""

    def test_default_values(self):
        m = EngineMetrics()
        assert m.bars_processed == 0
        assert m.ticks_processed == 0
        assert m.signals_generated == 0
        assert m.orders_submitted == 0
        assert m.orders_filled == 0
        assert m.orders_rejected == 0

    def test_snapshot_structure(self):
        m = EngineMetrics()
        snap = m.snapshot()
        assert "bars_processed" in snap
        assert "ticks_processed" in snap
        assert "signals_generated" in snap
        assert "uptime_seconds" in snap

    def test_uptime_zero_before_start(self):
        m = EngineMetrics()
        assert m.uptime_seconds == 0.0

    def test_uptime_positive_after_start(self):
        m = EngineMetrics()
        m.engine_start_time = time.monotonic() - 10.0
        assert m.uptime_seconds >= 9.0

    def test_increment_counters(self):
        m = EngineMetrics()
        m.bars_processed += 1
        m.orders_filled += 1
        assert m.bars_processed == 1
        assert m.orders_filled == 1


class TestAsyncTradingEngineInit:
    """Test engine initialization and wiring."""

    @pytest.fixture
    def mock_components(self):
        config = OmegaConfig(symbols=["BTC/USDT"])
        exchange = AsyncMock()
        exchange.name = "binance"
        exchanges = {"binance": exchange}

        strategy = AsyncMock()
        strategy.strategy_id = "strat_1"
        strategy.required_symbols.return_value = ["BTC/USDT"]

        risk_engine = AsyncMock()
        return config, exchanges, [strategy], risk_engine

    def test_engine_creates(self, mock_components):
        config, exchanges, strategies, risk = mock_components
        engine = AsyncTradingEngine(config, exchanges, strategies, risk)
        assert engine._running is False
        assert engine.metrics.bars_processed == 0

    def test_engine_has_data_feed(self, mock_components):
        config, exchanges, strategies, risk = mock_components
        engine = AsyncTradingEngine(config, exchanges, strategies, risk)
        assert engine._data_feed is not None

    def test_engine_has_order_pipeline(self, mock_components):
        config, exchanges, strategies, risk = mock_components
        engine = AsyncTradingEngine(config, exchanges, strategies, risk)
        assert engine._order_pipeline is not None

    def test_engine_has_event_bus(self, mock_components):
        config, exchanges, strategies, risk = mock_components
        engine = AsyncTradingEngine(config, exchanges, strategies, risk)
        assert engine._event_bus is not None

    def test_positions_property(self, mock_components):
        config, exchanges, strategies, risk = mock_components
        engine = AsyncTradingEngine(config, exchanges, strategies, risk)
        assert engine.positions == {}

    def test_portfolio_initially_none(self, mock_components):
        config, exchanges, strategies, risk = mock_components
        engine = AsyncTradingEngine(config, exchanges, strategies, risk)
        assert engine.portfolio is None


class TestAsyncTradingEngineEvents:
    """Test engine event handlers."""

    @pytest.fixture
    def engine(self):
        config = OmegaConfig(symbols=["BTC/USDT"])
        exchange = AsyncMock()
        exchanges = {"binance": exchange}
        strategy = AsyncMock()
        strategy.strategy_id = "strat_1"
        risk = AsyncMock()
        return AsyncTradingEngine(config, exchanges, [strategy], risk)

    @pytest.mark.asyncio
    async def test_on_bar_increments_counter(self, engine):
        bar = OHLCVBar(
            timestamp=datetime.now(timezone.utc), symbol="BTC/USDT", timeframe=Timeframe.H1,
            open=50000, high=51000, low=49000, close=50500, volume=100, exchange="binance",
        )
        await engine._on_bar(bar)
        assert engine.metrics.bars_processed == 1
        assert engine.metrics.last_bar_time is not None

    @pytest.mark.asyncio
    async def test_on_tick_increments_counter(self, engine):
        from tradingbot.core.types import Tick
        tick = Tick(
            timestamp=datetime.now(timezone.utc), symbol="BTC/USDT",
            price=50000.0, quantity=1.0, side=Side.BUY, exchange="binance",
        )
        await engine._on_tick(tick)
        assert engine.metrics.ticks_processed == 1

    @pytest.mark.asyncio
    async def test_on_order_rejected(self, engine):
        from tradingbot.core.types import Order
        order = Order(id="reject-1", metadata={"reject_reason": "test"})
        await engine._on_order_rejected(order)
        assert engine.metrics.orders_rejected == 1

    @pytest.mark.asyncio
    async def test_on_order_cancelled(self, engine):
        from tradingbot.core.types import Order
        order = Order(id="cancel-1")
        await engine._on_order_cancelled(order)
        assert engine.metrics.orders_cancelled == 1

    @pytest.mark.asyncio
    async def test_on_position_updated(self, engine):
        pos = Position(
            symbol="BTC/USDT", strategy_id="strat_1", side=Side.BUY,
            quantity=0.5, avg_entry_price=50000.0, current_price=51000.0,
        )
        await engine._on_position_updated(pos)
        assert "BTC/USDT:strat_1" in engine.positions
        assert engine.metrics.open_positions == 1
