"""Tests for graceful shutdown — signal handling, task cleanup, resource teardown."""
from __future__ import annotations

import asyncio
import signal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from tradingbot.config import OmegaConfig
from tradingbot.core.enums import Side, Timeframe
from tradingbot.core.events import Event, EventBus
from tradingbot.core.types import OHLCVBar, Position
from tradingbot.engine.async_engine import AsyncTradingEngine
from tradingbot.engine.data_feed import DataFeedManager
from tradingbot.engine.order_pipeline import OrderPipeline


class TestEventBusShutdown:
    @pytest.mark.asyncio
    async def test_stop_stops_running(self):
        bus = EventBus()
        task = asyncio.create_task(bus.run())
        await asyncio.sleep(0.05)
        assert bus._running is True
        await bus.stop()
        assert bus._running is False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_publish_shutdown_event(self):
        bus = EventBus()
        received = []

        async def handler(payload):
            received.append(payload)

        bus.subscribe(Event.SHUTDOWN, handler)
        task = asyncio.create_task(bus.run())
        await bus.publish(Event.SHUTDOWN, {"reason": "test"})
        await asyncio.sleep(0.1)
        await bus.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert len(received) == 1


class TestOrderPipelineShutdown:
    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self):
        pipeline = OrderPipeline(exchanges={}, risk_engine=AsyncMock(), event_bus=EventBus())
        await pipeline.start()
        assert pipeline._running is True
        assert len(pipeline._tasks) > 0

        await pipeline.stop()
        assert pipeline._running is False
        assert len(pipeline._tasks) == 0

    @pytest.mark.asyncio
    async def test_flush_drains_queue(self):
        pipeline = OrderPipeline(exchanges={}, risk_engine=AsyncMock(), event_bus=EventBus())
        pipeline._running = True
        await pipeline._fill_queue.put(MagicMock())
        await pipeline._fill_queue.put(MagicMock())
        assert pipeline._fill_queue.qsize() == 2

        await pipeline.flush()
        assert pipeline._fill_queue.qsize() == 0


class TestEngineShutdown:
    @pytest.fixture
    def mock_components(self):
        config = OmegaConfig(symbols=["BTC/USDT"])
        exchange = AsyncMock()
        exchange.name = "binance"
        exchange.is_connected = True
        exchanges = {"binance": exchange}
        strategy = AsyncMock()
        strategy.strategy_id = "strat_1"
        strategy.required_symbols.return_value = ["BTC/USDT"]
        risk = AsyncMock()
        risk.get_portfolio_state = AsyncMock(return_value=None)
        return config, exchanges, [strategy], risk

    def test_engine_not_running_initially(self, mock_components):
        config, exchanges, strategies, risk = mock_components
        engine = AsyncTradingEngine(config, exchanges, strategies, risk)
        assert engine._running is False

    def test_shutdown_event_initially_clear(self, mock_components):
        config, exchanges, strategies, risk = mock_components
        engine = AsyncTradingEngine(config, exchanges, strategies, risk)
        assert not engine._shutdown_event.is_set()

    def test_handle_signal_sets_shutdown(self, mock_components):
        config, exchanges, strategies, risk = mock_components
        engine = AsyncTradingEngine(config, exchanges, strategies, risk)
        engine._handle_signal(signal.SIGTERM)
        assert engine._shutdown_event.is_set()

    def test_handle_signal_different_signals(self, mock_components):
        config, exchanges, strategies, risk = mock_components
        engine = AsyncTradingEngine(config, exchanges, strategies, risk)
        engine._handle_signal(signal.SIGINT)
        assert engine._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_stop_disconnects_exchanges(self, mock_components):
        config, exchanges, strategies, risk = mock_components
        engine = AsyncTradingEngine(config, exchanges, strategies, risk)
        engine._running = True
        # Need to start data feed and pipeline tasks so stop works
        engine._tasks = []
        await engine.stop()
        exchanges["binance"].disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, mock_components):
        config, exchanges, strategies, risk = mock_components
        engine = AsyncTradingEngine(config, exchanges, strategies, risk)
        engine._running = False
        # Calling stop when already stopped should not raise
        await engine.stop()
        assert engine._running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_all_tasks(self, mock_components):
        config, exchanges, strategies, risk = mock_components
        engine = AsyncTradingEngine(config, exchanges, strategies, risk)
        engine._running = True

        # Create some mock tasks
        async def dummy():
            await asyncio.sleep(100)

        t1 = asyncio.create_task(dummy())
        t2 = asyncio.create_task(dummy())
        engine._tasks = [t1, t2]

        await engine.stop()
        assert t1.cancelled() or t1.done()
        assert t2.cancelled() or t2.done()


class TestDataFeedShutdown:
    @pytest.mark.asyncio
    async def test_stop_cancels_stream_tasks(self):
        bus = EventBus()
        manager = DataFeedManager(exchanges={}, event_bus=bus, symbols=["BTC/USDT"])

        async def dummy_stream():
            await asyncio.sleep(100)

        manager._tasks = [asyncio.create_task(dummy_stream())]
        manager._running = True

        await manager.stop()
        assert manager._running is False
        assert len(manager._tasks) == 0
