"""Tests for WebSocketManager — persistent connections with reconnection."""
from __future__ import annotations

import asyncio

import pytest

from tradingbot.infrastructure.websocket_manager import WebSocketManager, WSConnection


class TestWebSocketManager:
    @pytest.fixture
    def manager(self):
        return WebSocketManager()

    @pytest.fixture
    def configured_manager(self):
        return WebSocketManager({
            "max_reconnect": 5,
            "base_delay": 0.01,
            "max_delay": 0.5,
            "ping_interval": 10,
        })

    def test_register_connection(self, manager):
        handler = lambda data: None
        manager.register("wss://stream.binance.com", handler)
        assert "wss://stream.binance.com" in manager._connections
        assert len(manager._handlers["wss://stream.binance.com"]) == 1

    def test_register_multiple_handlers(self, manager):
        manager.register("wss://stream.binance.com", lambda d: None)
        manager.register("wss://stream.binance.com", lambda d: None)
        assert len(manager._handlers["wss://stream.binance.com"]) == 2

    @pytest.mark.asyncio
    async def test_connect(self, manager):
        manager.register("wss://stream.binance.com", lambda d: None)
        result = await manager.connect("wss://stream.binance.com")
        assert result is True
        assert manager.is_connected("wss://stream.binance.com")

    @pytest.mark.asyncio
    async def test_connect_unregistered(self, manager):
        result = await manager.connect("wss://unknown.url")
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect(self, manager):
        manager.register("wss://stream.binance.com", lambda d: None)
        await manager.connect("wss://stream.binance.com")
        assert manager.is_connected("wss://stream.binance.com")

        await manager.disconnect("wss://stream.binance.com")
        assert not manager.is_connected("wss://stream.binance.com")

    @pytest.mark.asyncio
    async def test_send_when_connected(self, manager):
        manager.register("wss://stream.binance.com", lambda d: None)
        await manager.connect("wss://stream.binance.com")
        result = await manager.send("wss://stream.binance.com", {"method": "subscribe"})
        assert result is True

    @pytest.mark.asyncio
    async def test_send_when_disconnected(self, manager):
        manager.register("wss://stream.binance.com", lambda d: None)
        result = await manager.send("wss://stream.binance.com", {"method": "subscribe"})
        assert result is False

    def test_dispatch_to_handlers(self, manager):
        received = []
        manager.register("wss://stream.binance.com", lambda d: received.append(d))
        manager.dispatch("wss://stream.binance.com", {"symbol": "BTC/USDT", "price": 50000})
        assert len(received) == 1
        assert received[0]["price"] == 50000

    def test_dispatch_unknown_url(self, manager):
        # Should not raise
        manager.dispatch("wss://unknown", {"data": 1})

    def test_get_status(self, manager):
        manager.register("wss://stream.binance.com", lambda d: None)
        status = manager.get_status()
        assert "wss://stream.binance.com" in status
        assert status["wss://stream.binance.com"]["connected"] is False
        assert status["wss://stream.binance.com"]["reconnect_count"] == 0

    def test_is_connected_unknown_url(self, manager):
        assert manager.is_connected("wss://unknown") is False

    def test_custom_config(self, configured_manager):
        assert configured_manager.max_reconnect == 5
        assert configured_manager.base_delay == 0.01
        assert configured_manager.max_delay == 0.5

    @pytest.mark.asyncio
    async def test_reconnect_success(self, configured_manager):
        configured_manager.register("wss://stream.binance.com", lambda d: None)
        result = await configured_manager.reconnect("wss://stream.binance.com")
        assert result is True
        assert configured_manager.is_connected("wss://stream.binance.com")

    @pytest.mark.asyncio
    async def test_reconnect_unregistered(self, configured_manager):
        result = await configured_manager.reconnect("wss://unknown")
        assert result is False
