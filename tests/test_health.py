"""Tests for health checks — API server endpoints, liveness, readiness."""
from __future__ import annotations

import pytest

from tradingbot.infrastructure.api_server import APIServer, HAS_FASTAPI
from tradingbot.engine.async_engine import EngineMetrics


class TestAPIServerInit:
    def test_default_config(self):
        server = APIServer()
        assert server.port == 8080
        assert server.host == "0.0.0.0"

    def test_custom_config(self):
        server = APIServer({"port": 9090, "host": "127.0.0.1"})
        assert server.port == 9090
        assert server.host == "127.0.0.1"

    def test_websocket_clients_empty(self):
        server = APIServer()
        assert server._websocket_clients == []


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
class TestHealthEndpoints:
    @pytest.fixture
    def server(self):
        return APIServer()

    @pytest.fixture
    def app(self, server):
        return server.get_app()

    @pytest.fixture
    def client(self, app):
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Omega Trading Bot"
        assert data["status"] == "running"

    def test_portfolio_endpoint(self, client):
        response = client.get("/api/portfolio")
        assert response.status_code == 200
        data = response.json()
        assert "equity" in data
        assert "cash" in data

    def test_strategies_endpoint(self, client):
        response = client.get("/api/strategies")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_strategy_not_found(self, client):
        response = client.get("/api/strategies/nonexistent")
        assert response.status_code == 404

    def test_risk_endpoint(self, client):
        response = client.get("/api/risk")
        assert response.status_code == 200
        data = response.json()
        assert "max_drawdown" in data
        assert "current_drawdown" in data

    def test_evolution_endpoint(self, client):
        response = client.get("/api/evolution")
        assert response.status_code == 200
        data = response.json()
        assert "generation" in data

    def test_trades_endpoint(self, client):
        response = client.get("/api/trades")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_signals_endpoint(self, client):
        response = client.get("/api/signals")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_equity_endpoint(self, client):
        response = client.get("/api/equity")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_config_endpoint(self, client):
        response = client.get("/api/config")
        assert response.status_code == 200

    def test_start_strategy(self, client):
        response = client.post("/api/strategies/test_strat/start")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"

    def test_stop_strategy(self, client):
        response = client.post("/api/strategies/test_strat/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"


class TestEngineHealthMetrics:
    """Test that EngineMetrics provides health-check-compatible data."""

    def test_metrics_snapshot_for_health(self):
        m = EngineMetrics()
        m.engine_start_time = m.engine_start_time or 0
        snap = m.snapshot()
        assert isinstance(snap, dict)
        assert all(isinstance(v, (int, float, str, type(None))) for v in snap.values())

    def test_metrics_uptime_in_snapshot(self):
        m = EngineMetrics()
        snap = m.snapshot()
        assert "uptime_seconds" in snap
        assert snap["uptime_seconds"] == 0.0
