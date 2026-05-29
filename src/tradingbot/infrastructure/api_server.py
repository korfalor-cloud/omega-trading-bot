"""REST API Server — FastAPI-based monitoring and control.

Implements:
- Portfolio status endpoint
- Strategy management endpoints
- Trade history endpoints
- Signal history endpoints
- Evolution status endpoints
- Config management endpoints
- WebSocket for real-time updates
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# FastAPI is optional — graceful degradation
try:
    from fastapi import FastAPI, WebSocket, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


class APIServer:
    """REST API for monitoring and control."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.port = config.get("port", 8080)
        self.host = config.get("host", "0.0.0.0")
        self._app = None
        self._websocket_clients: list = []

        if HAS_FASTAPI:
            self._setup_app()

    def _setup_app(self):
        self._app = FastAPI(title="Omega Trading Bot", version="1.0.0")
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._register_routes()

    def _register_routes(self):
        app = self._app

        @app.get("/")
        async def root():
            return {"name": "Omega Trading Bot", "version": "1.0.0", "status": "running"}

        @app.get("/api/portfolio")
        async def get_portfolio():
            return self._get_portfolio()

        @app.get("/api/strategies")
        async def get_strategies():
            return self._get_strategies()

        @app.get("/api/strategies/{strategy_id}")
        async def get_strategy(strategy_id: str):
            result = self._get_strategy(strategy_id)
            if not result:
                raise HTTPException(404, "Strategy not found")
            return result

        @app.post("/api/strategies/{strategy_id}/start")
        async def start_strategy(strategy_id: str):
            return self._start_strategy(strategy_id)

        @app.post("/api/strategies/{strategy_id}/stop")
        async def stop_strategy(strategy_id: str):
            return self._stop_strategy(strategy_id)

        @app.get("/api/trades")
        async def get_trades(limit: int = 100, strategy_id: str = "", symbol: str = ""):
            return self._get_trades(limit, strategy_id, symbol)

        @app.get("/api/signals")
        async def get_signals(limit: int = 100, strategy_id: str = ""):
            return self._get_signals(limit, strategy_id)

        @app.get("/api/risk")
        async def get_risk():
            return self._get_risk()

        @app.get("/api/evolution")
        async def get_evolution():
            return self._get_evolution()

        @app.get("/api/equity")
        async def get_equity(limit: int = 1000):
            return self._get_equity(limit)

        @app.get("/api/config")
        async def get_config():
            return self._get_config()

        @app.put("/api/config/{section}/{key}")
        async def update_config(section: str, key: str, value: str):
            return self._update_config(section, key, value)

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._websocket_clients.append(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    # Handle commands
                    await websocket.send_text(json.dumps({"type": "ack", "data": data}))
            except Exception:
                self._websocket_clients.remove(websocket)

    # ── Data Providers (override these) ──────────────────────────

    def _get_portfolio(self) -> dict:
        return {
            "equity": 100000,
            "cash": 50000,
            "positions_value": 50000,
            "daily_pnl": 0,
            "total_pnl": 0,
            "leverage": 0.5,
        }

    def _get_strategies(self) -> list:
        return []

    def _get_strategy(self, strategy_id: str) -> Optional[dict]:
        return None

    def _start_strategy(self, strategy_id: str) -> dict:
        return {"status": "started", "strategy_id": strategy_id}

    def _stop_strategy(self, strategy_id: str) -> dict:
        return {"status": "stopped", "strategy_id": strategy_id}

    def _get_trades(self, limit: int, strategy_id: str, symbol: str) -> list:
        return []

    def _get_signals(self, limit: int, strategy_id: str) -> list:
        return []

    def _get_risk(self) -> dict:
        return {
            "max_drawdown": 0,
            "current_drawdown": 0,
            "daily_loss": 0,
            "leverage": 0,
            "var_95": 0,
        }

    def _get_evolution(self) -> dict:
        return {
            "generation": 0,
            "population_size": 0,
            "best_fitness": 0,
            "avg_fitness": 0,
        }

    def _get_equity(self, limit: int) -> list:
        return []

    def _get_config(self) -> dict:
        return {}

    def _update_config(self, section: str, key: str, value: str) -> dict:
        return {"status": "updated", "section": section, "key": key}

    # ── WebSocket Broadcasting ───────────────────────────────────

    async def broadcast(self, message: dict) -> None:
        """Broadcast message to all WebSocket clients."""
        for ws in self._websocket_clients[:]:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                self._websocket_clients.remove(ws)

    # ── Server Control ───────────────────────────────────────────

    def run(self) -> None:
        """Start the API server."""
        if not HAS_FASTAPI:
            logger.warning("FastAPI not installed, API server disabled")
            return

        import uvicorn
        uvicorn.run(self._app, host=self.host, port=self.port)

    def get_app(self):
        """Get the FastAPI app for external mounting."""
        return self._app
