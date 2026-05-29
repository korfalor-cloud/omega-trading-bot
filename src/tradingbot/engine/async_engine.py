"""Production asyncio trading engine.

Manages the full trading lifecycle:
- Event loop creation and teardown
- Concurrent strategy execution across symbols/timeframes
- Multi-stream data feed orchestration
- Order routing through the pipeline (risk gate -> exchange -> fill tracking)
- Position synchronisation with exchange state
- Graceful startup / shutdown with SIGTERM / SIGINT handlers
- Health-check HTTP endpoint (FastAPI)
- Prometheus-compatible metrics collection
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from dataclasses import dataclass
from typing import Any, Optional

from ..config import OmegaConfig
from ..core.enums import Side
from ..core.events import Event, EventBus
from ..core.interfaces import ExchangeAdapter, RiskEngine, Strategy
from ..core.types import Fill, OHLCVBar, Order, Position, PortfolioState, Signal, Tick
from .data_feed import DataFeedManager
from .order_pipeline import OrderPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metrics collector (lightweight, Prometheus-compatible)
# ---------------------------------------------------------------------------


@dataclass
class EngineMetrics:
    """In-process counters and gauges exposed via /health and optionally Prometheus."""

    engine_start_time: float = 0.0
    bars_processed: int = 0
    ticks_processed: int = 0
    signals_generated: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    orders_cancelled: int = 0
    risk_checks_passed: int = 0
    risk_checks_failed: int = 0
    data_feed_reconnects: int = 0
    strategy_errors: int = 0
    pipeline_errors: int = 0

    # Gauges
    active_strategies: int = 0
    open_positions: int = 0
    event_queue_depth: int = 0
    active_data_streams: int = 0

    # Timestamps
    last_bar_time: Optional[str] = None
    last_order_time: Optional[str] = None

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self.engine_start_time if self.engine_start_time else 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "uptime_seconds": round(self.uptime_seconds, 1),
            "bars_processed": self.bars_processed,
            "ticks_processed": self.ticks_processed,
            "signals_generated": self.signals_generated,
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "orders_cancelled": self.orders_cancelled,
            "risk_checks_passed": self.risk_checks_passed,
            "risk_checks_failed": self.risk_checks_failed,
            "data_feed_reconnects": self.data_feed_reconnects,
            "strategy_errors": self.strategy_errors,
            "pipeline_errors": self.pipeline_errors,
            "active_strategies": self.active_strategies,
            "open_positions": self.open_positions,
            "event_queue_depth": self.event_queue_depth,
            "active_data_streams": self.active_data_streams,
            "last_bar_time": self.last_bar_time,
            "last_order_time": self.last_order_time,
        }


# ---------------------------------------------------------------------------
# Health check server (FastAPI / uvicorn, optional)
# ---------------------------------------------------------------------------

async def _start_health_server(
    host: str,
    port: int,
    metrics: EngineMetrics,
    engine_ref: "AsyncTradingEngine",
) -> None:
    """Start a minimal FastAPI health endpoint in the background."""
    try:
        from fastapi import FastAPI
        import uvicorn
    except ImportError:
        logger.warning("fastapi/uvicorn not installed -- health endpoint disabled")
        return

    app = FastAPI(title="Omega Health")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok" if engine_ref._running else "shutting_down",
            "metrics": metrics.snapshot(),
        }

    @app.get("/metrics")
    async def prometheus_metrics() -> str:
        lines: list[str] = []
        for key, value in metrics.snapshot().items():
            if isinstance(value, (int, float)):
                lines.append(f"omega_{key} {value}")
        return "\n".join(lines) + "\n"

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class AsyncTradingEngine:
    """Production asyncio trading engine.

    Lifecycle::

        engine = AsyncTradingEngine(config, exchanges, strategies, risk_engine)
        await engine.start()   # blocks until shutdown signal

    The engine wires together:
    * ``DataFeedManager``  -- multiple concurrent websocket / REST data streams
    * ``OrderPipeline``    -- pre-trade risk, validation, routing, fill tracking
    * ``EventBus``         -- pub/sub for bar/tick/signal/fill events
    * Strategy tasks       -- one ``asyncio.Task`` per strategy
    * Health endpoint      -- optional FastAPI server
    """

    def __init__(
        self,
        config: OmegaConfig,
        exchanges: dict[str, ExchangeAdapter],
        strategies: list[Strategy],
        risk_engine: RiskEngine,
    ) -> None:
        self._config = config
        self._exchanges = exchanges
        self._strategies = {s.strategy_id: s for s in strategies}
        self._risk_engine = risk_engine

        # Internal subsystems
        self._event_bus = EventBus()
        self._metrics = EngineMetrics()
        self._positions: dict[str, Position] = {}
        self._portfolio: Optional[PortfolioState] = None

        self._data_feed = DataFeedManager(
            exchanges=exchanges,
            event_bus=self._event_bus,
            symbols=config.symbols,
        )

        self._order_pipeline = OrderPipeline(
            exchanges=exchanges,
            risk_engine=risk_engine,
            event_bus=self._event_bus,
        )

        # Task bookkeeping
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._shutdown_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Full startup sequence: signal handlers, subsystems, strategy tasks."""
        self._metrics.engine_start_time = time.monotonic()
        self._running = True

        logger.info("=" * 60)
        logger.info("  ASYNC TRADING ENGINE -- STARTING")
        logger.info("  Symbols: %s", self._config.symbols)
        logger.info("  Strategies: %d", len(self._strategies))
        logger.info("  Exchanges: %s", list(self._exchanges.keys()))
        logger.info("=" * 60)

        loop = asyncio.get_running_loop()

        # Register signal handlers for graceful shutdown
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_signal, sig)

        # Wire event bus subscribers
        self._event_bus.subscribe(Event.BAR_CLOSED, self._on_bar)
        self._event_bus.subscribe(Event.TICK_RECEIVED, self._on_tick)
        self._event_bus.subscribe(Event.ORDER_FILLED, self._on_fill)
        self._event_bus.subscribe(Event.ORDER_REJECTED, self._on_order_rejected)
        self._event_bus.subscribe(Event.ORDER_CANCELLED, self._on_order_cancelled)
        self._event_bus.subscribe(Event.RISK_ALERT, self._on_risk_alert)
        self._event_bus.subscribe(Event.POSITION_UPDATED, self._on_position_updated)

        # Start subsystems
        self._tasks.append(asyncio.create_task(self._event_bus.run(), name="event_bus"))
        self._tasks.append(asyncio.create_task(self._data_feed.start(), name="data_feed"))
        self._tasks.append(asyncio.create_task(self._order_pipeline.start(), name="order_pipeline"))

        # Start strategy tasks
        for sid, strategy in self._strategies.items():
            self._tasks.append(
                asyncio.create_task(self._run_strategy(strategy), name=f"strategy:{sid}")
            )

        # Position sync loop
        self._tasks.append(asyncio.create_task(self._position_sync_loop(), name="pos_sync"))

        # Metrics heartbeat
        self._tasks.append(asyncio.create_task(self._metrics_loop(), name="metrics"))

        # Health endpoint
        health_port = int(os.environ.get("OMEGA_HEALTH_PORT", "8080"))
        self._tasks.append(
            asyncio.create_task(
                _start_health_server("0.0.0.0", health_port, self._metrics, self),
                name="health_server",
            )
        )

        logger.info("Engine fully started -- %d background tasks", len(self._tasks))

        # Block until shutdown signal
        await self._shutdown_event.wait()

        # Graceful teardown
        await self.stop()

    async def stop(self) -> None:
        """Gracefully tear down all subsystems and tasks."""
        if not self._running:
            return
        self._running = False
        logger.info("Engine shutting down ...")

        # Stop data feeds first (prevents new bars/ticks)
        await self._data_feed.stop()

        # Drain pending orders
        await self._order_pipeline.flush()

        # Publish shutdown event so strategies can clean up
        await self._event_bus.publish(Event.SHUTDOWN, None)
        # Small yield so handlers can react
        await asyncio.sleep(0.1)

        # Cancel all background tasks
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Disconnect exchanges
        for name, exchange in self._exchanges.items():
            try:
                await exchange.disconnect()
                logger.info("Exchange %s disconnected", name)
            except Exception:
                logger.exception("Error disconnecting exchange %s", name)

        await self._event_bus.stop()
        logger.info("Engine stopped cleanly")

    @property
    def metrics(self) -> EngineMetrics:
        return self._metrics

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    @property
    def portfolio(self) -> Optional[PortfolioState]:
        return self._portfolio

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _handle_signal(self, sig: signal.Signals) -> None:
        logger.info("Received signal %s -- initiating shutdown", sig.name)
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Event handlers (called from EventBus)
    # ------------------------------------------------------------------

    async def _on_bar(self, bar: OHLCVBar) -> None:
        self._metrics.bars_processed += 1
        self._metrics.last_bar_time = bar.timestamp.isoformat()

    async def _on_tick(self, tick: Tick) -> None:
        self._metrics.ticks_processed += 1

    async def _on_fill(self, fill: Fill) -> None:
        self._metrics.orders_filled += 1
        self._metrics.last_order_time = fill.timestamp.isoformat()
        # Update positions
        await self._apply_fill(fill)

    async def _on_order_rejected(self, order: Order) -> None:
        self._metrics.orders_rejected += 1
        logger.warning("Order rejected: %s -- %s", order.id, order.metadata.get("reject_reason", ""))

    async def _on_order_cancelled(self, order: Order) -> None:
        self._metrics.orders_cancelled += 1

    async def _on_risk_alert(self, alert: Any) -> None:
        logger.warning("Risk alert: %s", alert)

    async def _on_position_updated(self, position: Position) -> None:
        key = f"{position.symbol}:{position.strategy_id}"
        self._positions[key] = position
        self._metrics.open_positions = len(self._positions)

    # ------------------------------------------------------------------
    # Strategy execution
    # ------------------------------------------------------------------

    async def _run_strategy(self, strategy: Strategy) -> None:
        """Consumer loop: wait for bars/ticks from the EventBus, feed them to the strategy."""
        logger.info("Strategy %s started", strategy.strategy_id)
        self._metrics.active_strategies += 1

        bar_queue: asyncio.Queue[OHLCVBar] = asyncio.Queue(maxsize=5_000)
        tick_queue: asyncio.Queue[Tick] = asyncio.Queue(maxsize=50_000)

        async def _bar_enqueue(bar: OHLCVBar) -> None:
            # Only forward bars this strategy cares about
            if bar.symbol in strategy.required_symbols():
                try:
                    bar_queue.put_nowait(bar)
                except asyncio.QueueFull:
                    logger.warning("Bar queue full for %s", strategy.strategy_id)

        async def _tick_enqueue(tick: Tick) -> None:
            if tick.symbol in strategy.required_symbols():
                try:
                    tick_queue.put_nowait(tick)
                except asyncio.QueueFull:
                    pass  # drop ticks on overflow

        self._event_bus.subscribe(Event.BAR_CLOSED, _bar_enqueue)
        self._event_bus.subscribe(Event.TICK_RECEIVED, _tick_enqueue)

        try:
            while self._running:
                try:
                    # Priority: bars drive most strategies
                    bar = await asyncio.wait_for(bar_queue.get(), timeout=0.5)
                    signal_result = await strategy.on_bar(bar)
                    if signal_result is not None:
                        await self._process_signal(signal_result)
                except asyncio.TimeoutError:
                    # Drain ticks while waiting for bars
                    while not tick_queue.empty():
                        tick = tick_queue.get_nowait()
                        signal_result = await strategy.on_tick(tick)
                        if signal_result is not None:
                            await self._process_signal(signal_result)
                    continue
                except asyncio.CancelledError:
                    break
                except Exception:
                    self._metrics.strategy_errors += 1
                    logger.exception("Strategy %s error", strategy.strategy_id)
                    await asyncio.sleep(1.0)  # backoff on error
        finally:
            self._metrics.active_strategies -= 1
            logger.info("Strategy %s stopped", strategy.strategy_id)

    async def _process_signal(self, signal: Signal) -> None:
        """Route a strategy signal through the order pipeline."""
        self._metrics.signals_generated += 1
        order = await self._order_pipeline.submit_signal(signal, self._portfolio)
        if order is not None:
            self._metrics.orders_submitted += 1

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    async def _apply_fill(self, fill: Fill) -> None:
        """Update local position book after a fill."""
        key_candidates = [k for k in self._positions if k.startswith(fill.symbol + ":")]
        if key_candidates:
            pos = self._positions[key_candidates[0]]
            if fill.side == Side.BUY:
                # Increase / flip long
                total_cost = pos.avg_entry_price * pos.quantity + fill.price * fill.quantity
                pos.quantity += fill.quantity
                pos.avg_entry_price = total_cost / pos.quantity if pos.quantity else 0.0
            else:
                # Reduce long or increase short
                pnl = (fill.price - pos.avg_entry_price) * fill.quantity
                if pos.side == Side.BUY:
                    pos.realized_pnl += pnl
                else:
                    pos.realized_pnl -= pnl
                pos.quantity -= fill.quantity
                if pos.quantity <= 0:
                    # Position closed or flipped
                    if pos.quantity < 0:
                        pos.side = Side.SELL
                        pos.quantity = abs(pos.quantity)
                        pos.avg_entry_price = fill.price
                    else:
                        del self._positions[key_candidates[0]]
            await self._event_bus.publish(Event.POSITION_UPDATED, pos)
        else:
            # New position
            pos = Position(
                symbol=fill.symbol,
                strategy_id="",
                side=fill.side,
                quantity=fill.quantity,
                avg_entry_price=fill.price,
                current_price=fill.price,
                exchange=fill.exchange,
            )
            self._positions[f"{fill.symbol}:"] = pos
            await self._event_bus.publish(Event.POSITION_UPDATED, pos)

    async def _position_sync_loop(self) -> None:
        """Periodically reconcile local positions with exchange state."""
        sync_interval = int(os.environ.get("OMEGA_POSITION_SYNC_SECONDS", "60"))
        while self._running:
            try:
                for exchange_name, exchange in self._exchanges.items():
                    if not exchange.is_connected:
                        continue
                    remote_positions = await exchange.fetch_positions()
                    for rp in remote_positions:
                        key = f"{rp.symbol}:{rp.strategy_id}"
                        local = self._positions.get(key)
                        if local is None:
                            self._positions[key] = rp
                        else:
                            # Reconcile quantity / price drift
                            local.quantity = rp.quantity
                            local.current_price = rp.current_price
                            local.unrealized_pnl = rp.unrealized_pnl
                            await self._event_bus.publish(Event.POSITION_UPDATED, local)

                self._metrics.open_positions = len(self._positions)

                # Refresh portfolio state
                self._portfolio = await self._risk_engine.get_portfolio_state()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Position sync error")
            await asyncio.sleep(sync_interval)

    # ------------------------------------------------------------------
    # Metrics heartbeat
    # ------------------------------------------------------------------

    async def _metrics_loop(self) -> None:
        """Periodically log engine metrics."""
        while self._running:
            try:
                self._metrics.event_queue_depth = self._event_bus.queue_size
                self._metrics.active_data_streams = self._data_feed.active_stream_count
                snap = self._metrics.snapshot()
                logger.info("METRICS | %s", snap)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Metrics loop error")
            await asyncio.sleep(30)
