"""Async order pipeline.

Processes orders through a multi-stage pipeline:
1. Signal -> Order conversion (sizing, price estimation)
2. Pre-trade risk checks (position limits, drawdown, exposure)
3. Order validation (symbol, quantity, price sanity)
4. Exchange routing (submit to correct venue)
5. Fill tracking (order status polling / websocket)
6. Position update on fill
7. P&L calculation on fill
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ..core.enums import OrderState, OrderType
from ..core.errors import OrderError, OrderRejectedError
from ..core.events import Event, EventBus
from ..core.interfaces import ExchangeAdapter, RiskEngine
from ..core.types import Fill, Order, PortfolioState, RiskCheck, Signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline metrics
# ---------------------------------------------------------------------------

@dataclass
class PipelineMetrics:
    """Counters for the order pipeline."""
    signals_received: int = 0
    orders_created: int = 0
    risk_passed: int = 0
    risk_failed: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    orders_cancelled: int = 0
    fills_tracked: int = 0
    pipeline_errors: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "signals_received": self.signals_received,
            "orders_created": self.orders_created,
            "risk_passed": self.risk_passed,
            "risk_failed": self.risk_failed,
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "orders_cancelled": self.orders_cancelled,
            "fills_tracked": self.fills_tracked,
            "pipeline_errors": self.pipeline_errors,
        }


# ---------------------------------------------------------------------------
# Order Pipeline
# ---------------------------------------------------------------------------

class OrderPipeline:
    """Async pipeline that turns trading signals into exchange orders.

    Lifecycle::

        pipeline = OrderPipeline(exchanges, risk_engine, event_bus)
        await pipeline.start()
        order = await pipeline.submit_signal(signal, portfolio)
        ...
        await pipeline.flush()   # drain pending orders
        await pipeline.stop()
    """

    def __init__(
        self,
        exchanges: dict[str, ExchangeAdapter],
        risk_engine: RiskEngine,
        event_bus: EventBus,
        default_exchange: Optional[str] = None,
    ) -> None:
        self._exchanges = exchanges
        self._risk_engine = risk_engine
        self._event_bus = event_bus
        self._default_exchange = default_exchange or (next(iter(exchanges)) if exchanges else "")

        self._open_orders: dict[str, Order] = {}
        self._metrics = PipelineMetrics()
        self._fill_queue: asyncio.Queue[Fill] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._running = False

    @property
    def metrics(self) -> PipelineMetrics:
        return self._metrics

    @property
    def open_orders(self) -> dict[str, Order]:
        return dict(self._open_orders)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start fill-tracking background tasks."""
        self._running = True
        self._tasks.append(
            asyncio.create_task(self._fill_processor_loop(), name="fill_processor")
        )
        self._tasks.append(
            asyncio.create_task(self._order_status_poller_loop(), name="order_poller")
        )
        logger.info("OrderPipeline started with %d background tasks", len(self._tasks))

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("OrderPipeline stopped")

    async def flush(self) -> None:
        """Cancel all open orders and drain the fill queue."""
        for order in list(self._open_orders.values()):
            try:
                await self._cancel_order(order)
            except Exception:
                logger.exception("Error cancelling order %s during flush", order.id)
        # Drain fill queue
        while not self._fill_queue.empty():
            try:
                self._fill_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ------------------------------------------------------------------
    # Signal -> Order conversion
    # ------------------------------------------------------------------

    async def submit_signal(
        self,
        signal: Signal,
        portfolio: Optional[PortfolioState] = None,
    ) -> Optional[Order]:
        """Full pipeline: signal -> risk check -> validate -> route -> track.

        Returns the created ``Order`` if successfully submitted, ``None`` if
        rejected by risk or validation.
        """
        self._metrics.signals_received += 1

        try:
            # Step 1: Convert signal to order
            order = self._signal_to_order(signal)
            self._metrics.orders_created += 1

            # Step 2: Pre-trade risk check
            if portfolio is not None:
                risk_check = await self._risk_engine.pre_trade_check(signal, portfolio)
            else:
                risk_check = RiskCheck(approved=True, reason="No portfolio state -- skipping risk check")

            if not risk_check.approved:
                self._metrics.risk_failed += 1
                order.state = OrderState.REJECTED
                order.metadata["reject_reason"] = risk_check.reason
                logger.warning(
                    "Order %s rejected by risk: %s (warnings=%s)",
                    order.id, risk_check.reason, risk_check.warnings,
                )
                await self._event_bus.publish(Event.ORDER_REJECTED, order)
                return None

            self._metrics.risk_passed += 1

            # Clamp quantity to risk-approved max
            if risk_check.max_allowed_quantity > 0:
                order.quantity = min(order.quantity, risk_check.max_allowed_quantity)

            # Step 3: Validate order
            self._validate_order(order)

            # Step 4: Route to exchange
            submitted_order = await self._route_to_exchange(order)
            self._metrics.orders_submitted += 1
            self._open_orders[submitted_order.id] = submitted_order

            await self._event_bus.publish(Event.ORDER_SUBMITTED, submitted_order)
            return submitted_order

        except OrderRejectedError as exc:
            self._metrics.risk_failed += 1
            logger.warning("Signal rejected: %s", exc)
            return None
        except OrderError as exc:
            self._metrics.pipeline_errors += 1
            logger.error("Order pipeline error: %s", exc)
            return None
        except Exception:
            self._metrics.pipeline_errors += 1
            logger.exception("Unexpected pipeline error")
            return None

    # ------------------------------------------------------------------
    # Internal pipeline stages
    # ------------------------------------------------------------------

    def _signal_to_order(self, signal: Signal) -> Order:
        """Derive an Order from a Signal with position sizing."""
        # Simple sizing: use signal metadata or default
        quantity = signal.metadata.get("quantity", 0.0)
        if quantity <= 0:
            # Default sizing: fixed fraction (caller should provide via metadata)
            quantity = signal.metadata.get("default_quantity", 0.001)

        price = signal.metadata.get("limit_price")
        order_type = OrderType.LIMIT if price is not None else OrderType.MARKET

        exchange_name = signal.metadata.get("exchange", self._default_exchange)

        return Order(
            id=str(uuid.uuid4()),
            strategy_id=signal.strategy_id,
            symbol=signal.symbol,
            side=signal.side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=signal.stop_loss,
            state=OrderState.PENDING,
            exchange=exchange_name,
            metadata={
                "signal_id": signal.id,
                "signal_strength": signal.strength,
                "signal_confidence": signal.confidence,
                "take_profit": signal.take_profit,
                "trailing_stop_atr_mult": signal.trailing_stop_atr_mult,
            },
        )

    def _validate_order(self, order: Order) -> None:
        """Raise ``OrderRejectedError`` if the order is invalid."""
        if not order.symbol:
            raise OrderRejectedError("Empty symbol")
        if order.quantity <= 0:
            raise OrderRejectedError(f"Non-positive quantity: {order.quantity}")
        if order.order_type == OrderType.LIMIT and (order.price is None or order.price <= 0):
            raise OrderRejectedError(f"Limit order requires positive price, got {order.price}")
        if order.exchange not in self._exchanges and self._exchanges:
            raise OrderRejectedError(f"Unknown exchange: {order.exchange}")

    async def _route_to_exchange(self, order: Order) -> Order:
        """Submit the order to the appropriate exchange adapter."""
        exchange = self._exchanges.get(order.exchange)
        if exchange is None:
            # Fallback to default
            exchange = self._exchanges.get(self._default_exchange)
        if exchange is None:
            raise OrderError(f"No exchange adapter for {order.exchange}")

        order.state = OrderState.SUBMITTED
        submitted = await exchange.submit_order(order)
        return submitted

    async def _cancel_order(self, order: Order) -> None:
        """Cancel an open order on the exchange."""
        exchange = self._exchanges.get(order.exchange)
        if exchange is None:
            return
        try:
            cancelled = await exchange.cancel_order(order.id, order.symbol)
            cancelled.state = OrderState.CANCELLED
            self._open_orders.pop(order.id, None)
            self._metrics.orders_cancelled += 1
            await self._event_bus.publish(Event.ORDER_CANCELLED, cancelled)
        except Exception:
            logger.exception("Failed to cancel order %s", order.id)

    # ------------------------------------------------------------------
    # Fill tracking
    # ------------------------------------------------------------------

    async def _fill_processor_loop(self) -> None:
        """Consume fills from the queue and publish events."""
        while self._running:
            try:
                fill = await asyncio.wait_for(self._fill_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                self._metrics.fills_tracked += 1
                self._metrics.orders_filled += 1

                # Update order state
                order = self._open_orders.get(fill.order_id)
                if order is not None:
                    order.filled_quantity += fill.quantity
                    order.avg_fill_price = fill.price
                    order.commission += fill.commission
                    if order.filled_quantity >= order.quantity:
                        order.state = OrderState.FILLED
                        self._open_orders.pop(order.id, None)
                    else:
                        order.state = OrderState.PARTIAL

                # Publish fill event (engine will update positions + P&L)
                await self._event_bus.publish(Event.ORDER_FILLED, fill)

                logger.info(
                    "Fill: %s %s %.6f @ %.2f (commission=%.4f)",
                    fill.side.value.upper(), fill.symbol, fill.quantity, fill.price, fill.commission,
                )
            except Exception:
                self._metrics.pipeline_errors += 1
                logger.exception("Fill processing error")

    async def _order_status_poller_loop(self) -> None:
        """Periodically poll exchange for order status updates."""
        while self._running:
            try:
                await asyncio.sleep(5)
                if not self._open_orders:
                    continue

                for order in list(self._open_orders.values()):
                    exchange = self._exchanges.get(order.exchange)
                    if exchange is None:
                        continue
                    try:
                        updated = await exchange.fetch_order(order.id, order.symbol)
                        if updated.state == OrderState.FILLED and order.state != OrderState.FILLED:
                            # New fill detected
                            fill = Fill(
                                order_id=order.id,
                                symbol=order.symbol,
                                side=order.side,
                                price=updated.avg_fill_price,
                                quantity=updated.filled_quantity - order.filled_quantity,
                                commission=updated.commission - order.commission,
                                exchange=order.exchange,
                                timestamp=datetime.now(timezone.utc),
                            )
                            await self._fill_queue.put(fill)
                        elif updated.state == OrderState.CANCELLED:
                            self._open_orders.pop(order.id, None)
                            self._metrics.orders_cancelled += 1
                            await self._event_bus.publish(Event.ORDER_CANCELLED, updated)
                        elif updated.state == OrderState.REJECTED:
                            self._open_orders.pop(order.id, None)
                            self._metrics.orders_rejected += 1
                            await self._event_bus.publish(Event.ORDER_REJECTED, updated)
                    except Exception:
                        logger.debug("Poll error for order %s", order.id, exc_info=True)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Order poller error")
