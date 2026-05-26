"""Order Manager — Manages the complete order lifecycle."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

from ..core.enums import OrderState, OrderType, Side
from ..core.events import Event, EventBus
from ..core.types import Fill, Order, Position, Signal

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages order lifecycle: creation → submission → fill → position update.

    Converts signals into orders, tracks their state, and publishes events.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._active_orders: dict[str, Order] = {}
        self._order_history: list[Order] = []
        self._fill_history: list[Fill] = []

    async def create_order_from_signal(
        self,
        signal: Signal,
        portfolio_equity: float,
        current_price: float,
        exchange: str = "",
    ) -> Order:
        """Convert a signal into an order."""
        # Position sizing based on signal confidence
        max_position_pct = signal.metadata.get("max_position_pct", 0.05)
        position_value = portfolio_equity * max_position_pct * signal.confidence
        quantity = position_value / current_price if current_price > 0 else 0

        order = Order(
            strategy_id=signal.strategy_id,
            symbol=signal.symbol,
            side=signal.side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            exchange=exchange,
        )

        self._active_orders[order.id] = order
        logger.info(
            f"Order created: {order.side.value} {order.quantity:.4f} {order.symbol} "
            f"(signal confidence={signal.confidence:.2f})"
        )
        return order

    async def on_order_filled(self, order: Order, fill: Fill) -> None:
        """Handle order fill."""
        order.state = OrderState.FILLED
        order.filled_quantity = fill.quantity
        order.avg_fill_price = fill.price
        order.commission = fill.commission

        self._fill_history.append(fill)
        self._order_history.append(order)

        if order.id in self._active_orders:
            del self._active_orders[order.id]

        await self.event_bus.publish(Event.ORDER_FILLED, fill)
        logger.info(f"Order filled: {fill.side.value} {fill.quantity:.4f} {fill.symbol} @ {fill.price:.2f}")

    async def cancel_order(self, order_id: str) -> Optional[Order]:
        """Cancel an active order."""
        order = self._active_orders.get(order_id)
        if order:
            order.state = OrderState.CANCELLED
            del self._active_orders[order_id]
            await self.event_bus.publish(Event.ORDER_CANCELLED, order)
        return order

    async def cancel_all_orders(self, strategy_id: Optional[str] = None) -> int:
        """Cancel all active orders, optionally filtered by strategy."""
        cancelled = 0
        to_cancel = [
            oid for oid, order in self._active_orders.items()
            if strategy_id is None or order.strategy_id == strategy_id
        ]
        for oid in to_cancel:
            await self.cancel_order(oid)
            cancelled += 1
        return cancelled

    @property
    def active_orders(self) -> list[Order]:
        return list(self._active_orders.values())

    @property
    def fill_history(self) -> list[Fill]:
        return list(self._fill_history)
