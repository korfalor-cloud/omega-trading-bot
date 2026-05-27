"""Order Management System.

Implements:
- Order lifecycle management (create, amend, cancel)
- Order state machine
- Fill tracking and partial fills
- Order book management
- Smart order routing
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = auto()
    SUBMITTED = auto()
    PARTIAL = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


@dataclass
class Fill:
    """A single fill (execution)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    price: float = 0.0
    quantity: float = 0.0
    fee: float = 0.0
    fee_currency: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_maker: bool = False


@dataclass
class ManagedOrder:
    """An order managed by the OMS."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: str = ""
    order_type: str = "limit"
    price: float = 0.0
    quantity: float = 0.0
    filled_quantity: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    fills: list[Fill] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    time_in_force: str = "GTC"
    expire_at: Optional[datetime] = None
    strategy_id: str = ""
    parent_order_id: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL)

    @property
    def average_fill_price(self) -> float:
        if not self.fills:
            return 0.0
        total_value = sum(f.price * f.quantity for f in self.fills)
        total_qty = sum(f.quantity for f in self.fills)
        return total_value / total_qty if total_qty > 0 else 0.0

    @property
    def total_fees(self) -> float:
        return sum(f.fee for f in self.fills)


class OrderManager:
    """Order Management System."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.max_open_orders = config.get("max_open_orders", 100)
        self.default_fee_rate = config.get("default_fee_rate", 0.001)
        self._orders: dict[str, ManagedOrder] = {}
        self._fills: list[Fill] = []

    def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float = 0.0,
        order_type: str = "limit",
        strategy_id: str = "",
        time_in_force: str = "GTC",
    ) -> ManagedOrder:
        open_count = sum(1 for o in self._orders.values() if o.is_active)
        if open_count >= self.max_open_orders:
            raise ValueError(f"Max open orders ({self.max_open_orders}) reached")

        order = ManagedOrder(
            symbol=symbol, side=side, order_type=order_type,
            price=price, quantity=quantity, strategy_id=strategy_id,
            time_in_force=time_in_force,
        )
        order.status = OrderStatus.SUBMITTED
        self._orders[order.id] = order
        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if not order or not order.is_active:
            return False
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()
        return True

    def amend_order(self, order_id: str, new_price: float | None = None, new_quantity: float | None = None) -> bool:
        order = self._orders.get(order_id)
        if not order or not order.is_active:
            return False
        if new_price is not None:
            order.price = new_price
        if new_quantity is not None:
            if new_quantity < order.filled_quantity:
                return False
            order.quantity = new_quantity
        order.updated_at = datetime.utcnow()
        return True

    def process_fill(self, order_id: str, price: float, quantity: float, fee: float = 0.0, is_maker: bool = False) -> Fill | None:
        order = self._orders.get(order_id)
        if not order or not order.is_active:
            return None

        fill = Fill(
            order_id=order_id, price=price, quantity=quantity,
            fee=fee if fee > 0 else quantity * price * self.default_fee_rate,
            is_maker=is_maker,
        )
        order.fills.append(fill)
        order.filled_quantity += quantity
        self._fills.append(fill)

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIAL

        order.updated_at = datetime.utcnow()
        return fill

    def get_order(self, order_id: str) -> ManagedOrder | None:
        return self._orders.get(order_id)

    def get_active_orders(self, symbol: str = "") -> list[ManagedOrder]:
        orders = [o for o in self._orders.values() if o.is_active]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_filled_orders(self, symbol: str = "") -> list[ManagedOrder]:
        orders = [o for o in self._orders.values() if o.status == OrderStatus.FILLED]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_fills(self, order_id: str = "") -> list[Fill]:
        if order_id:
            return [f for f in self._fills if f.order_id == order_id]
        return list(self._fills)

    def get_position(self, symbol: str) -> dict:
        net_qty = 0.0
        total_cost = 0.0
        total_fees = 0.0

        for order in self._orders.values():
            if order.symbol != symbol or order.status != OrderStatus.FILLED:
                continue
            for fill in order.fills:
                sign = 1 if order.side == "buy" else -1
                net_qty += fill.quantity * sign
                total_cost += fill.price * fill.quantity * sign
                total_fees += fill.fee

        avg_price = abs(total_cost / net_qty) if net_qty != 0 else 0.0
        return {
            "symbol": symbol, "net_quantity": net_qty,
            "average_price": avg_price, "total_cost": total_cost, "total_fees": total_fees,
        }

    def cancel_all(self, symbol: str = "") -> int:
        cancelled = 0
        for order in self._orders.values():
            if not order.is_active:
                continue
            if symbol and order.symbol != symbol:
                continue
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.utcnow()
            cancelled += 1
        return cancelled

    def get_stats(self) -> dict:
        total = len(self._orders)
        active = sum(1 for o in self._orders.values() if o.is_active)
        filled = sum(1 for o in self._orders.values() if o.status == OrderStatus.FILLED)
        cancelled = sum(1 for o in self._orders.values() if o.status == OrderStatus.CANCELLED)
        total_fees = sum(o.total_fees for o in self._orders.values())
        return {
            "total_orders": total, "active_orders": active,
            "filled_orders": filled, "cancelled_orders": cancelled,
            "total_fills": len(self._fills), "total_fees": total_fees,
        }
