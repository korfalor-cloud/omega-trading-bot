"""Paper Exchange — Simulated execution for testing strategies."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

from ..core.enums import OrderState, OrderType, Side
from ..core.types import Fill, Order, Position
from ..core.interfaces import ExecutionBackend

logger = logging.getLogger(__name__)


class PaperExecutionBackend(ExecutionBackend):
    """Simulated execution backend for paper trading.

    Simulates realistic fills with configurable:
    - Slippage (fixed bps or volume-based)
    - Commission (fixed bps or tiered)
    - Latency (simulated delay)
    """

    def __init__(self, config: dict):
        self.slippage_bps = config.get("slippage_bps", 5.0)
        self.commission_bps = config.get("commission_bps", 10.0)
        self.latency_ms = config.get("latency_ms", 50)
        self.initial_balance = config.get("initial_balance", 100_000.0)

        self._balances: dict[str, float] = {"USDT": self.initial_balance}
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._fill_log: list[Fill] = []
        self._current_prices: dict[str, float] = {}

    def update_price(self, symbol: str, price: float) -> None:
        """Update current price for a symbol."""
        self._current_prices[symbol] = price

    async def execute(self, order: Order) -> Order:
        """Simulate order execution."""
        # Simulate latency
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)

        price = self._current_prices.get(order.symbol)
        if price is None:
            order.state = OrderState.REJECTED
            order.metadata["reject_reason"] = f"No price data for {order.symbol}"
            return order

        # Apply slippage
        slippage_mult = self.slippage_bps / 10000
        if order.side == Side.BUY:
            fill_price = price * (1 + slippage_mult)
        else:
            fill_price = price * (1 - slippage_mult)

        # Check balance
        if order.side == Side.BUY:
            cost = order.quantity * fill_price * (1 + self.commission_bps / 10000)
            if self._balances.get("USDT", 0) < cost:
                order.state = OrderState.REJECTED
                order.metadata["reject_reason"] = f"Insufficient balance: need {cost}, have {self._balances.get('USDT', 0)}"
                return order
            self._balances["USDT"] -= cost
        else:
            # Check if we have the position to sell
            pos_key = f"{order.symbol}:{order.strategy_id}"
            pos = self._positions.get(pos_key)
            if pos is None or pos.quantity < order.quantity:
                order.state = OrderState.REJECTED
                order.metadata["reject_reason"] = "Insufficient position"
                return order
            revenue = order.quantity * fill_price * (1 - self.commission_bps / 10000)
            self._balances["USDT"] += revenue

        # Apply commission
        commission = order.quantity * fill_price * self.commission_bps / 10000

        # Create fill
        fill = Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            price=fill_price,
            quantity=order.quantity,
            commission=commission,
            exchange="paper",
            timestamp=datetime.utcnow(),
        )
        self._fill_log.append(fill)

        # Update order
        order.state = OrderState.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.commission = commission
        self._orders[order.id] = order

        # Update position
        self._update_position(fill, order.strategy_id)

        logger.info(
            f"Paper fill: {order.side.value} {order.quantity} {order.symbol} @ {fill_price:.2f} "
            f"(slip={self.slippage_bps}bps, comm={self.commission_bps}bps)"
        )
        return order

    async def cancel(self, order: Order) -> Order:
        order.state = OrderState.CANCELLED
        return order

    async def get_positions(self, strategy_id: str) -> list[Position]:
        return [p for p in self._positions.values() if p.strategy_id == strategy_id and p.quantity > 0]

    async def get_balance(self) -> dict[str, float]:
        return dict(self._balances)

    async def get_order_status(self, order_id: str) -> Order:
        return self._orders.get(order_id, Order(id=order_id, state=OrderState.EXPIRED))

    def _update_position(self, fill: Fill, strategy_id: str) -> None:
        """Update position after a fill."""
        pos_key = f"{fill.symbol}:{strategy_id}"

        if pos_key not in self._positions:
            self._positions[pos_key] = Position(
                symbol=fill.symbol,
                strategy_id=strategy_id,
                side=fill.side,
                quantity=0,
                avg_entry_price=0,
                exchange="paper",
            )

        pos = self._positions[pos_key]

        if fill.side == Side.BUY:
            if pos.side == Side.BUY:
                # Adding to long
                total_cost = pos.avg_entry_price * pos.quantity + fill.price * fill.quantity
                pos.quantity += fill.quantity
                pos.avg_entry_price = total_cost / pos.quantity if pos.quantity > 0 else 0
            else:
                # Closing short
                pos.quantity -= fill.quantity
                if pos.quantity <= 0:
                    pos.realized_pnl += (pos.avg_entry_price - fill.price) * fill.quantity
                    if pos.quantity < 0:
                        pos.side = Side.BUY
                        pos.quantity = abs(pos.quantity)
                        pos.avg_entry_price = fill.price
                    else:
                        pos.quantity = 0
        else:
            if pos.side == Side.SELL:
                # Adding to short
                total_cost = pos.avg_entry_price * pos.quantity + fill.price * fill.quantity
                pos.quantity += fill.quantity
                pos.avg_entry_price = total_cost / pos.quantity if pos.quantity > 0 else 0
            else:
                # Closing long
                pos.quantity -= fill.quantity
                if pos.quantity <= 0:
                    pos.realized_pnl += (fill.price - pos.avg_entry_price) * fill.quantity
                    if pos.quantity < 0:
                        pos.side = Side.SELL
                        pos.quantity = abs(pos.quantity)
                        pos.avg_entry_price = fill.price
                    else:
                        pos.quantity = 0

        pos.current_price = fill.price
        pos.update_price(fill.price)
