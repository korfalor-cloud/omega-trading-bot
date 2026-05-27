"""TWAP (Time-Weighted Average Price) Execution Algorithm.

Splits a large order into equal-sized child orders executed at regular intervals
over a specified time window to minimize market impact.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from ...core.enums import OrderState, OrderType, Side
from ...core.types import Fill, Order

logger = logging.getLogger(__name__)


@dataclass
class TWAPSlice:
    """A single slice of a TWAP order."""
    parent_id: str
    index: int
    order: Order
    scheduled_time: datetime
    filled: bool = False
    fill_price: float = 0.0
    fill_quantity: float = 0.0


@dataclass
class TWAPState:
    """Current state of a TWAP execution."""
    parent_id: str
    symbol: str
    side: Side
    total_quantity: float
    num_slices: int
    interval_seconds: float
    slices: list[TWAPSlice] = field(default_factory=list)
    filled_quantity: float = 0.0
    vwap_executed: float = 0.0
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed: bool = False

    @property
    def completion_pct(self) -> float:
        if self.total_quantity == 0:
            return 0.0
        return self.filled_quantity / self.total_quantity

    @property
    def remaining_quantity(self) -> float:
        return self.total_quantity - self.filled_quantity


class TWAPAlgorithm:
    """TWAP execution algorithm.

    Splits parent order into N equal slices over a time window.
    Each slice is a market order submitted at regular intervals.

    Parameters:
        num_slices: Number of child orders (default 10)
        duration_minutes: Total execution window (default 30)
        randomize_pct: Randomize slice size by +/- pct (default 0.1)
        price_limit: Optional price limit — skip slice if price moves beyond
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.num_slices = config.get("num_slices", 10)
        self.duration_minutes = config.get("duration_minutes", 30)
        self.randomize_pct = config.get("randomize_pct", 0.1)
        self.price_limit = config.get("price_limit", None)
        self._active_taps: dict[str, TWAPState] = {}

    def create_twap_order(
        self,
        symbol: str,
        side: Side,
        total_quantity: float,
        start_time: datetime | None = None,
    ) -> TWAPState:
        """Create a new TWAP execution plan."""
        parent_id = str(uuid.uuid4())
        start = start_time or datetime.utcnow()
        interval = (self.duration_minutes * 60) / self.num_slices

        base_qty = total_quantity / self.num_slices
        slices = []

        for i in range(self.num_slices):
            scheduled = start + timedelta(seconds=i * interval)
            # Add randomization to avoid predictable patterns
            if self.randomize_pct > 0:
                import random
                jitter = base_qty * self.randomize_pct * (2 * random.random() - 1)
                slice_qty = max(0.001, base_qty + jitter)
            else:
                slice_qty = base_qty

            # Ensure last slice absorbs rounding
            if i == self.num_slices - 1:
                allocated = sum(s.order.quantity for s in slices)
                slice_qty = total_quantity - allocated

            order = Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=slice_qty,
            )

            slices.append(TWAPSlice(
                parent_id=parent_id,
                index=i,
                order=order,
                scheduled_time=scheduled,
            ))

        state = TWAPState(
            parent_id=parent_id,
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            num_slices=self.num_slices,
            interval_seconds=interval,
            slices=slices,
        )
        self._active_taps[parent_id] = state
        logger.info(
            f"TWAP created: {side.value} {total_quantity:.4f} {symbol} "
            f"in {self.num_slices} slices over {self.duration_minutes}min"
        )
        return state

    def get_pending_slices(self, parent_id: str, current_time: datetime | None = None) -> list[TWAPSlice]:
        """Get slices that are due for execution."""
        state = self._active_taps.get(parent_id)
        if not state:
            return []
        now = current_time or datetime.utcnow()
        return [
            s for s in state.slices
            if not s.filled and s.scheduled_time <= now
        ]

    def record_fill(self, parent_id: str, slice_index: int, fill_price: float, fill_qty: float) -> None:
        """Record a fill for a TWAP slice."""
        state = self._active_taps.get(parent_id)
        if not state or slice_index >= len(state.slices):
            return

        slice_obj = state.slices[slice_index]
        slice_obj.filled = True
        slice_obj.fill_price = fill_price
        slice_obj.fill_quantity = fill_qty

        # Update running VWAP
        old_notional = state.vwap_executed * state.filled_quantity
        new_notional = old_notional + fill_price * fill_qty
        state.filled_quantity += fill_qty
        if state.filled_quantity > 0:
            state.vwap_executed = new_notional / state.filled_quantity

        if state.filled_quantity >= state.total_quantity * 0.999:
            state.completed = True
            logger.info(
                f"TWAP completed: {state.symbol} VWAP={state.vwap_executed:.2f} "
                f"({state.filled_quantity:.4f}/{state.total_quantity:.4f})"
            )

    def get_slippage_bps(self, parent_id: str, arrival_price: float) -> float:
        """Calculate slippage vs arrival price in bps."""
        state = self._active_taps.get(parent_id)
        if not state or state.filled_quantity == 0 or arrival_price == 0:
            return 0.0
        if state.side == Side.BUY:
            return (state.vwap_executed - arrival_price) / arrival_price * 10000
        else:
            return (arrival_price - state.vwap_executed) / arrival_price * 10000

    def get_state(self, parent_id: str) -> Optional[TWAPState]:
        return self._active_taps.get(parent_id)

    def cancel(self, parent_id: str) -> None:
        """Cancel remaining slices."""
        state = self._active_taps.get(parent_id)
        if state:
            for s in state.slices:
                if not s.filled:
                    s.order.state = OrderState.CANCELLED
            state.completed = True
            logger.info(f"TWAP cancelled: {state.symbol} filled {state.completion_pct:.1%}")
