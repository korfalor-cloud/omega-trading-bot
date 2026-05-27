"""VWAP (Volume-Weighted Average Price) Execution Algorithm.

Splits orders based on historical volume distribution to execute at or near
the market VWAP. Participates more heavily during high-volume periods.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from ...core.enums import OrderState, OrderType, Side
from ...core.types import Order

logger = logging.getLogger(__name__)


@dataclass
class VWAPBucket:
    """A time bucket with its target participation."""
    index: int
    start_minute: int
    end_minute: int
    volume_pct: float  # Fraction of total daily volume
    target_quantity: float
    filled_quantity: float = 0.0
    vwap_price: float = 0.0
    completed: bool = False


@dataclass
class VWAPState:
    """State of a VWAP execution."""
    parent_id: str
    symbol: str
    side: Side
    total_quantity: float
    buckets: list[VWAPBucket] = field(default_factory=list)
    filled_quantity: float = 0.0
    vwap_executed: float = 0.0
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed: bool = False

    @property
    def completion_pct(self) -> float:
        if self.total_quantity == 0:
            return 0.0
        return self.filled_quantity / self.total_quantity


class VWAPAlgorithm:
    """VWAP execution algorithm.

    Distributes order quantity across time buckets proportional to
    historical intraday volume profile.

    Parameters:
        num_buckets: Number of time buckets (default 24 — one per 10min in 4h)
        participation_rate: Max participation rate per bucket (default 0.1)
        volume_profile: Optional historical volume weights; uses uniform if None
        min_quantity: Minimum slice size (default 0.001)
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.num_buckets = config.get("num_buckets", 24)
        self.participation_rate = config.get("participation_rate", 0.1)
        self.min_quantity = config.get("min_quantity", 0.001)
        self._volume_profile: Optional[np.ndarray] = config.get("volume_profile")
        self._active_executions: dict[str, VWAPState] = {}

    def set_volume_profile(self, profile: np.ndarray) -> None:
        """Set intraday volume distribution (will be normalized)."""
        total = np.sum(profile)
        if total > 0:
            self._volume_profile = profile / total
        else:
            self._volume_profile = np.ones(len(profile)) / len(profile)

    def create_vwap_order(
        self,
        symbol: str,
        side: Side,
        total_quantity: float,
        market_volume_estimate: float = 0.0,
    ) -> VWAPState:
        """Create a VWAP execution plan."""
        parent_id = str(uuid.uuid4())

        # Determine volume weights per bucket
        if self._volume_profile is not None and len(self._volume_profile) == self.num_buckets:
            weights = self._volume_profile
        else:
            # U-shaped intraday volume profile (higher at open/close)
            weights = self._default_volume_profile(self.num_buckets)

        # Create buckets
        bucket_duration = 240 // self.num_buckets  # minutes per bucket (4h window)
        buckets = []

        for i in range(self.num_buckets):
            target_qty = total_quantity * weights[i]

            # Cap at participation rate
            if market_volume_estimate > 0:
                max_bucket_qty = market_volume_estimate * weights[i] * self.participation_rate
                target_qty = min(target_qty, max_bucket_qty)

            target_qty = max(target_qty, self.min_quantity) if target_qty > 0 else 0

            buckets.append(VWAPBucket(
                index=i,
                start_minute=i * bucket_duration,
                end_minute=(i + 1) * bucket_duration,
                volume_pct=float(weights[i]),
                target_quantity=target_qty,
            ))

        # Adjust last bucket to absorb rounding
        allocated = sum(b.target_quantity for b in buckets)
        if buckets:
            buckets[-1].target_quantity += total_quantity - allocated

        state = VWAPState(
            parent_id=parent_id,
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            buckets=buckets,
        )
        self._active_executions[parent_id] = state
        logger.info(
            f"VWAP created: {side.value} {total_quantity:.4f} {symbol} "
            f"across {self.num_buckets} buckets"
        )
        return state

    def get_current_bucket(self, parent_id: str, minute_of_day: int) -> Optional[VWAPBucket]:
        """Get the bucket for the current time."""
        state = self._active_executions.get(parent_id)
        if not state:
            return None
        for bucket in state.buckets:
            if bucket.start_minute <= minute_of_day < bucket.end_minute and not bucket.completed:
                return bucket
        return None

    def record_fill(self, parent_id: str, bucket_index: int, fill_price: float, fill_qty: float) -> None:
        """Record a fill for a VWAP bucket."""
        state = self._active_executions.get(parent_id)
        if not state or bucket_index >= len(state.buckets):
            return

        bucket = state.buckets[bucket_index]
        bucket.filled_quantity += fill_qty

        # Update bucket VWAP
        old_notional = bucket.vwap_price * (bucket.filled_quantity - fill_qty)
        new_notional = old_notional + fill_price * fill_qty
        if bucket.filled_quantity > 0:
            bucket.vwap_price = new_notional / bucket.filled_quantity

        if bucket.filled_quantity >= bucket.target_quantity * 0.99:
            bucket.completed = True

        # Update overall state
        old_total = state.vwap_executed * state.filled_quantity
        state.filled_quantity += fill_qty
        if state.filled_quantity > 0:
            state.vwap_executed = (old_total + fill_price * fill_qty) / state.filled_quantity

        if state.filled_quantity >= state.total_quantity * 0.999:
            state.completed = True
            logger.info(
                f"VWAP completed: {state.symbol} VWAP={state.vwap_executed:.2f}"
            )

    def get_slippage_bps(self, parent_id: str, market_vwap: float) -> float:
        """Calculate slippage vs market VWAP in bps."""
        state = self._active_executions.get(parent_id)
        if not state or state.filled_quantity == 0 or market_vwap == 0:
            return 0.0
        if state.side == Side.BUY:
            return (state.vwap_executed - market_vwap) / market_vwap * 10000
        else:
            return (market_vwap - state.vwap_executed) / market_vwap * 10000

    def get_state(self, parent_id: str) -> Optional[VWAPState]:
        return self._active_executions.get(parent_id)

    def cancel(self, parent_id: str) -> None:
        """Cancel remaining buckets."""
        state = self._active_executions.get(parent_id)
        if state:
            for b in state.buckets:
                b.completed = True
            state.completed = True

    @staticmethod
    def _default_volume_profile(n: int) -> np.ndarray:
        """Generate a U-shaped intraday volume profile."""
        x = np.linspace(0, 1, n)
        # U-shape: higher volume at start and end
        profile = 2.0 * (x - 0.5) ** 2 + 0.5
        return profile / profile.sum()
