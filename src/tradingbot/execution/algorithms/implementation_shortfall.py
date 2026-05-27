"""Implementation Shortfall (IS) Execution Algorithm.

Minimizes the difference between the decision price and the average execution price.
Balances market impact cost (fast execution) vs timing risk (slow execution).

Based on the Almgren-Chriss model: optimal execution trajectory that minimizes
expected shortfall + risk penalty.
"""
from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from ...core.enums import OrderState, OrderType, Side
from ...core.types import Order

logger = logging.getLogger(__name__)


@dataclass
class ISTrajectory:
    """Optimal execution trajectory slice."""
    index: int
    time_fraction: float  # 0 to 1 of total horizon
    remaining_quantity: float  # Optimal remaining qty at this point
    trade_quantity: float  # Quantity to trade in this interval
    scheduled_time: datetime


@dataclass
class ISState:
    """State of an Implementation Shortfall execution."""
    parent_id: str
    symbol: str
    side: Side
    total_quantity: float
    decision_price: float
    trajectory: list[ISTrajectory] = field(default_factory=list)
    filled_quantity: float = 0.0
    total_cost: float = 0.0
    shortfall: float = 0.0
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed: bool = False

    @property
    def arrival_price(self) -> float:
        return self.decision_price

    @property
    def avg_execution_price(self) -> float:
        if self.filled_quantity == 0:
            return 0.0
        return self.total_cost / self.filled_quantity

    @property
    def implementation_shortfall(self) -> float:
        """IS in basis points."""
        if self.decision_price == 0 or self.filled_quantity == 0:
            return 0.0
        if self.side == Side.BUY:
            return (self.avg_execution_price - self.decision_price) / self.decision_price * 10000
        else:
            return (self.decision_price - self.avg_execution_price) / self.decision_price * 10000


class ImplementationShortfallAlgorithm:
    """Implementation Shortfall (Almgren-Chriss) execution algorithm.

    Finds the optimal trade schedule that minimizes:
        E[cost] + lambda * Var[cost]

    where lambda is the risk aversion parameter.

    Parameters:
        risk_aversion: Risk aversion parameter (default 1e-6). Higher = more aggressive.
        num_slices: Number of execution intervals (default 10)
        duration_minutes: Total execution horizon (default 60)
        volatility: Asset volatility estimate (default 0.02 = 2%)
        eta: Temporary impact coefficient (default 2.5e-7)
        gamma: Permanent impact coefficient (default 2.5e-7)
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.risk_aversion = config.get("risk_aversion", 1e-6)
        self.num_slices = config.get("num_slices", 10)
        self.duration_minutes = config.get("duration_minutes", 60)
        self.volatility = config.get("volatility", 0.02)
        self.eta = config.get("eta", 2.5e-7)
        self.gamma = config.get("gamma", 2.5e-7)
        self._active_executions: dict[str, ISState] = {}

    def compute_optimal_trajectory(
        self,
        total_quantity: float,
        start_time: datetime,
    ) -> list[ISTrajectory]:
        """Compute Almgren-Chriss optimal execution trajectory.

        The optimal remaining holdings follow an exponential decay:
            x(t) = X * sinh(kappa * (T - t)) / sinh(kappa * T)

        where kappa depends on risk aversion, volatility, and market impact.
        """
        T = self.duration_minutes * 60  # total seconds
        n = self.num_slices
        dt = T / n

        # Almgren-Chriss parameters
        sigma_sq = self.volatility ** 2
        kappa = math.sqrt(self.risk_aversion * sigma_sq / self.eta) if self.eta > 0 else 0.01
        kappa = min(kappa, 50)  # Numerical stability

        trajectories = []
        for i in range(n + 1):
            t = i * dt
            tau = T - t

            if kappa > 1e-10 and tau > 0:
                # Optimal remaining quantity
                if kappa * T < 50:
                    remaining = total_quantity * math.sinh(kappa * tau) / math.sinh(kappa * T)
                else:
                    remaining = total_quantity * math.exp(-kappa * t)
            else:
                # Linear decay for very low risk aversion
                remaining = total_quantity * (1 - t / T) if T > 0 else 0

            remaining = max(0, remaining)
            scheduled = start_time + timedelta(seconds=t)

            if i < n:
                next_remaining = 0
                t_next = (i + 1) * dt
                tau_next = T - t_next
                if kappa > 1e-10 and tau_next > 0:
                    if kappa * T < 50:
                        next_remaining = total_quantity * math.sinh(kappa * tau_next) / math.sinh(kappa * T)
                    else:
                        next_remaining = total_quantity * math.exp(-kappa * t_next)
                else:
                    next_remaining = total_quantity * (1 - t_next / T) if T > 0 else 0
                next_remaining = max(0, next_remaining)

                trade_qty = remaining - next_remaining
            else:
                trade_qty = remaining

            trajectories.append(ISTrajectory(
                index=i,
                time_fraction=t / T if T > 0 else 1.0,
                remaining_quantity=remaining,
                trade_quantity=max(0, trade_qty),
                scheduled_time=scheduled,
            ))

        return trajectories

    def create_is_order(
        self,
        symbol: str,
        side: Side,
        total_quantity: float,
        decision_price: float,
        start_time: datetime | None = None,
    ) -> ISState:
        """Create an IS execution plan."""
        parent_id = str(uuid.uuid4())
        start = start_time or datetime.utcnow()

        trajectory = self.compute_optimal_trajectory(total_quantity, start)

        state = ISState(
            parent_id=parent_id,
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            decision_price=decision_price,
            trajectory=trajectory,
        )
        self._active_executions[parent_id] = state
        logger.info(
            f"IS created: {side.value} {total_quantity:.4f} {symbol} "
            f"decision_price={decision_price:.2f} slices={len(trajectory)}"
        )
        return state

    def get_pending_slices(self, parent_id: str, current_time: datetime | None = None) -> list[ISTrajectory]:
        """Get slices due for execution."""
        state = self._active_executions.get(parent_id)
        if not state:
            return []
        now = current_time or datetime.utcnow()
        return [
            t for t in state.trajectory
            if t.scheduled_time <= now and t.trade_quantity > 0
        ]

    def record_fill(self, parent_id: str, slice_index: int, fill_price: float, fill_qty: float) -> None:
        """Record a fill."""
        state = self._active_executions.get(parent_id)
        if not state:
            return

        old_cost = state.total_cost
        state.total_cost += fill_price * fill_qty
        state.filled_quantity += fill_qty

        if state.filled_quantity >= state.total_quantity * 0.999:
            state.completed = True
            state.shortfall = state.implementation_shortfall
            logger.info(
                f"IS completed: {state.symbol} IS={state.shortfall:.1f}bps "
                f"avg_price={state.avg_execution_price:.2f}"
            )

    def get_state(self, parent_id: str) -> Optional[ISState]:
        return self._active_executions.get(parent_id)

    def cancel(self, parent_id: str) -> None:
        """Cancel remaining execution."""
        state = self._active_executions.get(parent_id)
        if state:
            state.completed = True
            for t in state.trajectory:
                t.trade_quantity = 0
