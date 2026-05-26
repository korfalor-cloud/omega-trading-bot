"""Risk Manager — Pre-trade and post-trade risk checks."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from ..core.enums import Side
from ..core.events import Event, EventBus
from ..core.types import Fill, PortfolioState, Position, RiskAlert, RiskCheck, Signal

logger = logging.getLogger(__name__)


class RiskManager:
    """Comprehensive risk management system.

    Pre-trade checks:
    - Position size limits
    - Exposure limits
    - Drawdown limits
    - Daily loss limits
    - Leverage limits

    Circuit breakers:
    - Daily loss > 2% → reduce positions 50%
    - Daily loss > 5% → flatten everything
    - Drawdown > 10% → halt new entries
    - Emergency kill switch
    """

    def __init__(self, config: dict, event_bus: EventBus):
        self.event_bus = event_bus
        self.max_position_pct = config.get("max_position_pct", 0.05)
        self.max_gross_exposure = config.get("max_gross_exposure", 2.0)
        self.max_drawdown_pct = config.get("max_drawdown_pct", 0.15)
        self.daily_loss_limit_pct = config.get("daily_loss_limit_pct", 0.05)
        self.max_leverage = config.get("max_leverage", 3.0)
        self.circuit_breaker_dd = config.get("circuit_breaker_dd_pct", 0.10)

        self._daily_pnl = 0.0
        self._peak_equity = 0.0
        self._circuit_breaker_active = False
        self._emergency_stop = False

    async def pre_trade_check(
        self, signal: Signal, portfolio: PortfolioState, current_price: float
    ) -> RiskCheck:
        """Run all pre-trade risk checks."""
        warnings = []

        # Emergency stop
        if self._emergency_stop:
            return RiskCheck(approved=False, reason="Emergency stop active", risk_score=1.0)

        # Circuit breaker
        if self._circuit_breaker_active:
            return RiskCheck(approved=False, reason="Circuit breaker active", risk_score=1.0)

        # Drawdown check
        if portfolio.current_drawdown > self.max_drawdown_pct:
            await self._trigger_circuit_breaker("Max drawdown exceeded")
            return RiskCheck(
                approved=False,
                reason=f"Drawdown {portfolio.current_drawdown:.1%} exceeds limit {self.max_drawdown_pct:.1%}",
                risk_score=0.9,
            )

        # Daily loss check
        if portfolio.total_equity > 0:
            daily_loss_pct = -self._daily_pnl / portfolio.total_equity
            if daily_loss_pct > self.daily_loss_limit_pct:
                return RiskCheck(
                    approved=False,
                    reason=f"Daily loss {daily_loss_pct:.1%} exceeds limit {self.daily_loss_limit_pct:.1%}",
                    risk_score=0.8,
                )

        # Position size check
        position_value = portfolio.total_equity * self.max_position_pct * signal.confidence
        max_quantity = position_value / current_price if current_price > 0 else 0

        # Exposure check
        total_exposure = portfolio.gross_exposure + position_value
        if total_exposure > portfolio.total_equity * self.max_gross_exposure:
            warnings.append("Approaching gross exposure limit")
            max_quantity *= 0.5

        # Leverage check
        new_leverage = total_exposure / portfolio.total_equity if portfolio.total_equity > 0 else 0
        if new_leverage > self.max_leverage:
            return RiskCheck(
                approved=False,
                reason=f"Leverage {new_leverage:.1f}x exceeds limit {self.max_leverage:.1f}x",
                risk_score=0.7,
            )

        # Correlation check (simplified)
        if len(portfolio.positions) > 10:
            warnings.append("High number of open positions — check correlation")

        risk_score = min(1.0, portfolio.current_drawdown / self.max_drawdown_pct)

        return RiskCheck(
            approved=True,
            max_allowed_quantity=max_quantity,
            risk_score=risk_score,
            warnings=warnings,
        )

    async def post_trade_update(self, fill: Fill) -> None:
        """Update risk state after a trade."""
        # Track daily P&L
        if fill.side == Side.SELL:
            self._daily_pnl += fill.price * fill.quantity
        else:
            self._daily_pnl -= fill.price * fill.quantity

    async def update_portfolio(self, portfolio: PortfolioState) -> None:
        """Update risk state from portfolio."""
        if portfolio.total_equity > self._peak_equity:
            self._peak_equity = portfolio.total_equity

    async def _trigger_circuit_breaker(self, reason: str) -> None:
        """Trigger circuit breaker."""
        self._circuit_breaker_active = True
        alert = RiskAlert(
            level="critical",
            message=f"Circuit breaker triggered: {reason}",
            metric="drawdown",
            current_value=0,
            threshold=self.circuit_breaker_dd,
            action_taken="Circuit breaker activated — no new entries",
        )
        await self.event_bus.publish(Event.CIRCUIT_BREAKER_TRIGGERED, alert)
        logger.critical(f"CIRCUIT BREAKER: {reason}")

    async def emergency_stop(self) -> None:
        """Activate emergency stop — flatten everything."""
        self._emergency_stop = True
        logger.critical("EMERGENCY STOP ACTIVATED")

    async def reset(self) -> None:
        """Reset risk state (e.g., new day)."""
        self._daily_pnl = 0.0
        self._circuit_breaker_active = False

    @property
    def is_circuit_breaker_active(self) -> bool:
        return self._circuit_breaker_active

    @property
    def is_emergency_stop(self) -> bool:
        return self._emergency_stop
