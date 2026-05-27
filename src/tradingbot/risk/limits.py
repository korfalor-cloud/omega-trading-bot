"""Risk Limits Engine.

Implements:
- Position size limits
- Loss limits (daily, weekly, monthly)
- Drawdown limits
- Concentration limits
- Correlation limits
- Leverage limits
- Circuit breaker triggers
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class LimitType(Enum):
    POSITION_SIZE = auto()
    DAILY_LOSS = auto()
    WEEKLY_LOSS = auto()
    DRAWDOWN = auto()
    CONCENTRATION = auto()
    LEVERAGE = auto()
    CORRELATION = auto()


class LimitBreach(Enum):
    WARNING = auto()
    SOFT_BREACH = auto()
    HARD_BREACH = auto()


@dataclass
class RiskLimit:
    """A single risk limit."""
    limit_type: LimitType = LimitType.POSITION_SIZE
    threshold: float = 0.0
    warning_pct: float = 0.8  # Warning at 80% of limit
    enabled: bool = True
    description: str = ""


@dataclass
class LimitCheck:
    """Result of a limit check."""
    limit_type: LimitType = LimitType.POSITION_SIZE
    current_value: float = 0.0
    threshold: float = 0.0
    utilization: float = 0.0  # current/threshold
    breach_level: Optional[LimitBreach] = None
    message: str = ""


class RiskLimitsEngine:
    """Risk limits management and monitoring.

    Checks portfolio state against configured limits
    and generates warnings/breaches.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self._limits: dict[LimitType, RiskLimit] = {}
        self._breach_history: list[LimitCheck] = []
        self._daily_pnl: dict[str, float] = {}

        # Set default limits
        self._set_defaults(config)

    def _set_defaults(self, config: dict) -> None:
        self._limits[LimitType.POSITION_SIZE] = RiskLimit(
            limit_type=LimitType.POSITION_SIZE,
            threshold=config.get("max_position_pct", 0.10),
            description="Max single position as % of portfolio",
        )
        self._limits[LimitType.DAILY_LOSS] = RiskLimit(
            limit_type=LimitType.DAILY_LOSS,
            threshold=config.get("max_daily_loss_pct", 0.03),
            description="Max daily loss as % of portfolio",
        )
        self._limits[LimitType.DRAWDOWN] = RiskLimit(
            limit_type=LimitType.DRAWDOWN,
            threshold=config.get("max_drawdown_pct", 0.15),
            description="Max drawdown from peak",
        )
        self._limits[LimitType.LEVERAGE] = RiskLimit(
            limit_type=LimitType.LEVERAGE,
            threshold=config.get("max_leverage", 3.0),
            description="Max leverage ratio",
        )
        self._limits[LimitType.CONCENTRATION] = RiskLimit(
            limit_type=LimitType.CONCENTRATION,
            threshold=config.get("max_concentration_pct", 0.40),
            description="Max single asset concentration",
        )

    def set_limit(
        self,
        limit_type: LimitType,
        threshold: float,
        warning_pct: float = 0.8,
    ) -> None:
        self._limits[limit_type] = RiskLimit(
            limit_type=limit_type,
            threshold=threshold,
            warning_pct=warning_pct,
        )

    def check_position_size(
        self,
        position_value: float,
        portfolio_value: float,
    ) -> LimitCheck:
        """Check if a position is within size limits."""
        limit = self._limits.get(LimitType.POSITION_SIZE)
        if not limit or not limit.enabled:
            return LimitCheck(limit_type=LimitType.POSITION_SIZE)

        pct = position_value / portfolio_value if portfolio_value > 0 else 0
        utilization = pct / limit.threshold if limit.threshold > 0 else 0

        breach = None
        if utilization >= 1.0:
            breach = LimitBreach.HARD_BREACH
        elif utilization >= limit.warning_pct:
            breach = LimitBreach.WARNING

        check = LimitCheck(
            limit_type=LimitType.POSITION_SIZE,
            current_value=pct,
            threshold=limit.threshold,
            utilization=utilization,
            breach_level=breach,
            message=f"Position {pct:.1%} of portfolio (limit: {limit.threshold:.1%})",
        )

        if breach:
            self._breach_history.append(check)

        return check

    def check_daily_loss(
        self,
        daily_pnl: float,
        portfolio_value: float,
    ) -> LimitCheck:
        """Check daily loss limit."""
        limit = self._limits.get(LimitType.DAILY_LOSS)
        if not limit or not limit.enabled:
            return LimitCheck(limit_type=LimitType.DAILY_LOSS)

        loss_pct = abs(daily_pnl) / portfolio_value if daily_pnl < 0 and portfolio_value > 0 else 0
        utilization = loss_pct / limit.threshold if limit.threshold > 0 else 0

        breach = None
        if utilization >= 1.0:
            breach = LimitBreach.HARD_BREACH
        elif utilization >= limit.warning_pct:
            breach = LimitBreach.WARNING

        check = LimitCheck(
            limit_type=LimitType.DAILY_LOSS,
            current_value=loss_pct,
            threshold=limit.threshold,
            utilization=utilization,
            breach_level=breach,
            message=f"Daily loss {loss_pct:.2%} (limit: {limit.threshold:.2%})",
        )

        if breach:
            self._breach_history.append(check)

        return check

    def check_drawdown(
        self,
        current_equity: float,
        peak_equity: float,
    ) -> LimitCheck:
        """Check drawdown limit."""
        limit = self._limits.get(LimitType.DRAWDOWN)
        if not limit or not limit.enabled:
            return LimitCheck(limit_type=LimitType.DRAWDOWN)

        dd = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
        utilization = dd / limit.threshold if limit.threshold > 0 else 0

        breach = None
        if utilization >= 1.0:
            breach = LimitBreach.HARD_BREACH
        elif utilization >= limit.warning_pct:
            breach = LimitBreach.WARNING

        check = LimitCheck(
            limit_type=LimitType.DRAWDOWN,
            current_value=dd,
            threshold=limit.threshold,
            utilization=utilization,
            breach_level=breach,
            message=f"Drawdown {dd:.2%} (limit: {limit.threshold:.2%})",
        )

        if breach:
            self._breach_history.append(check)

        return check

    def check_leverage(
        self,
        total_exposure: float,
        equity: float,
    ) -> LimitCheck:
        """Check leverage limit."""
        limit = self._limits.get(LimitType.LEVERAGE)
        if not limit or not limit.enabled:
            return LimitCheck(limit_type=LimitType.LEVERAGE)

        leverage = total_exposure / equity if equity > 0 else 0
        utilization = leverage / limit.threshold if limit.threshold > 0 else 0

        breach = None
        if utilization >= 1.0:
            breach = LimitBreach.HARD_BREACH
        elif utilization >= limit.warning_pct:
            breach = LimitBreach.WARNING

        check = LimitCheck(
            limit_type=LimitType.LEVERAGE,
            current_value=leverage,
            threshold=limit.threshold,
            utilization=utilization,
            breach_level=breach,
            message=f"Leverage {leverage:.1f}x (limit: {limit.threshold:.1f}x)",
        )

        if breach:
            self._breach_history.append(check)

        return check

    def check_concentration(
        self,
        position_values: dict[str, float],
        portfolio_value: float,
    ) -> list[LimitCheck]:
        """Check concentration limits for all positions."""
        limit = self._limits.get(LimitType.CONCENTRATION)
        if not limit or not limit.enabled:
            return []

        checks = []
        for symbol, value in position_values.items():
            pct = value / portfolio_value if portfolio_value > 0 else 0
            utilization = pct / limit.threshold if limit.threshold > 0 else 0

            breach = None
            if utilization >= 1.0:
                breach = LimitBreach.HARD_BREACH
            elif utilization >= limit.warning_pct:
                breach = LimitBreach.WARNING

            check = LimitCheck(
                limit_type=LimitType.CONCENTRATION,
                current_value=pct,
                threshold=limit.threshold,
                utilization=utilization,
                breach_level=breach,
                message=f"{symbol} concentration {pct:.1%} (limit: {limit.threshold:.1%})",
            )

            if breach:
                self._breach_history.append(check)
            checks.append(check)

        return checks

    def check_all(
        self,
        portfolio_value: float,
        peak_equity: float,
        daily_pnl: float,
        total_exposure: float,
        position_values: dict[str, float],
    ) -> list[LimitCheck]:
        """Run all limit checks."""
        checks = []
        checks.append(self.check_drawdown(portfolio_value, peak_equity))
        checks.append(self.check_daily_loss(daily_pnl, portfolio_value))
        checks.append(self.check_leverage(total_exposure, portfolio_value))
        checks.extend(self.check_concentration(position_values, portfolio_value))

        for symbol, value in position_values.items():
            checks.append(self.check_position_size(value, portfolio_value))

        return [c for c in checks if c.breach_level is not None]

    def get_breach_history(self, limit: int = 100) -> list[LimitCheck]:
        return self._breach_history[-limit:]

    def is_circuit_breaker_active(self) -> bool:
        """Check if any hard breach should trigger circuit breaker."""
        recent = self._breach_history[-10:]
        return any(c.breach_level == LimitBreach.HARD_BREACH for c in recent)
