"""Portfolio Risk Dashboard.

Implements:
- Real-time portfolio risk metrics
- Risk decomposition by strategy/symbol
- P&L attribution
- Performance visualization data
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DashboardState:
    """Portfolio dashboard state."""
    total_equity: float = 0.0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    n_positions: int = 0
    total_exposure: float = 0.0
    leverage: float = 0.0
    margin_utilization: float = 0.0
    strategy_breakdown: dict = field(default_factory=dict)
    symbol_breakdown: dict = field(default_factory=dict)
    risk_metrics: dict = field(default_factory=dict)


class RiskDashboard:
    """Portfolio risk dashboard."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self._equity_history: list[tuple[datetime, float]] = []
        self._pnl_history: list[tuple[datetime, float]] = []
        self._strategy_pnl: dict[str, float] = {}
        self._symbol_pnl: dict[str, float] = {}

    def update_equity(self, equity: float, timestamp: datetime = None) -> None:
        self._equity_history.append((timestamp or datetime.utcnow(), equity))

    def record_pnl(self, pnl: float, strategy_id: str = "", symbol: str = "", timestamp: datetime = None) -> None:
        self._pnl_history.append((timestamp or datetime.utcnow(), pnl))
        if strategy_id:
            self._strategy_pnl[strategy_id] = self._strategy_pnl.get(strategy_id, 0) + pnl
        if symbol:
            self._symbol_pnl[symbol] = self._symbol_pnl.get(symbol, 0) + pnl

    def get_state(
        self,
        positions: dict = None,
        equity: float = 0.0,
        daily_pnl: float = 0.0,
    ) -> DashboardState:
        """Get current dashboard state."""
        positions = positions or {}

        total_exposure = sum(abs(p.get("notional", 0)) for p in positions.values())
        leverage = total_exposure / equity if equity > 0 else 0

        # Max drawdown from equity curve
        max_dd = 0.0
        if self._equity_history:
            equities = np.array([e for _, e in self._equity_history])
            peak = np.maximum.accumulate(equities)
            dd = (peak - equities) / peak
            max_dd = float(np.max(dd))

        # Sharpe from pnl history
        sharpe = 0.0
        if len(self._pnl_history) > 10:
            pnls = np.array([p for _, p in self._pnl_history])
            if np.std(pnls) > 0:
                sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(365))

        # Win rate
        win_rate = 0.0
        if self._pnl_history:
            pnls = np.array([p for _, p in self._pnl_history])
            win_rate = float(np.sum(pnls > 0) / len(pnls))

        total_pnl = sum(p for _, p in self._pnl_history)

        return DashboardState(
            total_equity=equity,
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            n_positions=len(positions),
            total_exposure=total_exposure,
            leverage=leverage,
            strategy_breakdown=dict(self._strategy_pnl),
            symbol_breakdown=dict(self._symbol_pnl),
        )

    def get_equity_curve(self) -> list[tuple[datetime, float]]:
        return list(self._equity_history)

    def get_pnl_history(self) -> list[tuple[datetime, float]]:
        return list(self._pnl_history)

    def get_strategy_ranking(self) -> list[tuple[str, float]]:
        return sorted(self._strategy_pnl.items(), key=lambda x: x[1], reverse=True)

    def get_symbol_ranking(self) -> list[tuple[str, float]]:
        return sorted(self._symbol_pnl.items(), key=lambda x: x[1], reverse=True)
