"""Multi-Objective Fitness Evaluation for Strategy Genomes."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FitnessResult:
    """Result of a fitness evaluation."""
    composite_fitness: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    stability: float
    total_trades: int
    total_return: float
    profit_factor: float
    avg_trade_pnl: float
    metadata: dict


class FitnessEvaluator:
    """Evaluates strategy performance using multiple objectives.

    Combines Sharpe, Sortino, Max Drawdown, Win Rate, and Stability
    into a single composite fitness score with configurable weights.
    """

    def __init__(self, config: dict):
        self.sharpe_weight = config.get("sharpe_weight", 0.35)
        self.sortino_weight = config.get("sortino_weight", 0.25)
        self.max_dd_weight = config.get("max_dd_weight", 0.20)
        self.win_rate_weight = config.get("win_rate_weight", 0.10)
        self.stability_weight = config.get("stability_weight", 0.10)
        self.min_trades = config.get("min_trades", 30)
        self.risk_free_rate = config.get("risk_free_rate", 0.0)

    def evaluate(
        self,
        equity_curve: list[float],
        trade_returns: list[float],
        timestamps: Optional[list] = None,
    ) -> FitnessResult:
        """Evaluate fitness from equity curve and trade returns."""
        if len(trade_returns) < self.min_trades:
            return FitnessResult(
                composite_fitness=-1.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=1.0,
                win_rate=0.0,
                stability=0.0,
                total_trades=len(trade_returns),
                total_return=0.0,
                profit_factor=0.0,
                avg_trade_pnl=0.0,
                metadata={"reason": "insufficient_trades"},
            )

        sharpe = self._sharpe_ratio(trade_returns)
        sortino = self._sortino_ratio(trade_returns)
        max_dd = self._max_drawdown(equity_curve)
        win_rate = self._win_rate(trade_returns)
        stability = self._stability(equity_curve)
        total_return = self._total_return(equity_curve)
        profit_factor = self._profit_factor(trade_returns)
        avg_pnl = sum(trade_returns) / len(trade_returns)

        # Normalize metrics to [0, 1] range
        sharpe_norm = self._normalize_sharpe(sharpe)
        sortino_norm = self._normalize_sortino(sortino)
        dd_norm = max(0.0, 1.0 - max_dd)  # Lower DD is better
        wr_norm = win_rate
        stab_norm = stability

        # Composite fitness
        composite = (
            self.sharpe_weight * sharpe_norm
            + self.sortino_weight * sortino_norm
            + self.max_dd_weight * dd_norm
            + self.win_rate_weight * wr_norm
            + self.stability_weight * stab_norm
        )

        # Penalty for too few trades
        if len(trade_returns) < self.min_trades * 2:
            composite *= 0.8

        # Penalty for negative total return
        if total_return < 0:
            composite *= 0.5

        return FitnessResult(
            composite_fitness=composite,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            stability=stability,
            total_trades=len(trade_returns),
            total_return=total_return,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_pnl,
            metadata={},
        )

    def _sharpe_ratio(self, returns: list[float]) -> float:
        """Annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(variance) if variance > 0 else 1e-10
        excess = mean - self.risk_free_rate / 252  # Daily risk-free
        return (excess / std) * math.sqrt(252)

    def _sortino_ratio(self, returns: list[float]) -> float:
        """Annualized Sortino ratio (only penalizes downside volatility)."""
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        downside_returns = [r for r in returns if r < 0]
        if not downside_returns:
            return 10.0  # Cap at 10
        downside_var = sum(r ** 2 for r in downside_returns) / len(downside_returns)
        downside_std = math.sqrt(downside_var) if downside_var > 0 else 1e-10
        excess = mean - self.risk_free_rate / 252
        return (excess / downside_std) * math.sqrt(252)

    def _max_drawdown(self, equity_curve: list[float]) -> float:
        """Maximum drawdown from peak."""
        if not equity_curve:
            return 1.0
        peak = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd

    def _win_rate(self, returns: list[float]) -> float:
        """Fraction of positive trades."""
        if not returns:
            return 0.0
        wins = sum(1 for r in returns if r > 0)
        return wins / len(returns)

    def _stability(self, equity_curve: list[float]) -> float:
        """Stability of equity curve (R² of linear fit)."""
        if len(equity_curve) < 10:
            return 0.0

        n = len(equity_curve)
        x = list(range(n))
        x_mean = (n - 1) / 2
        y_mean = sum(equity_curve) / n

        ss_xy = sum((x[i] - x_mean) * (equity_curve[i] - y_mean) for i in range(n))
        ss_xx = sum((x[i] - x_mean) ** 2 for i in range(n))
        ss_yy = sum((equity_curve[i] - y_mean) ** 2 for i in range(n))

        if ss_xx == 0 or ss_yy == 0:
            return 0.0

        r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)
        return min(1.0, r_squared)

    def _total_return(self, equity_curve: list[float]) -> float:
        """Total return from start to end."""
        if len(equity_curve) < 2:
            return 0.0
        return (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

    def _profit_factor(self, returns: list[float]) -> float:
        """Gross profit / gross loss."""
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        if gross_loss == 0:
            return 10.0 if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def _normalize_sharpe(self, sharpe: float) -> float:
        """Normalize Sharpe to [0, 1]."""
        # Sharpe of 2.0+ is excellent, map to ~1.0
        return max(0.0, min(1.0, sharpe / 3.0 + 0.5))

    def _normalize_sortino(self, sortino: float) -> float:
        """Normalize Sortino to [0, 1]."""
        return max(0.0, min(1.0, sortino / 4.0 + 0.5))
