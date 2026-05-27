"""Performance Reporting and Analytics.

Generates comprehensive reports on:
- Strategy performance
- Portfolio analytics
- Risk metrics
- Trade journaling
- Period summaries
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from ..core.types import Fill, PortfolioState

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """A completed trade for journaling."""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    fees: float = 0.0
    strategy_id: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def duration(self) -> timedelta:
        return self.exit_time - self.entry_time

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade_duration_hours: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_rr_ratio: float = 0.0  # Average risk/reward
    expectancy: float = 0.0
    total_fees: float = 0.0


class PerformanceReporter:
    """Generates performance reports and analytics.

    Usage:
        reporter = PerformanceReporter()
        reporter.add_trade(trade)
        metrics = reporter.compute_metrics()
        report = reporter.generate_report()
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.risk_free_rate = config.get("risk_free_rate", 0.04)
        self.trading_days = config.get("trading_days", 365)
        self._trades: list[TradeRecord] = []
        self._equity_curve: list[tuple[datetime, float]] = []
        self._daily_returns: list[float] = []

    def add_trade(self, trade: TradeRecord) -> None:
        self._trades.append(trade)

    def add_equity_point(self, timestamp: datetime, equity: float) -> None:
        self._equity_curve.append((timestamp, equity))

    def compute_metrics(self) -> PerformanceMetrics:
        """Compute all performance metrics from trade history."""
        if not self._trades:
            return PerformanceMetrics()

        pnls = [t.pnl for t in self._trades]
        pnl_pcts = [t.pnl_pct for t in self._trades]
        winners = [t for t in self._trades if t.is_winner]
        losers = [t for t in self._trades if not t.is_winner]

        # Basic stats
        total_return = sum(pnls)
        win_rate = len(winners) / len(self._trades) if self._trades else 0

        # Win/loss stats
        avg_win = np.mean([t.pnl for t in winners]) if winners else 0
        avg_loss = np.mean([t.pnl for t in losers]) if losers else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Expectancy
        expectancy = (win_rate * avg_win + (1 - win_rate) * avg_loss) if self._trades else 0

        # Drawdown from equity curve
        max_dd = 0.0
        max_dd_duration = timedelta()
        if self._equity_curve:
            equities = [e[1] for e in self._equity_curve]
            peak = equities[0]
            dd_start = self._equity_curve[0][0]
            for i, (ts, eq) in enumerate(self._equity_curve):
                if eq > peak:
                    peak = eq
                    dd_start = ts
                dd = (peak - eq) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
                    max_dd_duration = ts - dd_start

        # Sharpe ratio from daily returns
        sharpe = 0.0
        sortino = 0.0
        if self._daily_returns:
            returns = np.array(self._daily_returns)
            excess = returns - self.risk_free_rate / self.trading_days
            std = np.std(excess)
            if std > 0:
                sharpe = np.mean(excess) / std * np.sqrt(self.trading_days)
            downside = returns[returns < 0]
            downside_std = np.std(downside) if len(downside) > 0 else 0
            if downside_std > 0:
                sortino = np.mean(excess) / downside_std * np.sqrt(self.trading_days)

        # Calmar ratio
        ann_return = total_return  # Simplified
        calmar = ann_return / max_dd if max_dd > 0 else 0

        # Average trade duration
        durations = [t.duration.total_seconds() / 3600 for t in self._trades]
        avg_duration = np.mean(durations) if durations else 0

        # Total fees
        total_fees = sum(t.fees for t in self._trades)

        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration_days=max_dd_duration.days,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_duration_hours=avg_duration,
            total_trades=len(self._trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            best_trade=max(pnls) if pnls else 0,
            worst_trade=min(pnls) if pnls else 0,
            expectancy=expectancy,
            total_fees=total_fees,
        )

    def generate_report(self) -> str:
        """Generate a text performance report."""
        m = self.compute_metrics()

        lines = [
            "=" * 60,
            "PERFORMANCE REPORT",
            "=" * 60,
            "",
            "--- Returns ---",
            f"Total Return:        ${m.total_return:,.2f}",
            f"Sharpe Ratio:        {m.sharpe_ratio:.2f}",
            f"Sortino Ratio:       {m.sortino_ratio:.2f}",
            f"Calmar Ratio:        {m.calmar_ratio:.2f}",
            "",
            "--- Risk ---",
            f"Max Drawdown:        {m.max_drawdown:.2%}",
            f"Max DD Duration:     {m.max_drawdown_duration_days:.0f} days",
            "",
            "--- Trades ---",
            f"Total Trades:        {m.total_trades}",
            f"Win Rate:            {m.win_rate:.1%}",
            f"Profit Factor:       {m.profit_factor:.2f}",
            f"Avg Win:             ${m.avg_win:,.2f}",
            f"Avg Loss:            ${m.avg_loss:,.2f}",
            f"Best Trade:          ${m.best_trade:,.2f}",
            f"Worst Trade:         ${m.worst_trade:,.2f}",
            f"Expectancy:          ${m.expectancy:,.2f}",
            "",
            "--- Costs ---",
            f"Total Fees:          ${m.total_fees:,.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def get_strategy_breakdown(self) -> dict[str, PerformanceMetrics]:
        """Get metrics per strategy."""
        strategy_trades: dict[str, list[TradeRecord]] = {}
        for trade in self._trades:
            sid = trade.strategy_id or "unknown"
            if sid not in strategy_trades:
                strategy_trades[sid] = []
            strategy_trades[sid].append(trade)

        breakdown = {}
        for sid, trades in strategy_trades.items():
            reporter = PerformanceReporter()
            reporter._trades = trades
            breakdown[sid] = reporter.compute_metrics()

        return breakdown

    def get_symbol_breakdown(self) -> dict[str, PerformanceMetrics]:
        """Get metrics per symbol."""
        symbol_trades: dict[str, list[TradeRecord]] = {}
        for trade in self._trades:
            if trade.symbol not in symbol_trades:
                symbol_trades[trade.symbol] = []
            symbol_trades[trade.symbol].append(trade)

        breakdown = {}
        for symbol, trades in symbol_trades.items():
            reporter = PerformanceReporter()
            reporter._trades = trades
            breakdown[symbol] = reporter.compute_metrics()

        return breakdown

    def get_daily_returns(self) -> list[float]:
        return list(self._daily_returns)

    def get_equity_curve(self) -> list[tuple[datetime, float]]:
        return list(self._equity_curve)
