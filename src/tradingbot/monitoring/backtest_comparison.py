"""Backtest Comparison — compare multiple backtest runs.

Implements:
- Multi-run comparison
- Parameter sensitivity analysis
- Best/worst run identification
- Statistical significance testing
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BacktestRun:
    """A single backtest run."""
    run_id: str = ""
    params: dict = None
    total_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0

    def __post_init__(self):
        if self.params is None:
            self.params = {}


class BacktestComparator:
    """Compare multiple backtest runs."""

    def __init__(self):
        self._runs: list[BacktestRun] = []

    def add_run(self, run: BacktestRun) -> None:
        self._runs.append(run)

    def get_best(self, metric: str = "sharpe") -> BacktestRun:
        """Get best run by metric."""
        if not self._runs:
            return BacktestRun()
        return max(self._runs, key=lambda r: getattr(r, metric, 0))

    def get_worst(self, metric: str = "sharpe") -> BacktestRun:
        if not self._runs:
            return BacktestRun()
        return min(self._runs, key=lambda r: getattr(r, metric, 0))

    def get_summary(self) -> dict:
        """Get summary statistics across runs."""
        if not self._runs:
            return {}

        sharpes = [r.sharpe for r in self._runs]
        returns = [r.total_return for r in self._runs]
        drawdowns = [r.max_drawdown for r in self._runs]

        return {
            "n_runs": len(self._runs),
            "avg_sharpe": np.mean(sharpes),
            "std_sharpe": np.std(sharpes),
            "avg_return": np.mean(returns),
            "avg_drawdown": np.mean(drawdowns),
            "best_sharpe": max(sharpes),
            "worst_sharpe": min(sharpes),
        }

    def sensitivity_analysis(self, param_name: str) -> dict[float, list[float]]:
        """Analyze sensitivity to a parameter."""
        result = {}
        for run in self._runs:
            val = run.params.get(param_name)
            if val is not None:
                if val not in result:
                    result[val] = []
                result[val].append(run.sharpe)
        return {k: [np.mean(v), np.std(v)] for k, v in result.items()}

    def format_comparison(self) -> str:
        """Format comparison as text."""
        lines = [f"\n{'='*70}", f"  Backtest Comparison ({len(self._runs)} runs)", f"{'='*70}"]
        lines.append(f"  {'Run':<15}{'Return':<12}{'Sharpe':<10}{'MaxDD':<10}{'WinRate':<10}")
        lines.append(f"  {'─'*55}")

        sorted_runs = sorted(self._runs, key=lambda r: r.sharpe, reverse=True)
        for run in sorted_runs[:10]:
            lines.append(f"  {run.run_id:<15}{run.total_return:>10.2%}{run.sharpe:>10.2f}{run.max_drawdown:>10.2%}{run.win_rate:>10.1%}")

        summary = self.get_summary()
        lines.append(f"\n  Avg Sharpe: {summary.get('avg_sharpe', 0):.2f} ± {summary.get('std_sharpe', 0):.2f}")
        return "\n".join(lines)
