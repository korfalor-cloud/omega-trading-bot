"""Strategy Comparison — side-by-side strategy analysis.

Implements:
- Multi-strategy comparison
- Performance ranking
- Risk-adjusted comparison
- Correlation between strategies
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Strategy comparison metrics."""
    strategy_id: str = ""
    total_return: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade: float = 0.0
    rank: int = 0


class StrategyComparator:
    """Compare multiple strategies."""

    def __init__(self):
        self._strategies: dict[str, list[float]] = {}

    def add_strategy(self, strategy_id: str, returns: list[float]) -> None:
        self._strategies[strategy_id] = returns

    def compare(self) -> list[StrategyMetrics]:
        """Compare all strategies."""
        results = []

        for sid, returns in self._strategies.items():
            r = np.array(returns)
            if len(r) < 2:
                continue

            total_return = float(np.prod(1 + r) - 1)
            mean_r = np.mean(r)
            std_r = np.std(r)
            down_std = np.std(r[r < 0]) if np.any(r < 0) else std_r

            sharpe = mean_r / std_r * np.sqrt(365) if std_r > 0 else 0
            sortino = mean_r / down_std * np.sqrt(365) if down_std > 0 else 0

            # Max drawdown
            equity = np.cumprod(1 + r)
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / peak
            max_dd = float(np.max(dd))

            win_rate = float(np.sum(r > 0) / len(r))
            wins = r[r > 0]
            losses = r[r < 0]
            pf = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else 0

            results.append(StrategyMetrics(
                strategy_id=sid,
                total_return=total_return,
                sharpe=sharpe,
                sortino=sortino,
                max_drawdown=max_dd,
                win_rate=win_rate,
                profit_factor=pf,
                total_trades=len(r),
                avg_trade=float(mean_r),
            ))

        # Rank by Sharpe
        results.sort(key=lambda x: x.sharpe, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    def get_correlation_matrix(self) -> dict[str, dict[str, float]]:
        """Get correlation between strategies."""
        sids = list(self._strategies.keys())
        n = len(sids)
        corr = {}

        for i in range(n):
            corr[sids[i]] = {}
            for j in range(n):
                if i == j:
                    corr[sids[i]][sids[j]] = 1.0
                else:
                    r_i = np.array(self._strategies[sids[i]])
                    r_j = np.array(self._strategies[sids[j]])
                    min_len = min(len(r_i), len(r_j))
                    if min_len > 1:
                        corr[sids[i]][sids[j]] = float(np.corrcoef(r_i[-min_len:], r_j[-min_len:])[0, 1])
                    else:
                        corr[sids[i]][sids[j]] = 0.0

        return corr

    def format_comparison(self) -> str:
        """Format comparison as text table."""
        results = self.compare()
        lines = [
            f"\n{'='*80}",
            f"  Strategy Comparison",
            f"{'='*80}",
            f"  {'Rank':<6}{'Strategy':<20}{'Return':<12}{'Sharpe':<10}{'Sortino':<10}{'MaxDD':<10}{'WinRate':<10}",
            f"  {'─'*70}",
        ]
        for r in results:
            lines.append(f"  {r.rank:<6}{r.strategy_id:<20}{r.total_return:>10.2%}{r.sharpe:>10.2f}{r.sortino:>10.2f}{r.max_drawdown:>10.2%}{r.win_rate:>10.1%}")
        return "\n".join(lines)
