"""Backtesting Analytics — detailed trade analysis and performance metrics.

Implements:
- Trade-level analysis (MAE, MFE, time in trade)
- Equity curve analysis (drawdowns, underwater periods)
- Return distribution analysis
- Rolling performance metrics
- Strategy comparison
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeAnalysis:
    """Per-trade analysis."""
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    mae: float = 0.0  # Max Adverse Excursion
    mfe: float = 0.0  # Max Favorable Excursion
    bars_held: int = 0
    side: str = ""


@dataclass
class EquityAnalysis:
    """Equity curve analysis."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    calmar_ratio: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    n_drawdowns: int = 0


class BacktestAnalytics:
    """Detailed backtesting analytics."""

    def __init__(self, annualization: int = 365):
        self.annualization = annualization

    def analyze_trades(self, trades: list[TradeAnalysis]) -> dict:
        """Analyze trade-level statistics."""
        if not trades:
            return {}

        pnls = np.array([t.pnl for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        maes = np.array([t.mae for t in trades])
        mfes = np.array([t.mfe for t in trades])
        durations = np.array([t.bars_held for t in trades])

        return {
            "n_trades": len(trades),
            "win_rate": len(wins) / len(trades),
            "avg_pnl": float(np.mean(pnls)),
            "median_pnl": float(np.median(pnls)),
            "avg_win": float(np.mean(wins)) if len(wins) > 0 else 0,
            "avg_loss": float(np.mean(losses)) if len(losses) > 0 else 0,
            "largest_win": float(np.max(wins)) if len(wins) > 0 else 0,
            "largest_loss": float(np.min(losses)) if len(losses) > 0 else 0,
            "profit_factor": float(np.sum(wins) / abs(np.sum(losses))) if np.sum(losses) != 0 else float("inf"),
            "avg_mae": float(np.mean(maes)),
            "avg_mfe": float(np.mean(mfes)),
            "avg_duration": float(np.mean(durations)),
            "payoff_ratio": float(np.mean(wins) / abs(np.mean(losses))) if len(losses) > 0 and np.mean(losses) != 0 else 0,
        }

    def analyze_equity(self, equity_curve: np.ndarray) -> EquityAnalysis:
        """Analyze equity curve metrics."""
        if len(equity_curve) < 2:
            return EquityAnalysis()

        returns = np.diff(equity_curve) / equity_curve[:-1]
        total_return = (equity_curve[-1] / equity_curve[0]) - 1

        # Drawdown analysis
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (peak - equity_curve) / peak
        max_dd = float(np.max(drawdowns))

        # Max drawdown duration
        underwater = drawdowns > 0
        max_dd_dur = 0
        current_dur = 0
        for uw in underwater:
            if uw:
                current_dur += 1
                max_dd_dur = max(max_dd_dur, current_dur)
            else:
                current_dur = 0

        # Count drawdown periods
        dd_changes = np.diff(underwater.astype(int))
        n_drawdowns = int(np.sum(dd_changes == 1))

        # Risk-adjusted metrics
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        down_std = np.std(returns[returns < 0]) if np.any(returns < 0) else std_ret

        sharpe = (mean_ret / std_ret * np.sqrt(self.annualization)) if std_ret > 0 else 0
        sortino = (mean_ret / down_std * np.sqrt(self.annualization)) if down_std > 0 else 0
        calmar = (total_return / max_dd) if max_dd > 0 else 0

        ann_ret = (1 + total_return) ** (self.annualization / len(equity_curve)) - 1

        return EquityAnalysis(
            total_return=total_return,
            annualized_return=ann_ret,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_dur,
            calmar_ratio=calmar,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            skewness=float(np.mean(((returns - mean_ret) / std_ret) ** 3)) if std_ret > 0 else 0,
            kurtosis=float(np.mean(((returns - mean_ret) / std_ret) ** 4) - 3) if std_ret > 0 else 0,
            n_drawdowns=n_drawdowns,
        )

    def rolling_metrics(
        self,
        equity_curve: np.ndarray,
        window: int = 30,
    ) -> dict[str, np.ndarray]:
        """Compute rolling performance metrics."""
        if len(equity_curve) < window + 1:
            return {}

        returns = np.diff(equity_curve) / equity_curve[:-1]
        n = len(returns)

        rolling_sharpe = np.full(n, np.nan)
        rolling_vol = np.full(n, np.nan)
        rolling_return = np.full(n, np.nan)

        for i in range(window, n):
            window_ret = returns[i - window:i]
            rolling_return[i] = np.mean(window_ret) * self.annualization
            rolling_vol[i] = np.std(window_ret) * np.sqrt(self.annualization)
            if rolling_vol[i] > 0:
                rolling_sharpe[i] = rolling_return[i] / rolling_vol[i]

        return {
            "rolling_return": rolling_return,
            "rolling_volatility": rolling_vol,
            "rolling_sharpe": rolling_sharpe,
        }

    def return_distribution(self, equity_curve: np.ndarray) -> dict:
        """Analyze return distribution."""
        returns = np.diff(equity_curve) / equity_curve[:-1]

        percentiles = {}
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            percentiles[f"p{p}"] = float(np.percentile(returns, p))

        return {
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
            "min": float(np.min(returns)),
            "max": float(np.max(returns)),
            "positive_days": int(np.sum(returns > 0)),
            "negative_days": int(np.sum(returns < 0)),
            "percentiles": percentiles,
        }
