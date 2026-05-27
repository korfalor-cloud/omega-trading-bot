"""Walk-Forward Optimization and Monte Carlo Simulation.

Implements:
- Walk-forward analysis (in-sample optimize, out-of-sample validate)
- Monte Carlo simulation for strategy robustness
- Bootstrap confidence intervals
- Regime-segmented backtesting
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """A single walk-forward fold result."""
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_metric: float = 0.0
    test_metric: float = 0.0
    best_params: dict = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Complete walk-forward analysis result."""
    folds: list[WalkForwardFold] = field(default_factory=list)
    avg_oos_metric: float = 0.0  # Out-of-sample average
    std_oos_metric: float = 0.0
    oos_metrics: list[float] = field(default_factory=list)
    degradation_pct: float = 0.0  # IS to OOS degradation

    @property
    def is_robust(self) -> bool:
        """Strategy is robust if OOS performance is within 50% of IS."""
        return self.degradation_pct < 0.5


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result."""
    n_simulations: int
    mean_return: float = 0.0
    std_return: float = 0.0
    percentile_5: float = 0.0
    percentile_25: float = 0.0
    percentile_50: float = 0.0
    percentile_75: float = 0.0
    percentile_95: float = 0.0
    prob_positive: float = 0.0
    prob_loss_10pct: float = 0.0
    max_drawdown_mean: float = 0.0
    max_drawdown_95: float = 0.0
    sharpe_mean: float = 0.0
    sharpe_5: float = 0.0
    paths: Optional[np.ndarray] = None


class WalkForwardAnalyzer:
    """Walk-forward optimization framework.

    Splits data into rolling train/test windows, optimizes parameters
    on train, and validates on test to detect overfitting.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.n_splits = config.get("n_splits", 5)
        self.train_ratio = config.get("train_ratio", 0.7)
        self.anchored = config.get("anchored", False)  # Expanding vs rolling window

    def analyze(
        self,
        data: np.ndarray,
        optimize_func: Callable[[np.ndarray], dict],
        evaluate_func: Callable[[np.ndarray, dict], float],
    ) -> WalkForwardResult:
        """Run walk-forward analysis.

        Args:
            data: Full dataset
            optimize_func: Takes train data, returns best params
            evaluate_func: Takes data + params, returns metric
        """
        n = len(data)
        fold_size = n // self.n_splits
        train_size = int(fold_size * self.train_ratio)
        test_size = fold_size - train_size

        folds = []
        oos_metrics = []

        for i in range(self.n_splits):
            if self.anchored:
                train_start = 0
            else:
                train_start = i * fold_size

            train_end = train_start + train_size + i * fold_size if self.anchored else train_start + train_size
            train_end = min(train_end, n - test_size)
            test_start = train_end
            test_end = min(test_start + test_size, n)

            if test_start >= n or train_end - train_start < 50:
                continue

            train_data = data[train_start:train_end]
            test_data = data[test_start:test_end]

            # Optimize on train
            best_params = optimize_func(train_data)
            train_metric = evaluate_func(train_data, best_params)
            test_metric = evaluate_func(test_data, best_params)

            fold = WalkForwardFold(
                fold_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_metric=train_metric,
                test_metric=test_metric,
                best_params=best_params,
            )
            folds.append(fold)
            oos_metrics.append(test_metric)

        # Compute summary
        avg_oos = np.mean(oos_metrics) if oos_metrics else 0
        std_oos = np.std(oos_metrics) if oos_metrics else 0
        avg_is = np.mean([f.train_metric for f in folds]) if folds else 0
        degradation = 1 - avg_oos / avg_is if avg_is != 0 else 0

        return WalkForwardResult(
            folds=folds,
            avg_oos_metric=avg_oos,
            std_oos_metric=std_oos,
            oos_metrics=oos_metrics,
            degradation_pct=degradation,
        )


class MonteCarloSimulator:
    """Monte Carlo simulation for strategy robustness.

    Bootstraps trade returns to generate distribution of outcomes.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.n_simulations = config.get("n_simulations", 10000)
        self.initial_capital = config.get("initial_capital", 100000)
        self.seed = config.get("seed", 42)

    def simulate(
        self,
        trade_returns: np.ndarray,
        trades_per_year: int = 252,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation from historical trade returns.

        Args:
            trade_returns: Array of individual trade returns
            trades_per_year: Annualization factor
        """
        rng = np.random.default_rng(self.seed)
        n_trades = len(trade_returns)

        if n_trades < 5:
            return MonteCarloResult(n_simulations=0)

        # Bootstrap paths
        n_periods = min(trades_per_year, 252)  # Simulate 1 year
        paths = np.zeros((self.n_simulations, n_periods + 1))
        paths[:, 0] = self.initial_capital

        terminal_returns = np.zeros(self.n_simulations)
        max_drawdowns = np.zeros(self.n_simulations)
        sharpes = np.zeros(self.n_simulations)

        for sim in range(self.n_simulations):
            # Bootstrap trade returns
            sampled_returns = rng.choice(trade_returns, size=n_periods, replace=True)

            # Build equity curve
            cumulative = np.cumprod(1 + sampled_returns)
            equity = self.initial_capital * np.concatenate([[1], cumulative])
            paths[sim] = equity

            # Terminal return
            terminal_returns[sim] = (equity[-1] / self.initial_capital) - 1

            # Max drawdown
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            max_drawdowns[sim] = np.max(drawdown)

            # Sharpe
            if np.std(sampled_returns) > 0:
                sharpes[sim] = np.mean(sampled_returns) / np.std(sampled_returns) * np.sqrt(trades_per_year)

        return MonteCarloResult(
            n_simulations=self.n_simulations,
            mean_return=float(np.mean(terminal_returns)),
            std_return=float(np.std(terminal_returns)),
            percentile_5=float(np.percentile(terminal_returns, 5)),
            percentile_25=float(np.percentile(terminal_returns, 25)),
            percentile_50=float(np.percentile(terminal_returns, 50)),
            percentile_75=float(np.percentile(terminal_returns, 75)),
            percentile_95=float(np.percentile(terminal_returns, 95)),
            prob_positive=float(np.mean(terminal_returns > 0)),
            prob_loss_10pct=float(np.mean(terminal_returns < -0.1)),
            max_drawdown_mean=float(np.mean(max_drawdowns)),
            max_drawdown_95=float(np.percentile(max_drawdowns, 95)),
            sharpe_mean=float(np.mean(sharpes)),
            sharpe_5=float(np.percentile(sharpes, 5)),
            paths=paths,
        )

    def bootstrap_confidence_interval(
        self,
        metric_values: np.ndarray,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Bootstrap confidence interval for a metric."""
        rng = np.random.default_rng(self.seed)
        n_bootstrap = min(self.n_simulations, 5000)
        bootstrap_metrics = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            sample = rng.choice(metric_values, size=len(metric_values), replace=True)
            bootstrap_metrics[i] = np.mean(sample)

        alpha = (1 - confidence) / 2
        lower = np.percentile(bootstrap_metrics, alpha * 100)
        upper = np.percentile(bootstrap_metrics, (1 - alpha) * 100)
        return float(lower), float(upper)

    def generate_report(self, result: MonteCarloResult) -> str:
        """Generate Monte Carlo report text."""
        lines = [
            "=" * 50,
            "MONTE CARLO SIMULATION REPORT",
            "=" * 50,
            f"Simulations:         {result.n_simulations:,}",
            "",
            "--- Returns ---",
            f"Mean Return:         {result.mean_return:.2%}",
            f"Median Return:       {result.percentile_50:.2%}",
            f"Std Deviation:       {result.std_return:.2%}",
            f"5th Percentile:      {result.percentile_5:.2%}",
            f"95th Percentile:     {result.percentile_95:.2%}",
            "",
            "--- Risk ---",
            f"Prob(Positive):      {result.prob_positive:.1%}",
            f"Prob(Loss > 10%):    {result.prob_loss_10pct:.1%}",
            f"Max DD (mean):       {result.max_drawdown_mean:.2%}",
            f"Max DD (95th pct):   {result.max_drawdown_95:.2%}",
            "",
            "--- Risk-Adjusted ---",
            f"Sharpe (mean):       {result.sharpe_mean:.2f}",
            f"Sharpe (5th pct):    {result.sharpe_5:.2f}",
            "=" * 50,
        ]
        return "\n".join(lines)
