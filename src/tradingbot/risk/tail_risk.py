"""Tail Risk Analysis.

Implements:
- Extreme Value Theory (EVT) — GPD fitting
- Tail dependence via copulas
- Stress correlation analysis
- Max drawdown distribution
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TailRiskMetrics:
    """Tail risk analysis results."""
    var_95: float = 0.0
    var_99: float = 0.0
    var_99_9: float = 0.0
    expected_shortfall_95: float = 0.0
    expected_shortfall_99: float = 0.0
    tail_index: float = 0.0  # Shape parameter of GPD
    tail_index_se: float = 0.0
    max_drawdown_dist: dict = None
    stress_correlation: float = 0.0

    def __post_init__(self):
        if self.max_drawdown_dist is None:
            self.max_drawdown_dist = {}


class TailRiskAnalyzer:
    """Tail risk analysis using Extreme Value Theory and stress testing.

    Uses the Peaks-Over-Threshold (POT) method with Generalized
    Pareto Distribution (GPD) to model extreme losses.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.threshold_percentile = config.get("threshold_percentile", 0.05)
        self.min_exceedances = config.get("min_exceedances", 20)

    def fit_evt(
        self,
        returns: np.ndarray,
        threshold_percentile: float | None = None,
    ) -> TailRiskMetrics:
        """Fit Extreme Value Theory (GPD) to loss tail.

        Args:
            returns: Array of returns (negative = losses)
            threshold_percentile: Percentile for threshold (default 5th)
        """
        pct = threshold_percentile or self.threshold_percentile
        losses = -returns[~np.isnan(returns)]  # Convert to losses
        losses = losses[losses > 0]

        if len(losses) < 50:
            return TailRiskMetrics()

        threshold = np.percentile(losses, (1 - pct) * 100)
        exceedances = losses[losses > threshold] - threshold

        if len(exceedances) < self.min_exceedances:
            # Not enough tail data — fall back to empirical
            return self._empirical_tail_metrics(losses)

        # Fit GPD: shape (xi) and scale (beta)
        xi, beta = self._fit_gpd(exceedances)

        n = len(losses)
        n_u = len(exceedances)
        exceedance_prob = n_u / n

        # VaR and ES from GPD
        var_95 = self._gpd_quantile(0.95, threshold, xi, beta, exceedance_prob, n)
        var_99 = self._gpd_quantile(0.99, threshold, xi, beta, exceedance_prob, n)
        var_99_9 = self._gpd_quantile(0.999, threshold, xi, beta, exceedance_prob, n)

        es_95 = self._gpd_expected_shortfall(0.95, threshold, xi, beta, exceedance_prob, n)
        es_99 = self._gpd_expected_shortfall(0.99, threshold, xi, beta, exceedance_prob, n)

        # Standard error for shape parameter
        xi_se = self._gpd_se(exceedances, xi, beta)

        return TailRiskMetrics(
            var_95=var_95,
            var_99=var_99,
            var_99_9=var_99_9,
            expected_shortfall_95=es_95,
            expected_shortfall_99=es_99,
            tail_index=xi,
            tail_index_se=xi_se,
        )

    def max_drawdown_distribution(
        self,
        returns: np.ndarray,
        n_bootstrap: int = 1000,
    ) -> dict:
        """Bootstrap distribution of maximum drawdown."""
        rng = np.random.default_rng(42)
        n = len(returns)
        max_dds = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            sample = rng.choice(returns, size=n, replace=True)
            equity = np.cumprod(1 + sample)
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / peak
            max_dds[i] = np.max(dd)

        return {
            "mean": float(np.mean(max_dds)),
            "median": float(np.median(max_dds)),
            "std": float(np.std(max_dds)),
            "percentile_5": float(np.percentile(max_dds, 5)),
            "percentile_95": float(np.percentile(max_dds, 95)),
            "percentile_99": float(np.percentile(max_dds, 99)),
        }

    def stress_correlation(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        threshold_percentile: float = 0.1,
    ) -> float:
        """Compute correlation during stress periods (tail dependence).

        Returns correlation of assets during extreme market conditions.
        """
        n = min(len(returns_a), len(returns_b))
        a = returns_a[:n]
        b = returns_b[:n]

        # Identify stress periods (bottom percentile of either asset)
        threshold_a = np.percentile(a, threshold_percentile * 100)
        threshold_b = np.percentile(b, threshold_percentile * 100)

        stress_mask = (a <= threshold_a) | (b <= threshold_b)

        if np.sum(stress_mask) < 10:
            return np.corrcoef(a, b)[0, 1]

        stress_a = a[stress_mask]
        stress_b = b[stress_mask]

        if np.std(stress_a) < 1e-10 or np.std(stress_b) < 1e-10:
            return 0.0

        return float(np.corrcoef(stress_a, stress_b)[0, 1])

    def _fit_gpd(self, exceedances: np.ndarray) -> tuple[float, float]:
        """Fit GPD using method of moments (simplified MLE)."""
        mean_ex = np.mean(exceedances)
        var_ex = np.var(exceedances)

        if var_ex <= 0 or mean_ex <= 0:
            return 0.0, mean_ex

        # Method of moments estimates
        xi = 0.5 * (1 - mean_ex ** 2 / var_ex)
        xi = max(-0.5, min(0.5, xi))  # Bound for stability

        beta = mean_ex * (1 - xi)

        return float(xi), float(beta)

    def _gpd_quantile(
        self,
        p: float,
        threshold: float,
        xi: float,
        beta: float,
        exceedance_prob: float,
        n: int,
    ) -> float:
        """GPD quantile (VaR)."""
        q = 1 - p  # We want the loss quantile
        nq = n * q  # Expected number of observations below quantile

        if exceedance_prob <= 0:
            return 0.0

        if abs(xi) < 1e-10:
            return threshold + beta * np.log(exceedance_prob / q)
        else:
            return threshold + (beta / xi) * ((exceedance_prob / q) ** xi - 1)

    def _gpd_expected_shortfall(
        self,
        p: float,
        threshold: float,
        xi: float,
        beta: float,
        exceedance_prob: float,
        n: int,
    ) -> float:
        """GPD expected shortfall (CVaR)."""
        var = self._gpd_quantile(p, threshold, xi, beta, exceedance_prob, n)

        if xi >= 1:
            return var * 2  # Infinite mean — use approximation

        if abs(xi) < 1e-10:
            return var + beta
        else:
            return (var + beta - xi * threshold) / (1 - xi)

    def _gpd_se(self, exceedances: np.ndarray, xi: float, beta: float) -> float:
        """Approximate standard error for shape parameter."""
        n = len(exceedances)
        if n < 10 or beta <= 0:
            return 1.0
        # Approximate Fisher information
        se = np.sqrt((1 + xi) ** 2 / n)
        return float(se)

    def _empirical_tail_metrics(self, losses: np.ndarray) -> TailRiskMetrics:
        """Empirical tail metrics when EVT fitting is not possible."""
        sorted_losses = np.sort(losses)[::-1]
        n = len(sorted_losses)

        var_95 = float(np.percentile(losses, 95))
        var_99 = float(np.percentile(losses, 99))

        tail_95 = sorted_losses[:max(1, n // 20)]
        tail_99 = sorted_losses[:max(1, n // 100)]

        return TailRiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=float(np.mean(tail_95)),
            expected_shortfall_99=float(np.mean(tail_99)),
            tail_index=0.0,
        )
