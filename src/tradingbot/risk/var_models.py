"""Value at Risk (VaR) and Conditional VaR (CVaR) Models.

Implements multiple VaR methodologies:
- Historical simulation
- Parametric (variance-covariance)
- Monte Carlo simulation
- Cornish-Fisher VaR (accounts for skewness and kurtosis)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    """Result of a VaR calculation."""
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    cvar_95: float  # 95% CVaR (Expected Shortfall)
    cvar_99: float  # 99% CVaR
    method: str
    confidence_level: float = 0.95
    portfolio_value: float = 0.0
    num_observations: int = 0

    @property
    def var_95_pct(self) -> float:
        if self.portfolio_value == 0:
            return 0.0
        return self.var_95 / self.portfolio_value

    @property
    def var_99_pct(self) -> float:
        if self.portfolio_value == 0:
            return 0.0
        return self.var_99 / self.portfolio_value


class VaRModel:
    """Multi-method VaR/CVaR calculator.

    Usage:
        model = VaRModel()
        result = model.historical_var(returns, portfolio_value=100000)
        result = model.parametric_var(returns, portfolio_value=100000)
        result = model.monte_carlo_var(returns, portfolio_value=100000, n_simulations=10000)
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.confidence_levels = config.get("confidence_levels", [0.95, 0.99])
        self.lookback_days = config.get("lookback_days", 252)

    def historical_var(
        self,
        returns: np.ndarray,
        portfolio_value: float = 100000.0,
    ) -> VaRResult:
        """Historical simulation VaR.

        Uses actual historical return distribution — no distributional assumptions.
        """
        clean = returns[~np.isnan(returns)]
        if len(clean) < 10:
            return VaRResult(0, 0, 0, 0, "historical", portfolio_value=portfolio_value)

        sorted_returns = np.sort(clean)

        var_95_idx = int(len(sorted_returns) * 0.05)
        var_99_idx = int(len(sorted_returns) * 0.01)

        var_95 = abs(sorted_returns[var_95_idx]) * portfolio_value
        var_99 = abs(sorted_returns[var_99_idx]) * portfolio_value

        # CVaR = average of losses beyond VaR
        tail_95 = sorted_returns[:var_95_idx + 1]
        tail_99 = sorted_returns[:var_99_idx + 1]

        cvar_95 = abs(np.mean(tail_95)) * portfolio_value if len(tail_95) > 0 else var_95
        cvar_99 = abs(np.mean(tail_99)) * portfolio_value if len(tail_99) > 0 else var_99

        return VaRResult(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            method="historical",
            portfolio_value=portfolio_value,
            num_observations=len(clean),
        )

    def parametric_var(
        self,
        returns: np.ndarray,
        portfolio_value: float = 100000.0,
    ) -> VaRResult:
        """Parametric (variance-covariance) VaR.

        Assumes returns are normally distributed.
        """
        clean = returns[~np.isnan(returns)]
        if len(clean) < 10:
            return VaRResult(0, 0, 0, 0, "parametric", portfolio_value=portfolio_value)

        mu = np.mean(clean)
        sigma = np.std(clean, ddof=1)

        # z-scores for confidence levels
        from scipy import stats
        z_95 = stats.norm.ppf(0.05)
        z_99 = stats.norm.ppf(0.01)

        var_95 = abs(mu + z_95 * sigma) * portfolio_value
        var_99 = abs(mu + z_99 * sigma) * portfolio_value

        # CVaR for normal distribution
        cvar_95 = abs(mu - sigma * stats.norm.pdf(z_95) / 0.05) * portfolio_value
        cvar_99 = abs(mu - sigma * stats.norm.pdf(z_99) / 0.01) * portfolio_value

        return VaRResult(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            method="parametric",
            portfolio_value=portfolio_value,
            num_observations=len(clean),
        )

    def cornish_fisher_var(
        self,
        returns: np.ndarray,
        portfolio_value: float = 100000.0,
    ) -> VaRResult:
        """Cornish-Fisher VaR — adjusted for skewness and kurtosis.

        Better than parametric for non-normal return distributions.
        """
        clean = returns[~np.isnan(returns)]
        if len(clean) < 20:
            return VaRResult(0, 0, 0, 0, "cornish_fisher", portfolio_value=portfolio_value)

        mu = np.mean(clean)
        sigma = np.std(clean, ddof=1)

        if sigma == 0:
            return VaRResult(0, 0, 0, 0, "cornish_fisher", portfolio_value=portfolio_value)

        skew = float(np.mean(((clean - mu) / sigma) ** 3))
        kurt = float(np.mean(((clean - mu) / sigma) ** 4) - 3)  # excess kurtosis

        from scipy import stats
        z_95 = stats.norm.ppf(0.05)
        z_99 = stats.norm.ppf(0.01)

        # Cornish-Fisher adjustment
        cf_95 = z_95 + (z_95**2 - 1) * skew / 6 + (z_95**3 - 3*z_95) * kurt / 24 - (2*z_95**3 - 5*z_95) * skew**2 / 36
        cf_99 = z_99 + (z_99**2 - 1) * skew / 6 + (z_99**3 - 3*z_99) * kurt / 24 - (2*z_99**3 - 5*z_99) * skew**2 / 36

        var_95 = abs(mu + cf_95 * sigma) * portfolio_value
        var_99 = abs(mu + cf_99 * sigma) * portfolio_value

        # Approximate CVaR using historical tail
        sorted_returns = np.sort(clean)
        tail_idx_95 = max(1, int(len(sorted_returns) * 0.05))
        tail_idx_99 = max(1, int(len(sorted_returns) * 0.01))
        cvar_95 = abs(np.mean(sorted_returns[:tail_idx_95])) * portfolio_value
        cvar_99 = abs(np.mean(sorted_returns[:tail_idx_99])) * portfolio_value

        return VaRResult(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            method="cornish_fisher",
            portfolio_value=portfolio_value,
            num_observations=len(clean),
        )

    def monte_carlo_var(
        self,
        returns: np.ndarray,
        portfolio_value: float = 100000.0,
        n_simulations: int = 10000,
        horizon_days: int = 1,
    ) -> VaRResult:
        """Monte Carlo simulation VaR.

        Simulates future portfolio values using bootstrapped returns.
        """
        clean = returns[~np.isnan(returns)]
        if len(clean) < 20:
            return VaRResult(0, 0, 0, 0, "monte_carlo", portfolio_value=portfolio_value)

        mu = np.mean(clean)
        sigma = np.std(clean, ddof=1)

        # Simulate returns
        rng = np.random.default_rng()
        simulated_returns = rng.normal(mu, sigma, (n_simulations, horizon_days))
        cumulative_returns = np.prod(1 + simulated_returns, axis=1) - 1

        # Compute P&L
        pnl = cumulative_returns * portfolio_value

        var_95 = abs(np.percentile(pnl, 5))
        var_99 = abs(np.percentile(pnl, 1))

        # CVaR
        tail_95 = pnl[pnl <= np.percentile(pnl, 5)]
        tail_99 = pnl[pnl <= np.percentile(pnl, 1)]

        cvar_95 = abs(np.mean(tail_95)) if len(tail_95) > 0 else var_95
        cvar_99 = abs(np.mean(tail_99)) if len(tail_99) > 0 else var_99

        return VaRResult(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            method="monte_carlo",
            portfolio_value=portfolio_value,
            num_observations=n_simulations,
        )

    def component_var(
        self,
        returns_matrix: np.ndarray,
        weights: np.ndarray,
        portfolio_value: float = 100000.0,
    ) -> dict[str, np.ndarray]:
        """Component VaR — risk contribution of each asset.

        Args:
            returns_matrix: (n_obs, n_assets) matrix of asset returns
            weights: portfolio weights
            portfolio_value: total portfolio value

        Returns:
            Dict with component_var, pct_contribution, marginal_var
        """
        n_assets = returns_matrix.shape[1]
        cov_matrix = np.cov(returns_matrix, rowvar=False)

        portfolio_var = np.sqrt(weights @ cov_matrix @ weights)

        # Marginal VaR = (Sigma * w) / sigma_p
        if portfolio_var > 0:
            marginal_var = (cov_matrix @ weights) / portfolio_var
        else:
            marginal_var = np.zeros(n_assets)

        # Component VaR = weight_i * marginal_var_i
        component_var = weights * marginal_var * portfolio_value

        # Percentage contribution
        total = np.sum(np.abs(component_var))
        pct_contribution = component_var / total if total > 0 else np.zeros(n_assets)

        return {
            "component_var": component_var,
            "pct_contribution": pct_contribution,
            "marginal_var": marginal_var,
            "portfolio_vol": portfolio_var,
        }

    def rolling_var(
        self,
        returns: np.ndarray,
        window: int = 63,
        portfolio_value: float = 100000.0,
    ) -> np.ndarray:
        """Compute rolling VaR over time."""
        n = len(returns)
        result = np.full(n, np.nan)

        for i in range(window, n):
            window_returns = returns[i - window:i]
            clean = window_returns[~np.isnan(window_returns)]
            if len(clean) >= 10:
                sorted_r = np.sort(clean)
                idx = int(len(sorted_r) * 0.05)
                result[i] = abs(sorted_r[idx]) * portfolio_value

        return result
