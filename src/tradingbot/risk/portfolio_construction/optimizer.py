"""Portfolio Optimization — Modern Portfolio Theory and beyond.

Implements:
- Mean-Variance Optimization (Markowitz)
- Minimum Variance Portfolio
- Maximum Sharpe Ratio Portfolio
- Risk Parity / Equal Risk Contribution
- Hierarchical Risk Parity (HRP)
- Black-Litterman Model
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PortfolioAllocation:
    """Result of portfolio optimization."""
    weights: np.ndarray
    asset_names: list[str]
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    method: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        return {name: float(w) for name, w in zip(self.asset_names, self.weights)}


class PortfolioOptimizer:
    """Portfolio optimization engine.

    Args:
        risk_free_rate: Annual risk-free rate (default 0.04)
        max_weight: Maximum weight per asset (default 0.4)
        min_weight: Minimum weight per asset (default 0.0)
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.risk_free_rate = config.get("risk_free_rate", 0.04)
        self.max_weight = config.get("max_weight", 0.4)
        self.min_weight = config.get("min_weight", 0.0)
        self.trading_days = config.get("trading_days", 365)  # Crypto = 365

    def mean_variance(
        self,
        returns: np.ndarray,
        target_return: float | None = None,
        asset_names: list[str] | None = None,
    ) -> PortfolioAllocation:
        """Classic Markowitz mean-variance optimization.

        Args:
            returns: (n_obs, n_assets) return matrix
            target_return: Target annual return; if None, maximizes Sharpe
            asset_names: Asset labels
        """
        n_assets = returns.shape[1]
        names = asset_names or [f"asset_{i}" for i in range(n_assets)]

        mu = np.mean(returns, axis=0) * self.trading_days
        cov = np.cov(returns, rowvar=False) * self.trading_days

        if target_return is not None:
            weights = self._solve_mean_variance(mu, cov, target_return)
        else:
            weights = self._max_sharpe(mu, cov)

        weights = self._clip_weights(weights)
        port_ret = weights @ mu
        port_vol = np.sqrt(weights @ cov @ weights)
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        return PortfolioAllocation(
            weights=weights,
            asset_names=names,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            method="mean_variance",
        )

    def minimum_variance(
        self,
        returns: np.ndarray,
        asset_names: list[str] | None = None,
    ) -> PortfolioAllocation:
        """Minimum variance portfolio — no expected return estimates needed."""
        n_assets = returns.shape[1]
        names = asset_names or [f"asset_{i}" for i in range(n_assets)]

        cov = np.cov(returns, rowvar=False) * self.trading_days
        weights = self._min_variance_weights(cov)
        weights = self._clip_weights(weights)

        mu = np.mean(returns, axis=0) * self.trading_days
        port_ret = weights @ mu
        port_vol = np.sqrt(weights @ cov @ weights)
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        return PortfolioAllocation(
            weights=weights,
            asset_names=names,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            method="minimum_variance",
        )

    def risk_parity(
        self,
        returns: np.ndarray,
        asset_names: list[str] | None = None,
    ) -> PortfolioAllocation:
        """Risk Parity / Equal Risk Contribution.

        Each asset contributes equally to total portfolio risk.
        """
        n_assets = returns.shape[1]
        names = asset_names or [f"asset_{i}" for i in range(n_assets)]

        cov = np.cov(returns, rowvar=False) * self.trading_days
        weights = self._risk_parity_weights(cov)
        weights = self._clip_weights(weights)

        mu = np.mean(returns, axis=0) * self.trading_days
        port_ret = weights @ mu
        port_vol = np.sqrt(weights @ cov @ weights)
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        # Verify risk contributions
        risk_contrib = self._risk_contributions(weights, cov)

        return PortfolioAllocation(
            weights=weights,
            asset_names=names,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            method="risk_parity",
            metadata={"risk_contributions": risk_contrib.tolist()},
        )

    def hierarchical_risk_parity(
        self,
        returns: np.ndarray,
        asset_names: list[str] | None = None,
    ) -> PortfolioAllocation:
        """Hierarchical Risk Parity (HRP) — Lopez de Prado.

        Uses hierarchical clustering on the correlation matrix to build
        a portfolio that is robust to estimation error.
        """
        n_assets = returns.shape[1]
        names = asset_names or [f"asset_{i}" for i in range(n_assets)]

        cov = np.cov(returns, rowvar=False) * self.trading_days
        corr = np.corrcoef(returns, rowvar=False)

        # Step 1: Tree clustering
        dist = self._correlation_distance(corr)
        link = self._quasi_diag(dist)

        # Step 2: Recursive bisection
        weights = self._recursive_bisection(cov, link)
        weights = self._clip_weights(weights)

        mu = np.mean(returns, axis=0) * self.trading_days
        port_ret = weights @ mu
        port_vol = np.sqrt(weights @ cov @ weights)
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        return PortfolioAllocation(
            weights=weights,
            asset_names=names,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            method="hrp",
        )

    def black_litterman(
        self,
        returns: np.ndarray,
        views_matrix: np.ndarray,
        views_returns: np.ndarray,
        tau: float = 0.05,
        asset_names: list[str] | None = None,
    ) -> PortfolioAllocation:
        """Black-Litterman Model.

        Combines market equilibrium returns with investor views.

        Args:
            returns: (n_obs, n_assets) historical returns
            views_matrix: (n_views, n_assets) picking matrix P
            views_returns: (n_views,) expected excess returns Q
            tau: Scaling factor for uncertainty of prior (default 0.05)
            asset_names: Asset labels
        """
        n_assets = returns.shape[1]
        names = asset_names or [f"asset_{i}" for i in range(n_assets)]

        cov = np.cov(returns, rowvar=False) * self.trading_days

        # Market-implied equilibrium returns (using equal-weight as proxy for market cap)
        w_market = np.ones(n_assets) / n_assets
        delta = 2.5  # Risk aversion parameter
        pi = delta * cov @ w_market  # Equilibrium excess returns

        # Black-Litterman formula
        P = views_matrix
        Q = views_returns
        Omega = np.diag(np.diag(P @ (tau * cov) @ P.T))  # Uncertainty in views

        # Posterior returns
        tau_cov_inv = np.linalg.inv(tau * cov)
        omega_inv = np.linalg.inv(Omega)

        posterior_cov = np.linalg.inv(tau_cov_inv + P.T @ omega_inv @ P)
        posterior_mu = posterior_cov @ (tau_cov_inv @ pi + P.T @ omega_inv @ Q)

        # Optimize with posterior estimates
        weights = self._max_sharpe(posterior_mu, cov)
        weights = self._clip_weights(weights)

        port_ret = weights @ posterior_mu
        port_vol = np.sqrt(weights @ cov @ weights)
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        return PortfolioAllocation(
            weights=weights,
            asset_names=names,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            method="black_litterman",
            metadata={
                "posterior_returns": posterior_mu.tolist(),
                "equilibrium_returns": pi.tolist(),
            },
        )

    def equal_weight(
        self,
        n_assets: int,
        asset_names: list[str] | None = None,
    ) -> PortfolioAllocation:
        """Simple equal-weight (1/N) portfolio baseline."""
        names = asset_names or [f"asset_{i}" for i in range(n_assets)]
        weights = np.ones(n_assets) / n_assets
        return PortfolioAllocation(
            weights=weights,
            asset_names=names,
            method="equal_weight",
        )

    def inverse_volatility(
        self,
        returns: np.ndarray,
        asset_names: list[str] | None = None,
    ) -> PortfolioAllocation:
        """Inverse volatility weighting."""
        n_assets = returns.shape[1]
        names = asset_names or [f"asset_{i}" for i in range(n_assets)]

        vols = np.std(returns, axis=0)
        vols = np.where(vols > 0, vols, 1e-8)
        inv_vol = 1.0 / vols
        weights = inv_vol / np.sum(inv_vol)

        cov = np.cov(returns, rowvar=False) * self.trading_days
        mu = np.mean(returns, axis=0) * self.trading_days
        port_ret = weights @ mu
        port_vol = np.sqrt(weights @ cov @ weights)

        return PortfolioAllocation(
            weights=weights,
            asset_names=names,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=(port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0,
            method="inverse_volatility",
        )

    # ── Internal helpers ────────────────────────────────────────────

    def _max_sharpe(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Maximum Sharpe ratio portfolio via numerical optimization."""
        n = len(mu)
        try:
            from scipy.optimize import minimize

            def neg_sharpe(w):
                ret = w @ mu
                vol = np.sqrt(w @ cov @ w)
                return -(ret - self.risk_free_rate) / vol if vol > 1e-10 else 0

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
            bounds = [(self.min_weight, self.max_weight)] * n
            x0 = np.ones(n) / n

            result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            return result.x if result.success else np.ones(n) / n
        except ImportError:
            return np.ones(n) / n

    def _min_variance_weights(self, cov: np.ndarray) -> np.ndarray:
        """Minimum variance portfolio weights."""
        n = cov.shape[0]
        try:
            from scipy.optimize import minimize

            def portfolio_var(w):
                return w @ cov @ w

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
            bounds = [(self.min_weight, self.max_weight)] * n
            x0 = np.ones(n) / n

            result = minimize(portfolio_var, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            return result.x if result.success else np.ones(n) / n
        except ImportError:
            # Fallback: inverse variance
            var_diag = np.diag(cov)
            inv_var = 1.0 / np.where(var_diag > 0, var_diag, 1e-8)
            return inv_var / np.sum(inv_var)

    def _risk_parity_weights(self, cov: np.ndarray) -> np.ndarray:
        """Risk parity weights via iterative optimization."""
        n = cov.shape[0]
        try:
            from scipy.optimize import minimize

            def risk_parity_obj(w):
                port_vol = np.sqrt(w @ cov @ w)
                if port_vol < 1e-10:
                    return 1e10
                marginal = cov @ w
                risk_contrib = w * marginal / port_vol
                target = port_vol / n
                return np.sum((risk_contrib - target) ** 2)

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
            bounds = [(0.01, self.max_weight)] * n
            x0 = np.ones(n) / n

            result = minimize(risk_parity_obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            return result.x if result.success else np.ones(n) / n
        except ImportError:
            # Fallback: inverse volatility
            vols = np.sqrt(np.diag(cov))
            inv_vol = 1.0 / np.where(vols > 0, vols, 1e-8)
            return inv_vol / np.sum(inv_vol)

    def _risk_contributions(self, weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Compute risk contribution of each asset."""
        port_vol = np.sqrt(weights @ cov @ weights)
        if port_vol < 1e-10:
            return np.zeros(len(weights))
        marginal = cov @ weights
        return weights * marginal / port_vol

    def _correlation_distance(self, corr: np.ndarray) -> np.ndarray:
        """Distance metric from correlation matrix for HRP."""
        return np.sqrt(0.5 * (1 - corr))

    def _quasi_diag(self, dist: np.ndarray) -> list[int]:
        """Quasi-diagonalization via hierarchical clustering (simplified)."""
        n = dist.shape[0]
        # Simple single-linkage clustering
        order = list(range(n))
        # Sort by average correlation
        corr = 1 - 2 * dist ** 2  # Convert back to correlation
        avg_corr = np.mean(corr, axis=1)
        order.sort(key=lambda i: avg_corr[i])
        return order

    def _recursive_bisection(self, cov: np.ndarray, order: list[int]) -> np.ndarray:
        """HRP recursive bisection."""
        n = len(order)
        weights = np.ones(n)

        def bisect(indices: list[int]):
            if len(indices) <= 1:
                return
            mid = len(indices) // 2
            left = indices[:mid]
            right = indices[mid:]

            # Variance of each cluster
            var_left = self._cluster_var(cov, left)
            var_right = self._cluster_var(cov, right)

            # Allocate inversely proportional to variance
            alpha = 1 - var_left / (var_left + var_right) if (var_left + var_right) > 0 else 0.5

            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= (1 - alpha)

            bisect(left)
            bisect(right)

        bisect(order)
        return weights / np.sum(weights)

    def _cluster_var(self, cov: np.ndarray, indices: list[int]) -> float:
        """Compute variance of an equal-weighted cluster."""
        sub_cov = cov[np.ix_(indices, indices)]
        n = len(indices)
        w = np.ones(n) / n
        return float(w @ sub_cov @ w)

    def _solve_mean_variance(self, mu: np.ndarray, cov: np.ndarray, target: float) -> np.ndarray:
        """Solve for minimum variance at target return."""
        n = len(mu)
        try:
            from scipy.optimize import minimize

            def portfolio_var(w):
                return w @ cov @ w

            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w: w @ mu - target},
            ]
            bounds = [(self.min_weight, self.max_weight)] * n
            x0 = np.ones(n) / n

            result = minimize(portfolio_var, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            return result.x if result.success else np.ones(n) / n
        except ImportError:
            return np.ones(n) / n

    def _clip_weights(self, weights: np.ndarray) -> np.ndarray:
        """Clip and renormalize weights."""
        w = np.clip(weights, self.min_weight, self.max_weight)
        total = np.sum(w)
        if total > 0:
            w = w / total
        return w
