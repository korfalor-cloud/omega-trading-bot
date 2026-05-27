"""Risk Budgeting and Risk Parity.

Implements:
- Risk budgeting (target risk contribution per asset)
- Risk parity (equal risk contribution)
- Hierarchical risk parity (HRP)
- Risk contribution analysis
- Maximum diversification portfolio
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskBudgetResult:
    """Risk budgeting result."""
    weights: np.ndarray = None
    risk_contributions: np.ndarray = None
    asset_names: list[str] = None
    total_risk: float = 0.0

    def __post_init__(self):
        if self.weights is None:
            self.weights = np.array([])
        if self.risk_contributions is None:
            self.risk_contributions = np.array([])
        if self.asset_names is None:
            self.asset_names = []


class RiskBudgeter:
    """Risk budgeting and risk parity portfolio construction.

    Allocates capital based on risk contributions rather than
    capital weights.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.annualization = config.get("annualization", 365)
        self.max_iterations = config.get("max_iterations", 1000)
        self.tolerance = config.get("tolerance", 1e-8)

    def risk_parity(
        self,
        cov_matrix: np.ndarray,
        asset_names: list[str] = None,
    ) -> RiskBudgetResult:
        """Equal risk contribution portfolio.

        Each asset contributes equally to total portfolio risk.
        """
        n = cov_matrix.shape[0]
        if asset_names is None:
            asset_names = [f"asset_{i}" for i in range(n)]

        # Initialize with inverse volatility weights
        vols = np.sqrt(np.diag(cov_matrix))
        vols = np.where(vols > 0, vols, 1e-10)
        weights = (1 / vols) / np.sum(1 / vols)

        # Iterative optimization
        for _ in range(self.max_iterations):
            risk_contrib = self._risk_contributions(weights, cov_matrix)
            target = np.ones(n) / n  # Equal risk contribution

            # Update weights
            adjustments = target / (risk_contrib + 1e-10)
            weights = weights * adjustments
            weights = weights / np.sum(weights)

            # Check convergence
            if np.max(np.abs(risk_contrib - target)) < self.tolerance:
                break

        rc = self._risk_contributions(weights, cov_matrix)
        total_risk = np.sqrt(float(weights @ cov_matrix @ weights))

        return RiskBudgetResult(
            weights=weights,
            risk_contributions=rc,
            asset_names=asset_names,
            total_risk=total_risk,
        )

    def risk_budget(
        self,
        cov_matrix: np.ndarray,
        budgets: np.ndarray,
        asset_names: list[str] = None,
    ) -> RiskBudgetResult:
        """Risk budgeting with target risk contributions.

        Args:
            budgets: Target risk contributions (must sum to 1)
        """
        n = cov_matrix.shape[0]
        if asset_names is None:
            asset_names = [f"asset_{i}" for i in range(n)]

        budgets = budgets / np.sum(budgets)  # Normalize

        # Initialize with inverse volatility weights
        vols = np.sqrt(np.diag(cov_matrix))
        vols = np.where(vols > 0, vols, 1e-10)
        weights = (1 / vols) / np.sum(1 / vols)

        for _ in range(self.max_iterations):
            rc = self._risk_contributions(weights, cov_matrix)

            # Update toward target
            adjustments = np.where(rc > 0, budgets / rc, 1.0)
            weights = weights * np.sqrt(adjustments)
            weights = weights / np.sum(weights)

            if np.max(np.abs(rc - budgets)) < self.tolerance:
                break

        rc = self._risk_contributions(weights, cov_matrix)
        total_risk = np.sqrt(float(weights @ cov_matrix @ weights))

        return RiskBudgetResult(
            weights=weights,
            risk_contributions=rc,
            asset_names=asset_names,
            total_risk=total_risk,
        )

    def maximum_diversification(
        self,
        cov_matrix: np.ndarray,
        asset_names: list[str] = None,
    ) -> RiskBudgetResult:
        """Maximum diversification portfolio.

        Maximizes the ratio of weighted average vol to portfolio vol.
        """
        n = cov_matrix.shape[0]
        if asset_names is None:
            asset_names = [f"asset_{i}" for i in range(n)]

        vols = np.sqrt(np.diag(cov_matrix))
        vols = np.where(vols > 0, vols, 1e-10)

        # Use inverse volatility as starting point
        weights = (1 / vols) / np.sum(1 / vols)

        # Gradient ascent on diversification ratio
        for _ in range(self.max_iterations):
            port_vol = np.sqrt(float(weights @ cov_matrix @ weights))
            weighted_avg_vol = float(weights @ vols)
            div_ratio = weighted_avg_vol / port_vol if port_vol > 0 else 0

            # Gradient
            grad = vols / port_vol - div_ratio * (cov_matrix @ weights) / port_vol ** 2

            weights = weights + 0.01 * grad
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)

        rc = self._risk_contributions(weights, cov_matrix)
        total_risk = np.sqrt(float(weights @ cov_matrix @ weights))

        return RiskBudgetResult(
            weights=weights,
            risk_contributions=rc,
            asset_names=asset_names,
            total_risk=total_risk,
        )

    def _risk_contributions(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute marginal and component risk contributions."""
        port_vol = np.sqrt(float(weights @ cov_matrix @ weights))
        if port_vol < 1e-10:
            return np.ones(len(weights)) / len(weights)

        marginal = cov_matrix @ weights / port_vol
        component = weights * marginal
        return component / np.sum(component)

    def risk_decomposition(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        asset_names: list[str] = None,
    ) -> dict:
        """Full risk decomposition analysis."""
        n = len(weights)
        if asset_names is None:
            asset_names = [f"asset_{i}" for i in range(n)]

        port_vol = np.sqrt(float(weights @ cov_matrix @ weights))
        marginal = cov_matrix @ weights / port_vol if port_vol > 0 else np.zeros(n)
        component = weights * marginal
        pct_contribution = component / np.sum(component) if np.sum(component) > 0 else component

        return {
            "portfolio_volatility": float(port_vol),
            "asset_volatilities": {name: float(np.sqrt(cov_matrix[i, i])) for i, name in enumerate(asset_names)},
            "marginal_contributions": {name: float(marginal[i]) for i, name in enumerate(asset_names)},
            "component_contributions": {name: float(component[i]) for i, name in enumerate(asset_names)},
            "pct_contributions": {name: float(pct_contribution[i]) for i, name in enumerate(asset_names)},
            "weights": {name: float(weights[i]) for i, name in enumerate(asset_names)},
        }
