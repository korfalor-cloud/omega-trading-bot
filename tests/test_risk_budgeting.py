"""Tests for risk budgeting."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.risk.risk_budgeting import RiskBudgeter, RiskBudgetResult


class TestRiskBudgeter:
    @pytest.fixture
    def budgeter(self):
        return RiskBudgeter()

    @pytest.fixture
    def cov_matrix(self):
        # 3-asset covariance matrix
        rng = np.random.default_rng(42)
        n = 3
        A = rng.standard_normal((n, n)) * 0.01
        cov = A @ A.T + np.eye(n) * 0.001  # Positive definite
        return cov

    def test_risk_parity(self, budgeter, cov_matrix):
        result = budgeter.risk_parity(cov_matrix, ["BTC", "ETH", "SOL"])
        assert isinstance(result, RiskBudgetResult)
        assert len(result.weights) == 3
        assert abs(sum(result.weights) - 1.0) < 0.01
        assert all(w >= 0 for w in result.weights)

    def test_risk_parity_equal_contributions(self, budgeter, cov_matrix):
        result = budgeter.risk_parity(cov_matrix)
        rc = result.risk_contributions
        # Risk contributions should be approximately equal
        assert max(rc) - min(rc) < 0.1

    def test_risk_budget_custom(self, budgeter, cov_matrix):
        budgets = np.array([0.5, 0.3, 0.2])
        result = budgeter.risk_budget(cov_matrix, budgets)
        assert len(result.weights) == 3
        assert abs(sum(result.weights) - 1.0) < 0.01

    def test_risk_budget_matches_targets(self, budgeter, cov_matrix):
        budgets = np.array([0.6, 0.3, 0.1])
        result = budgeter.risk_budget(cov_matrix, budgets)
        rc = result.risk_contributions
        # Should approximately match targets
        for i in range(3):
            assert abs(rc[i] - budgets[i]) < 0.15

    def test_maximum_diversification(self, budgeter, cov_matrix):
        result = budgeter.maximum_diversification(cov_matrix, ["BTC", "ETH", "SOL"])
        assert len(result.weights) == 3
        assert abs(sum(result.weights) - 1.0) < 0.01
        assert result.total_risk > 0

    def test_risk_decomposition(self, budgeter, cov_matrix):
        weights = np.array([0.4, 0.35, 0.25])
        decomp = budgeter.risk_decomposition(weights, cov_matrix, ["BTC", "ETH", "SOL"])
        assert "portfolio_volatility" in decomp
        assert "asset_volatilities" in decomp
        assert "pct_contributions" in decomp
        assert decomp["portfolio_volatility"] > 0

    def test_risk_decomposition_sums_to_one(self, budgeter, cov_matrix):
        weights = np.array([0.5, 0.3, 0.2])
        decomp = budgeter.risk_decomposition(weights, cov_matrix)
        pct = list(decomp["pct_contributions"].values())
        assert abs(sum(pct) - 1.0) < 0.01

    def test_two_asset_parity(self, budgeter):
        cov = np.array([[0.04, 0.01], [0.01, 0.01]])
        result = budgeter.risk_parity(cov)
        # Lower vol asset should get higher weight
        assert result.weights[1] > result.weights[0]

    def test_total_risk(self, budgeter, cov_matrix):
        result = budgeter.risk_parity(cov_matrix)
        assert result.total_risk > 0

    def test_custom_config(self):
        budgeter = RiskBudgeter(config={"annualization": 252, "max_iterations": 500})
        assert budgeter.annualization == 252
        assert budgeter.max_iterations == 500
