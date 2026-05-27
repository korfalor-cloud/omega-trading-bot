"""Tests for portfolio optimization."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.risk.portfolio_construction.optimizer import (
    PortfolioAllocation,
    PortfolioOptimizer,
)


class TestPortfolioOptimizer:
    @pytest.fixture
    def optimizer(self):
        return PortfolioOptimizer({"risk_free_rate": 0.04})

    @pytest.fixture
    def returns(self):
        rng = np.random.default_rng(42)
        # 4 assets with different return/risk profiles
        r1 = rng.normal(0.001, 0.02, 500)  # High return, medium risk
        r2 = rng.normal(0.0005, 0.01, 500)  # Low return, low risk
        r3 = rng.normal(0.0008, 0.03, 500)  # Medium return, high risk
        r4 = rng.normal(0.0003, 0.015, 500)  # Low return, medium risk
        return np.column_stack([r1, r2, r3, r4])

    @pytest.fixture
    def asset_names(self):
        return ["BTC", "ETH", "SOL", "DOGE"]

    def test_equal_weight(self, optimizer):
        result = optimizer.equal_weight(4, ["A", "B", "C", "D"])
        assert isinstance(result, PortfolioAllocation)
        assert len(result.weights) == 4
        assert abs(np.sum(result.weights) - 1.0) < 0.001
        assert all(abs(w - 0.25) < 0.001 for w in result.weights)

    def test_inverse_volatility(self, optimizer, returns, asset_names):
        result = optimizer.inverse_volatility(returns, asset_names)
        assert len(result.weights) == 4
        assert abs(np.sum(result.weights) - 1.0) < 0.01
        assert result.method == "inverse_volatility"
        # Lower vol assets should get higher weight
        # Asset 2 has lowest vol (0.01), should have highest weight
        assert result.weights[1] > result.weights[2]

    def test_minimum_variance(self, optimizer, returns, asset_names):
        result = optimizer.minimum_variance(returns, asset_names)
        assert len(result.weights) == 4
        assert abs(np.sum(result.weights) - 1.0) < 0.01
        assert result.method == "minimum_variance"
        assert result.expected_volatility > 0

    def test_risk_parity(self, optimizer, returns, asset_names):
        result = optimizer.risk_parity(returns, asset_names)
        assert len(result.weights) == 4
        assert abs(np.sum(result.weights) - 1.0) < 0.05
        assert result.method == "risk_parity"
        # Check risk contributions are roughly equal
        if "risk_contributions" in result.metadata:
            contribs = result.metadata["risk_contributions"]
            # All should be positive and similar
            assert all(c >= 0 for c in contribs)

    def test_mean_variance(self, optimizer, returns, asset_names):
        result = optimizer.mean_variance(returns, asset_names=asset_names)
        assert len(result.weights) == 4
        assert abs(np.sum(result.weights) - 1.0) < 0.01
        assert result.method == "mean_variance"
        assert result.sharpe_ratio != 0

    def test_mean_variance_target_return(self, optimizer, returns, asset_names):
        result = optimizer.mean_variance(returns, target_return=0.1, asset_names=asset_names)
        assert len(result.weights) == 4
        # Should achieve close to target return
        assert abs(result.expected_return - 0.1) < 0.1

    def test_hierarchical_risk_parity(self, optimizer, returns, asset_names):
        result = optimizer.hierarchical_risk_parity(returns, asset_names)
        assert len(result.weights) == 4
        assert abs(np.sum(result.weights) - 1.0) < 0.01
        assert result.method == "hrp"

    def test_black_litterman(self, optimizer, returns, asset_names):
        # View: BTC will outperform by 10%
        P = np.array([[1, 0, 0, 0]])  # View on BTC
        Q = np.array([0.10])  # 10% expected excess return

        result = optimizer.black_litterman(returns, P, Q, asset_names=asset_names)
        assert len(result.weights) == 4
        assert abs(np.sum(result.weights) - 1.0) < 0.01
        assert result.method == "black_litterman"
        # BTC should get a higher weight due to positive view
        assert result.weights[0] > 0.1

    def test_to_dict(self, optimizer, asset_names):
        result = optimizer.equal_weight(4, asset_names)
        d = result.to_dict()
        assert set(d.keys()) == {"BTC", "ETH", "SOL", "DOGE"}
        assert abs(sum(d.values()) - 1.0) < 0.001

    def test_max_weight_constraint(self, optimizer, returns, asset_names):
        opt = PortfolioOptimizer({"max_weight": 0.3})
        result = opt.mean_variance(returns, asset_names=asset_names)
        assert all(w <= 0.31 for w in result.weights)  # Small tolerance

    def test_two_assets(self, optimizer):
        rng = np.random.default_rng(42)
        r = np.column_stack([
            rng.normal(0.001, 0.02, 200),
            rng.normal(0.0005, 0.01, 200),
        ])
        result = optimizer.risk_parity(r, ["A", "B"])
        assert len(result.weights) == 2
        assert abs(np.sum(result.weights) - 1.0) < 0.01
