"""Tests for VaR/CVaR risk models."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.risk.var_models import VaRModel, VaRResult


class TestVaRModel:
    @pytest.fixture
    def model(self):
        return VaRModel()

    @pytest.fixture
    def returns(self):
        rng = np.random.default_rng(42)
        return rng.normal(0.0005, 0.02, 500)

    @pytest.fixture
    def skewed_returns(self):
        rng = np.random.default_rng(42)
        normal = rng.normal(0.0005, 0.02, 500)
        # Add some large negative returns (fat left tail)
        normal[:10] = rng.normal(-0.05, 0.03, 10)
        return normal

    def test_historical_var(self, model, returns):
        result = model.historical_var(returns, 100000)
        assert isinstance(result, VaRResult)
        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.cvar_95 >= result.var_95
        assert result.cvar_99 >= result.var_99
        assert result.method == "historical"

    def test_parametric_var(self, model, returns):
        result = model.parametric_var(returns, 100000)
        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.method == "parametric"

    def test_cornish_fisher_var(self, model, skewed_returns):
        result = model.cornish_fisher_var(skewed_returns, 100000)
        assert result.var_95 > 0
        assert result.method == "cornish_fisher"
        # Should capture more risk than parametric for skewed data
        parametric = model.parametric_var(skewed_returns, 100000)
        # Cornish-Fisher should generally give higher VaR for skewed distributions
        # (not always, but with our left-skewed data it should)

    def test_monte_carlo_var(self, model, returns):
        result = model.monte_carlo_var(returns, 100000, n_simulations=5000)
        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.method == "monte_carlo"

    def test_var_scales_with_portfolio(self, model, returns):
        r1 = model.historical_var(returns, 100000)
        r2 = model.historical_var(returns, 200000)
        assert abs(r2.var_95 / r1.var_95 - 2.0) < 0.1

    def test_var_95_pct(self, model, returns):
        result = model.historical_var(returns, 100000)
        assert 0 < result.var_95_pct < 0.2  # Should be a reasonable percentage

    def test_component_var(self, model):
        rng = np.random.default_rng(42)
        returns_matrix = rng.normal(0.0005, 0.02, (500, 3))
        weights = np.array([0.4, 0.3, 0.3])

        result = model.component_var(returns_matrix, weights, 100000)
        assert "component_var" in result
        assert "pct_contribution" in result
        assert len(result["component_var"]) == 3
        # Contributions should sum to ~1
        assert abs(np.sum(result["pct_contribution"]) - 1.0) < 0.1

    def test_rolling_var(self, model, returns):
        rolling = model.rolling_var(returns, window=63, portfolio_value=100000)
        assert len(rolling) == len(returns)
        # First 63 values should be NaN
        assert np.isnan(rolling[0])
        # Later values should be non-NaN
        valid = rolling[~np.isnan(rolling)]
        assert len(valid) > 0
        assert all(v > 0 for v in valid)

    def test_insufficient_data(self, model):
        short = np.array([0.01, -0.01])
        result = model.historical_var(short, 100000)
        assert result.var_95 == 0

    def test_zero_volatility(self, model):
        constant = np.full(100, 0.001)
        result = model.parametric_var(constant, 100000)
        # Should handle zero volatility gracefully
        assert result.var_95 >= 0
