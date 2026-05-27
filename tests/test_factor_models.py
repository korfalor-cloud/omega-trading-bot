"""Tests for factor models."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.risk.factor_models import FactorModel, FactorResult, PCAResult


class TestFactorModel:
    @pytest.fixture
    def model(self):
        return FactorModel()

    @pytest.fixture
    def market_returns(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal(200) * 0.015

    @pytest.fixture
    def correlated_returns(self, market_returns):
        rng = np.random.default_rng(123)
        return market_returns * 1.2 + rng.standard_normal(len(market_returns)) * 0.008

    def test_capm(self, model, correlated_returns, market_returns):
        result = model.capm(correlated_returns, market_returns)
        assert isinstance(result, FactorResult)
        assert "market" in result.betas
        assert result.betas["market"] > 0.5  # Should be close to 1.2
        assert result.r_squared > 0.5

    def test_capm_beta_near_true(self, model, correlated_returns, market_returns):
        result = model.capm(correlated_returns, market_returns)
        # True beta is 1.2
        assert abs(result.betas["market"] - 1.2) < 0.3

    def test_fama_french_3(self, model, correlated_returns, market_returns):
        rng = np.random.default_rng(456)
        n = len(market_returns)
        smb = rng.standard_normal(n) * 0.01
        hml = rng.standard_normal(n) * 0.01

        result = model.fama_french_3(correlated_returns, market_returns, smb, hml)
        assert isinstance(result, FactorResult)
        assert "market" in result.betas
        assert "smb" in result.betas
        assert "hml" in result.betas
        assert result.r_squared > 0

    def test_pca(self, model):
        rng = np.random.default_rng(42)
        n, p = 200, 5
        # Generate correlated data
        factors = rng.standard_normal((n, 2))
        loadings = rng.standard_normal((2, p))
        data = factors @ loadings + rng.standard_normal((n, p)) * 0.1

        result = model.pca(data)
        assert isinstance(result, PCAResult)
        assert len(result.eigenvalues) == p
        assert len(result.explained_variance_ratio) == p
        assert result.n_components_90 >= 1
        # First component should explain most variance
        assert result.explained_variance_ratio[0] > 0.3

    def test_pca_n_components(self, model):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 5))
        result = model.pca(data, n_components=2)
        assert len(result.eigenvalues) == 2
        assert result.eigenvectors.shape == (5, 2)

    def test_factor_exposure(self, model, correlated_returns, market_returns):
        factors = {"market": market_returns}
        exposures = model.factor_exposure(correlated_returns, factors)
        assert "market" in exposures
        assert exposures["market"] > 0

    def test_risk_decomposition(self, model):
        betas = {"market": 1.2}
        factor_cov = np.array([[0.000225]])  # 1.5% daily vol squared
        residual_vol = 0.008 * np.sqrt(365)

        result = model.risk_decomposition(betas, factor_cov, residual_vol)
        assert "total_risk" in result
        assert "factor_risk" in result
        assert "residual_risk" in result
        assert result["total_risk"] > 0

    def test_tracking_error(self, model, correlated_returns, market_returns):
        te = model.tracking_error(correlated_returns, market_returns)
        assert te > 0

    def test_information_ratio(self, model, correlated_returns, market_returns):
        ir = model.information_ratio(correlated_returns, market_returns)
        assert isinstance(ir, float)

    def test_information_ratio_identical(self, model, market_returns):
        ir = model.information_ratio(market_returns, market_returns)
        assert abs(ir) < 0.01  # Should be ~0

    def test_residual_vol(self, model, correlated_returns, market_returns):
        result = model.capm(correlated_returns, market_returns)
        assert result.residual_vol > 0

    def test_custom_config(self):
        model = FactorModel(config={"risk_free_rate": 0.05, "annualization": 252})
        assert model.risk_free_rate == 0.05
        assert model.annualization == 252
