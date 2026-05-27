"""Tests for cross-asset analysis."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.features.cross_asset import (
    CointegrationResult,
    CorrelationResult,
    CrossAssetAnalyzer,
)


class TestCrossAssetAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return CrossAssetAnalyzer()

    @pytest.fixture
    def correlated_returns(self):
        rng = np.random.default_rng(42)
        n = 200
        common = rng.standard_normal(n)
        a = common + rng.standard_normal(n) * 0.5
        b = common + rng.standard_normal(n) * 0.5
        return a, b

    @pytest.fixture
    def cointegrated_prices(self):
        rng = np.random.default_rng(42)
        n = 200
        # Generate cointegrated pair: shared trend + mean-reverting spread
        trend = np.cumsum(rng.standard_normal(n) * 0.5)
        spread = rng.standard_normal(n) * 0.3
        # Mean-revert the spread
        for i in range(1, len(spread)):
            spread[i] += 0.9 * (spread[i - 1] - spread[i])
        a = 100 + trend + spread
        b = 50 + trend * 0.5 - spread * 0.5
        return a, b

    def test_rolling_correlation(self, analyzer, correlated_returns):
        a, b = correlated_returns
        corr = analyzer.rolling_correlation(a, b, window=20)
        assert len(corr) == len(a)
        # First 19 should be NaN
        assert np.isnan(corr[0])
        assert not np.isnan(corr[20])
        # Should be positive for correlated series
        valid = corr[~np.isnan(corr)]
        assert np.mean(valid) > 0.3

    def test_correlation_matrix(self, analyzer):
        rng = np.random.default_rng(42)
        returns = {
            "BTC": rng.standard_normal(100),
            "ETH": rng.standard_normal(100),
            "SOL": rng.standard_normal(100),
        }
        result = analyzer.correlation_matrix(returns, window=50)
        assert isinstance(result, CorrelationResult)
        assert result.correlation_matrix.shape == (3, 3)
        assert len(result.asset_names) == 3
        # Diagonal should be 1
        for i in range(3):
            assert result.correlation_matrix[i, i] == pytest.approx(1.0, abs=0.01)

    def test_beta(self, analyzer, correlated_returns):
        a, b = correlated_returns
        beta = analyzer.beta(a, b, window=50)
        assert isinstance(beta, float)
        # Correlated assets should have beta near 1
        assert 0.3 < beta < 2.0

    def test_rolling_beta(self, analyzer, correlated_returns):
        a, b = correlated_returns
        rbeta = analyzer.rolling_beta(a, b, window=20)
        assert len(rbeta) == len(a)
        assert np.isnan(rbeta[0])
        assert not np.isnan(rbeta[20])

    def test_cointegration_basic(self, analyzer, cointegrated_prices):
        a, b = cointegrated_prices
        result = analyzer.cointegration_test(a, b)
        assert isinstance(result, CointegrationResult)
        assert result.hedge_ratio != 0
        assert result.half_life > 0

    def test_cointegration_short_series(self, analyzer):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = analyzer.cointegration_test(a, b)
        # Too short — should not crash
        assert isinstance(result, CointegrationResult)

    def test_lead_lag(self, analyzer):
        rng = np.random.default_rng(42)
        n = 200
        # A leads B by 3 bars
        a = rng.standard_normal(n)
        b = np.zeros(n)
        for i in range(3, n):
            b[i] = a[i - 3] * 0.5 + rng.standard_normal(n)[i] * 0.3

        best_lag, corr = analyzer.lead_lag(a, b, max_lag=10)
        # Should detect A leading B (positive lag)
        assert best_lag >= 0
        assert corr > 0

    def test_regime_conditional_correlation(self, analyzer):
        rng = np.random.default_rng(42)
        n = 200
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        # Regime 0: low correlation, Regime 1: high correlation
        regime = np.zeros(n)
        regime[100:] = 1
        b[100:] = a[100:] + rng.standard_normal(100) * 0.1

        result = analyzer.regime_conditional_correlation(a, b, regime)
        assert "regime_0.0" in result
        assert "regime_1.0" in result
        # Regime 1 should have higher correlation
        assert result["regime_1.0"] > result["regime_0.0"]

    def test_custom_lookback(self):
        analyzer = CrossAssetAnalyzer(config={"lookback": 30})
        assert analyzer.default_lookback == 30
