"""Tests for correlation analysis."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.risk.correlation import CorrelationAnalyzer, CorrelationResult


class TestCorrelationAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return CorrelationAnalyzer(config={"rolling_window": 20, "breakdown_threshold": 0.3})

    @pytest.fixture
    def correlated_returns(self):
        rng = np.random.default_rng(42)
        n = 100
        common = rng.normal(0, 0.02, n)
        return {
            "BTC": common + rng.normal(0, 0.005, n),
            "ETH": common + rng.normal(0, 0.005, n),
            "SOL": rng.normal(0, 0.02, n),  # Independent
        }

    def test_correlation_matrix(self, analyzer, correlated_returns):
        result = analyzer.correlation_matrix(correlated_returns)
        assert isinstance(result, CorrelationResult)
        assert result.matrix.shape == (3, 3)
        assert len(result.asset_names) == 3
        # Diagonal should be 1
        for i in range(3):
            assert result.matrix[i, i] == pytest.approx(1.0, abs=0.01)

    def test_correlated_assets(self, analyzer, correlated_returns):
        result = analyzer.correlation_matrix(correlated_returns)
        # BTC and ETH should be highly correlated
        btc_idx = result.asset_names.index("BTC")
        eth_idx = result.asset_names.index("ETH")
        assert result.matrix[btc_idx, eth_idx] > 0.8

    def test_independent_assets(self, analyzer, correlated_returns):
        result = analyzer.correlation_matrix(correlated_returns)
        btc_idx = result.asset_names.index("BTC")
        sol_idx = result.asset_names.index("SOL")
        assert abs(result.matrix[btc_idx, sol_idx]) < 0.5

    def test_rolling_correlation(self, analyzer, correlated_returns):
        rolling = analyzer.rolling_correlation(correlated_returns)
        assert len(rolling) > 0
        for pair, corr in rolling.items():
            assert len(corr) == 100

    def test_eigenvalues(self, analyzer, correlated_returns):
        result = analyzer.correlation_matrix(correlated_returns)
        assert len(result.eigenvalues) == 3
        assert all(ev > 0 for ev in result.eigenvalues)

    def test_avg_correlation(self, analyzer, correlated_returns):
        result = analyzer.correlation_matrix(correlated_returns)
        assert -1 <= result.avg_correlation <= 1

    def test_effective_n_assets(self, analyzer, correlated_returns):
        result = analyzer.correlation_matrix(correlated_returns)
        n_eff = analyzer.effective_n_assets(result.matrix)
        # With 2 correlated + 1 independent, effective N should be < 3
        assert 1 <= n_eff <= 3

    def test_detect_breakdown(self, analyzer):
        # Create returns with a correlation breakdown
        rng = np.random.default_rng(42)
        n = 100
        common = rng.normal(0, 0.02, n)
        btc = common + rng.normal(0, 0.005, n)
        eth = common + rng.normal(0, 0.005, n)
        # Make last 10 bars uncorrelated
        btc[-10:] = rng.normal(0, 0.02, 10)
        eth[-10:] = rng.normal(0, 0.02, 10)

        breakdowns = analyzer.detect_breakdown({"BTC": btc, "ETH": eth})
        assert isinstance(breakdowns, list)

    def test_tail_dependency(self, analyzer):
        rng = np.random.default_rng(42)
        n = 200
        a = rng.normal(0, 0.02, n)
        b = a + rng.normal(0, 0.01, n)  # Correlated

        result = analyzer.tail_dependency(a, b, threshold=0.05)
        assert "lower_tail_dependency" in result
        assert "upper_tail_dependency" in result
        assert result["lower_tail_dependency"] > 0

    def test_empty_returns(self, analyzer):
        result = analyzer.correlation_matrix({})
        assert len(result.asset_names) == 0
