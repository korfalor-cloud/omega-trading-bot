"""Tests for tail risk analysis."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.risk.tail_risk import TailRiskAnalyzer, TailRiskMetrics


class TestTailRiskAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return TailRiskAnalyzer()

    @pytest.fixture
    def returns(self):
        rng = np.random.default_rng(42)
        # Mix of normal and fat-tailed returns
        normal = rng.normal(0.0005, 0.02, 400)
        # Add some extreme losses
        extreme = rng.normal(-0.08, 0.03, 20)
        return np.concatenate([normal, extreme])

    def test_fit_evt(self, analyzer, returns):
        result = analyzer.fit_evt(returns)
        assert isinstance(result, TailRiskMetrics)
        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.expected_shortfall_95 >= result.var_95
        assert result.expected_shortfall_99 >= result.var_99

    def test_tail_index(self, analyzer, returns):
        result = analyzer.fit_evt(returns)
        # Tail index should be reasonable
        assert -1 < result.tail_index < 2

    def test_var_99_9(self, analyzer, returns):
        result = analyzer.fit_evt(returns)
        # var_99_9 may be 0 if empirical fallback is used
        assert result.var_99_9 >= 0

    def test_max_drawdown_distribution(self, analyzer, returns):
        dd_dist = analyzer.max_drawdown_distribution(returns, n_bootstrap=500)
        assert "mean" in dd_dist
        assert "percentile_95" in dd_dist
        assert dd_dist["mean"] > 0
        assert dd_dist["percentile_95"] >= dd_dist["mean"]

    def test_stress_correlation(self, analyzer):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 0.02, 200)
        b = 0.8 * a + rng.normal(0, 0.01, 200)
        stress_corr = analyzer.stress_correlation(a, b, threshold_percentile=0.1)
        assert -1 <= stress_corr <= 1

    def test_insufficient_data(self, analyzer):
        short = np.array([0.01, -0.01])
        result = analyzer.fit_evt(short)
        assert result.var_95 == 0

    def test_all_positive_returns(self, analyzer):
        positive = np.abs(np.random.randn(100)) * 0.01
        result = analyzer.fit_evt(positive)
        # Should handle gracefully
        assert result.var_95 >= 0
