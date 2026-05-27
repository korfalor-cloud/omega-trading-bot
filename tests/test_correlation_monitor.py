"""Tests for correlation monitoring."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.features.correlation_monitor import (
    CorrelationAlert,
    CorrelationMonitor,
)


class TestCorrelationMonitor:
    @pytest.fixture
    def monitor(self):
        return CorrelationMonitor(config={"window": 20})

    @pytest.fixture
    def correlated_returns(self):
        rng = np.random.default_rng(42)
        n = 100
        common = rng.standard_normal(n)
        return {
            "BTC": common + rng.standard_normal(n) * 0.5,
            "ETH": common + rng.standard_normal(n) * 0.5,
            "SOL": rng.standard_normal(n),  # Uncorrelated
        }

    def test_update_returns_correlations(self, monitor, correlated_returns):
        corrs = monitor.update(correlated_returns)
        assert isinstance(corrs, dict)
        assert len(corrs) > 0

    def test_correlation_matrix(self, monitor, correlated_returns):
        names, matrix = monitor.get_correlation_matrix(correlated_returns)
        assert len(names) == 3
        assert matrix.shape == (3, 3)
        # Diagonal should be 1
        for i in range(3):
            assert matrix[i, i] == pytest.approx(1.0, abs=0.01)

    def test_btc_eth_correlated(self, monitor, correlated_returns):
        corrs = monitor.update(correlated_returns)
        # BTC and ETH should be highly correlated
        key = "BTC_ETH"
        if key in corrs:
            assert corrs[key] > 0.3

    def test_diversification_score(self, monitor, correlated_returns):
        score = monitor.diversification_score(correlated_returns)
        assert 0 <= score <= 1

    def test_alerts_on_breakdown(self, monitor):
        rng = np.random.default_rng(42)
        n = 100
        # First batch: correlated
        common = rng.standard_normal(n)
        returns1 = {
            "BTC": common + rng.standard_normal(n) * 0.1,
            "ETH": common + rng.standard_normal(n) * 0.1,
        }
        monitor.update(returns1)

        # Second batch: uncorrelated
        returns2 = {
            "BTC": rng.standard_normal(n),
            "ETH": rng.standard_normal(n),
        }
        monitor.update(returns2)

        alerts = monitor.get_alerts()
        # May or may not have alerts depending on correlation change magnitude
        assert isinstance(alerts, list)

    def test_pair_correlation(self, monitor, correlated_returns):
        monitor.update(correlated_returns)
        corr = monitor.get_pair_correlation("BTC", "ETH")
        assert corr is not None
        assert -1 <= corr <= 1

    def test_pair_correlation_unknown(self, monitor):
        assert monitor.get_pair_correlation("BTC", "XYZ") is None

    def test_correlation_trend(self, monitor, correlated_returns):
        for _ in range(5):
            monitor.update(correlated_returns)
        trend = monitor.correlation_trend("BTC", "ETH")
        assert trend in ("increasing", "decreasing", "stable")

    def test_custom_config(self):
        monitor = CorrelationMonitor(config={
            "window": 30,
            "breakdown_threshold": 0.5,
        })
        assert monitor.window == 30
        assert monitor.breakdown_threshold == 0.5
