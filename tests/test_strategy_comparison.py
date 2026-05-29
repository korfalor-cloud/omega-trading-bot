"""Tests for StrategyComparator — side-by-side strategy analysis."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.monitoring.strategy_comparison import StrategyComparator, StrategyMetrics


class TestStrategyComparator:
    @pytest.fixture
    def comparator(self):
        c = StrategyComparator()
        # Strategy A: positive returns
        rng = np.random.RandomState(42)
        returns_a = list(rng.normal(0.005, 0.02, 100))
        # Strategy B: slightly negative returns
        returns_b = list(rng.normal(-0.001, 0.03, 100))
        c.add_strategy("momentum", returns_a)
        c.add_strategy("mean_reversion", returns_b)
        return c

    def test_add_strategy(self, comparator):
        assert "momentum" in comparator._strategies
        assert "mean_reversion" in comparator._strategies
        assert len(comparator._strategies["momentum"]) == 100

    def test_compare_returns_metrics(self, comparator):
        results = comparator.compare()
        assert len(results) == 2
        for r in results:
            assert isinstance(r, StrategyMetrics)
            assert r.total_return != 0
            assert r.sharpe != 0
            assert r.max_drawdown >= 0
            assert 0 <= r.win_rate <= 1

    def test_compare_ranked_by_sharpe(self, comparator):
        results = comparator.compare()
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[0].sharpe >= results[1].sharpe

    def test_compare_skips_short_series(self):
        c = StrategyComparator()
        c.add_strategy("short", [0.01])
        results = c.compare()
        assert len(results) == 0

    def test_get_correlation_matrix(self, comparator):
        corr = comparator.get_correlation_matrix()
        assert "momentum" in corr
        assert "mean_reversion" in corr
        assert corr["momentum"]["momentum"] == pytest.approx(1.0)
        assert corr["mean_reversion"]["mean_reversion"] == pytest.approx(1.0)
        # Cross-correlation should be between -1 and 1
        assert -1 <= corr["momentum"]["mean_reversion"] <= 1

    def test_format_comparison(self, comparator):
        text = comparator.format_comparison()
        assert "Strategy Comparison" in text
        assert "momentum" in text
        assert "mean_reversion" in text
        assert "Sharpe" in text or "sharpe" in text.lower()

    def test_single_strategy(self):
        c = StrategyComparator()
        c.add_strategy("only_one", [0.01, 0.02, -0.01, 0.015])
        results = c.compare()
        assert len(results) == 1
        assert results[0].rank == 1

    def test_sortino_computed(self, comparator):
        results = comparator.compare()
        for r in results:
            assert r.sortino != 0

    def test_profit_factor(self, comparator):
        results = comparator.compare()
        # Momentum strategy should have profit factor > 0
        momentum = next(r for r in results if r.strategy_id == "momentum")
        assert momentum.profit_factor > 0
