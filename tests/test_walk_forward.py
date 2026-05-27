"""Tests for walk-forward optimization and Monte Carlo simulation."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.backtesting.walk_forward import (
    MonteCarloResult,
    MonteCarloSimulator,
    WalkForwardAnalyzer,
    WalkForwardResult,
)


class TestWalkForwardAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return WalkForwardAnalyzer({"n_splits": 3, "train_ratio": 0.7})

    def test_walk_forward_basic(self, analyzer):
        data = np.random.randn(300)

        def optimize(train_data):
            return {"param": float(np.mean(train_data))}

        def evaluate(data, params):
            return float(np.mean(data) * params["param"])

        result = analyzer.analyze(data, optimize, evaluate)
        assert isinstance(result, WalkForwardResult)
        assert len(result.folds) == 3

    def test_oos_metrics(self, analyzer):
        data = np.random.randn(300)

        def optimize(train_data):
            return {"param": 1.0}

        def evaluate(data, params):
            return float(np.mean(data))

        result = analyzer.analyze(data, optimize, evaluate)
        assert len(result.oos_metrics) == 3
        assert result.std_oos_metric >= 0

    def test_anchored_walk_forward(self):
        analyzer = WalkForwardAnalyzer({"n_splits": 3, "anchored": True})
        data = np.random.randn(300)

        def optimize(train_data):
            return {"param": 1.0}

        def evaluate(data, params):
            return float(np.mean(data))

        result = analyzer.analyze(data, optimize, evaluate)
        assert len(result.folds) > 0
        # First fold should have smallest train set
        # Later folds should have larger train sets (anchored)
        if len(result.folds) > 1:
            assert result.folds[-1].train_end >= result.folds[0].train_end


class TestMonteCarloSimulator:
    @pytest.fixture
    def simulator(self):
        return MonteCarloSimulator({"n_simulations": 1000})

    def test_simulate(self, simulator):
        rng = np.random.default_rng(42)
        trade_returns = rng.normal(0.001, 0.02, 100)
        result = simulator.simulate(trade_returns, trades_per_year=50)

        assert isinstance(result, MonteCarloResult)
        assert result.n_simulations == 1000
        assert result.mean_return != 0
        assert result.prob_positive > 0

    def test_percentiles_ordered(self, simulator):
        rng = np.random.default_rng(42)
        trade_returns = rng.normal(0.001, 0.02, 100)
        result = simulator.simulate(trade_returns, trades_per_year=50)

        assert result.percentile_5 <= result.percentile_25
        assert result.percentile_25 <= result.percentile_50
        assert result.percentile_50 <= result.percentile_75
        assert result.percentile_75 <= result.percentile_95

    def test_max_drawdown(self, simulator):
        rng = np.random.default_rng(42)
        trade_returns = rng.normal(0.001, 0.02, 100)
        result = simulator.simulate(trade_returns, trades_per_year=50)

        assert result.max_drawdown_mean > 0
        assert result.max_drawdown_95 >= result.max_drawdown_mean

    def test_sharpe(self, simulator):
        rng = np.random.default_rng(42)
        trade_returns = rng.normal(0.002, 0.01, 100)  # Positive returns
        result = simulator.simulate(trade_returns, trades_per_year=50)

        assert result.sharpe_mean > 0  # Should be positive for positive returns

    def test_bootstrap_ci(self, simulator):
        values = np.random.randn(100)
        lower, upper = simulator.bootstrap_confidence_interval(values)
        assert lower < upper
        assert lower < np.mean(values) < upper

    def test_generate_report(self, simulator):
        rng = np.random.default_rng(42)
        trade_returns = rng.normal(0.001, 0.02, 100)
        result = simulator.simulate(trade_returns, trades_per_year=50)
        report = simulator.generate_report(result)
        assert "MONTE CARLO" in report
        assert "Simulations" in report

    def test_insufficient_data(self, simulator):
        trade_returns = np.array([0.01, -0.01])
        result = simulator.simulate(trade_returns)
        assert result.n_simulations == 0
