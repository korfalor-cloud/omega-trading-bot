"""Tests for fitness evaluation."""
from __future__ import annotations

import math

import pytest

from tradingbot.population.fitness import FitnessEvaluator, FitnessResult


class TestFitnessEvaluator:
    @pytest.fixture
    def evaluator(self):
        return FitnessEvaluator({
            "sharpe_weight": 0.35,
            "sortino_weight": 0.25,
            "max_dd_weight": 0.20,
            "win_rate_weight": 0.10,
            "stability_weight": 0.10,
            "min_trades": 30,
        })

    def test_insufficient_trades(self, evaluator):
        equity = [100000 + i for i in range(100)]
        returns = [0.01] * 10  # Too few trades
        result = evaluator.evaluate(equity, returns)
        assert result.composite_fitness == -1.0
        assert result.total_trades == 10

    def test_positive_performance(self, evaluator):
        # Steadily increasing equity
        equity = [100000 * (1.001 ** i) for i in range(500)]
        returns = [0.001] * 50
        result = evaluator.evaluate(equity, returns)
        assert result.composite_fitness > 0
        assert result.sharpe_ratio > 0
        assert result.win_rate == 1.0
        assert result.total_return > 0

    def test_negative_performance(self, evaluator):
        # Declining equity
        equity = [100000 * (0.999 ** i) for i in range(500)]
        returns = [-0.001] * 50
        result = evaluator.evaluate(equity, returns)
        # Should be penalized for negative return (composite * 0.5)
        assert result.composite_fitness < 0.5
        assert result.win_rate == 0.0

    def test_max_drawdown_calculation(self, evaluator):
        # Equity goes up then crashes
        equity = [100000]
        for i in range(100):
            equity.append(equity[-1] * 1.01)
        for i in range(50):
            equity.append(equity[-1] * 0.98)

        returns = [0.01] * 80 + [-0.02] * 50
        result = evaluator.evaluate(equity, returns)
        assert result.max_drawdown > 0
        assert result.max_drawdown < 1.0

    def test_sharpe_ratio(self, evaluator):
        # Positive returns with low variance
        returns = [0.001] * 100
        sharpe = evaluator._sharpe_ratio(returns)
        assert sharpe > 0

    def test_sortino_ratio(self, evaluator):
        # All positive returns
        returns = [0.001, 0.002, 0.001, 0.003, 0.001]
        sortino = evaluator._sortino_ratio(returns)
        assert sortino == 10.0  # Capped at 10 for no downside

    def test_profit_factor(self, evaluator):
        # Gross profit (0.06) / Gross loss (0.02) = 3.0
        returns = [0.02, 0.02, -0.01, 0.02, -0.01]
        pf = evaluator._profit_factor(returns)
        assert abs(pf - 3.0) < 0.01

    def test_stability_perfect(self, evaluator):
        # Perfectly linear equity curve
        equity = [100000 + i * 100 for i in range(100)]
        stability = evaluator._stability(equity)
        assert stability > 0.99

    def test_stability_chaotic(self, evaluator):
        # Very noisy equity curve
        import random
        random.seed(42)
        equity = [100000 + random.gauss(0, 10000) for _ in range(100)]
        stability = evaluator._stability(equity)
        assert stability < 0.5

    def test_result_fields(self, evaluator):
        equity = [100000 * (1.001 ** i) for i in range(500)]
        returns = [0.001] * 50
        result = evaluator.evaluate(equity, returns)

        assert isinstance(result, FitnessResult)
        assert hasattr(result, "composite_fitness")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "sortino_ratio")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "stability")
        assert hasattr(result, "total_trades")
        assert hasattr(result, "total_return")
        assert hasattr(result, "profit_factor")
