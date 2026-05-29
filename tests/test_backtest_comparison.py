"""Tests for BacktestComparator — compare multiple backtest runs."""
from __future__ import annotations

import pytest

from tradingbot.monitoring.backtest_comparison import BacktestComparator, BacktestRun


class TestBacktestComparator:
    @pytest.fixture
    def comparator(self):
        c = BacktestComparator()
        c.add_run(BacktestRun(
            run_id="run_001", params={"rsi_period": 14, "threshold": 30},
            total_return=0.15, sharpe=1.5, max_drawdown=0.05, win_rate=0.6, total_trades=50,
        ))
        c.add_run(BacktestRun(
            run_id="run_002", params={"rsi_period": 14, "threshold": 40},
            total_return=0.10, sharpe=1.2, max_drawdown=0.08, win_rate=0.55, total_trades=40,
        ))
        c.add_run(BacktestRun(
            run_id="run_003", params={"rsi_period": 21, "threshold": 30},
            total_return=0.20, sharpe=2.0, max_drawdown=0.04, win_rate=0.65, total_trades=60,
        ))
        return c

    def test_add_run(self, comparator):
        assert len(comparator._runs) == 3

    def test_get_best_by_sharpe(self, comparator):
        best = comparator.get_best("sharpe")
        assert best.run_id == "run_003"
        assert best.sharpe == 2.0

    def test_get_best_by_return(self, comparator):
        best = comparator.get_best("total_return")
        assert best.run_id == "run_003"

    def test_get_worst_by_sharpe(self, comparator):
        worst = comparator.get_worst("sharpe")
        assert worst.run_id == "run_002"
        assert worst.sharpe == 1.2

    def test_get_best_empty(self):
        c = BacktestComparator()
        best = c.get_best()
        assert best.sharpe == 0

    def test_get_worst_empty(self):
        c = BacktestComparator()
        worst = c.get_worst()
        assert worst.sharpe == 0

    def test_get_summary(self, comparator):
        summary = comparator.get_summary()
        assert summary["n_runs"] == 3
        assert summary["best_sharpe"] == 2.0
        assert summary["worst_sharpe"] == 1.2
        assert summary["avg_sharpe"] > 0
        assert summary["std_sharpe"] >= 0

    def test_get_summary_empty(self):
        c = BacktestComparator()
        assert c.get_summary() == {}

    def test_sensitivity_analysis(self, comparator):
        sensitivity = comparator.sensitivity_analysis("rsi_period")
        assert 14 in sensitivity
        assert 21 in sensitivity
        # For rsi_period=14, there are 2 runs
        assert len(sensitivity[14]) == 2  # [mean, std]

    def test_sensitivity_analysis_missing_param(self, comparator):
        sensitivity = comparator.sensitivity_analysis("nonexistent_param")
        assert sensitivity == {}

    def test_format_comparison(self, comparator):
        text = comparator.format_comparison()
        assert "Backtest Comparison" in text
        assert "run_003" in text
        assert "Avg Sharpe" in text

    def test_backtest_run_defaults(self):
        run = BacktestRun()
        assert run.params == {}
        assert run.total_return == 0.0
        assert run.run_id == ""
