"""Tests for backtesting analytics."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.backtesting.analytics import BacktestAnalytics, TradeAnalysis, EquityAnalysis


class TestBacktestAnalytics:
    @pytest.fixture
    def analytics(self):
        return BacktestAnalytics(annualization=365)

    @pytest.fixture
    def sample_trades(self):
        return [
            TradeAnalysis(entry_price=100, exit_price=110, pnl=10, pnl_pct=0.1, mae=-2, mfe=12, bars_held=5, side="buy"),
            TradeAnalysis(entry_price=110, exit_price=105, pnl=-5, pnl_pct=-0.045, mae=-8, mfe=3, bars_held=3, side="buy"),
            TradeAnalysis(entry_price=105, exit_price=115, pnl=10, pnl_pct=0.095, mae=-1, mfe=11, bars_held=7, side="buy"),
            TradeAnalysis(entry_price=115, exit_price=120, pnl=5, pnl_pct=0.043, mae=-3, mfe=6, bars_held=2, side="buy"),
        ]

    @pytest.fixture
    def equity_curve(self):
        # Simulated equity curve with some drawdowns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        equity = 100000 * np.cumprod(1 + returns)
        return equity

    def test_analyze_trades(self, analytics, sample_trades):
        result = analytics.analyze_trades(sample_trades)
        assert result["n_trades"] == 4
        assert result["win_rate"] == 0.75
        assert result["avg_pnl"] > 0

    def test_analyze_trades_empty(self, analytics):
        result = analytics.analyze_trades([])
        assert result == {}

    def test_profit_factor(self, analytics, sample_trades):
        result = analytics.analyze_trades(sample_trades)
        assert result["profit_factor"] > 1  # Net profitable

    def test_mae_mfe(self, analytics, sample_trades):
        result = analytics.analyze_trades(sample_trades)
        assert result["avg_mae"] < 0  # Adverse is negative
        assert result["avg_mfe"] > 0  # Favorable is positive

    def test_analyze_equity(self, analytics, equity_curve):
        result = analytics.analyze_equity(equity_curve)
        assert isinstance(result, EquityAnalysis)
        assert result.total_return != 0
        assert result.max_drawdown >= 0
        assert result.max_drawdown <= 1

    def test_sharpe_ratio(self, analytics, equity_curve):
        result = analytics.analyze_equity(equity_curve)
        assert isinstance(result.sharpe_ratio, float)

    def test_sortino_ratio(self, analytics, equity_curve):
        result = analytics.analyze_equity(equity_curve)
        assert isinstance(result.sortino_ratio, float)

    def test_calmar_ratio(self, analytics, equity_curve):
        result = analytics.analyze_equity(equity_curve)
        assert isinstance(result.calmar_ratio, float)

    def test_rolling_metrics(self, analytics, equity_curve):
        result = analytics.rolling_metrics(equity_curve, window=20)
        assert "rolling_return" in result
        assert "rolling_volatility" in result
        assert "rolling_sharpe" in result
        assert len(result["rolling_return"]) == len(equity_curve) - 1

    def test_rolling_metrics_short_data(self, analytics):
        result = analytics.rolling_metrics(np.array([100, 101, 102]), window=30)
        assert result == {}

    def test_return_distribution(self, analytics, equity_curve):
        result = analytics.return_distribution(equity_curve)
        assert "mean" in result
        assert "std" in result
        assert "percentiles" in result
        assert "p5" in result["percentiles"]

    def test_drawdown_duration(self, analytics):
        # Create equity with known drawdown
        equity = np.array([100, 105, 100, 95, 90, 85, 90, 95, 100, 110])
        result = analytics.analyze_equity(equity)
        assert result.max_drawdown_duration > 0

    def test_n_drawdowns(self, analytics, equity_curve):
        result = analytics.analyze_equity(equity_curve)
        assert result.n_drawdowns >= 0
