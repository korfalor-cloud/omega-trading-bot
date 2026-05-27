"""Tests for performance reporting."""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from tradingbot.monitoring.performance_report import (
    PerformanceMetrics,
    PerformanceReporter,
    TradeRecord,
)


class TestPerformanceReporter:
    @pytest.fixture
    def reporter(self):
        return PerformanceReporter()

    @pytest.fixture
    def sample_trades(self):
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        trades = [
            TradeRecord(
                symbol="BTC/USDT", side="buy",
                entry_price=50000, exit_price=51000, quantity=0.1,
                entry_time=base_time, exit_time=base_time + timedelta(hours=4),
                pnl=100, pnl_pct=0.02, fees=5, strategy_id="trend",
            ),
            TradeRecord(
                symbol="ETH/USDT", side="buy",
                entry_price=3000, exit_price=2900, quantity=1.0,
                entry_time=base_time + timedelta(hours=5),
                exit_time=base_time + timedelta(hours=8),
                pnl=-100, pnl_pct=-0.033, fees=3, strategy_id="mean_rev",
            ),
            TradeRecord(
                symbol="BTC/USDT", side="sell",
                entry_price=51000, exit_price=50500, quantity=0.1,
                entry_time=base_time + timedelta(hours=10),
                exit_time=base_time + timedelta(hours=12),
                pnl=50, pnl_pct=0.01, fees=4, strategy_id="trend",
            ),
        ]
        return trades

    def test_add_trade(self, reporter, sample_trades):
        for t in sample_trades:
            reporter.add_trade(t)
        assert len(reporter._trades) == 3

    def test_compute_metrics(self, reporter, sample_trades):
        for t in sample_trades:
            reporter.add_trade(t)
        m = reporter.compute_metrics()
        assert isinstance(m, PerformanceMetrics)
        assert m.total_trades == 3
        assert m.winning_trades == 2
        assert m.losing_trades == 1
        assert m.win_rate == pytest.approx(2 / 3, abs=0.01)

    def test_profit_factor(self, reporter, sample_trades):
        for t in sample_trades:
            reporter.add_trade(t)
        m = reporter.compute_metrics()
        # Gross profit: 150, Gross loss: 100, PF = 1.5
        assert m.profit_factor == pytest.approx(1.5, abs=0.1)

    def test_generate_report(self, reporter, sample_trades):
        for t in sample_trades:
            reporter.add_trade(t)
        report = reporter.generate_report()
        assert "PERFORMANCE REPORT" in report
        assert "Sharpe Ratio" in report
        assert "Win Rate" in report

    def test_strategy_breakdown(self, reporter, sample_trades):
        for t in sample_trades:
            reporter.add_trade(t)
        breakdown = reporter.get_strategy_breakdown()
        assert "trend" in breakdown
        assert "mean_rev" in breakdown
        assert breakdown["trend"].total_trades == 2

    def test_symbol_breakdown(self, reporter, sample_trades):
        for t in sample_trades:
            reporter.add_trade(t)
        breakdown = reporter.get_symbol_breakdown()
        assert "BTC/USDT" in breakdown
        assert "ETH/USDT" in breakdown

    def test_empty_reporter(self, reporter):
        m = reporter.compute_metrics()
        assert m.total_trades == 0
        report = reporter.generate_report()
        assert "PERFORMANCE REPORT" in report

    def test_equity_curve(self, reporter):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        reporter.add_equity_point(base, 100000)
        reporter.add_equity_point(base + timedelta(days=1), 101000)
        reporter.add_equity_point(base + timedelta(days=2), 99000)
        curve = reporter.get_equity_curve()
        assert len(curve) == 3

    def test_total_fees(self, reporter, sample_trades):
        for t in sample_trades:
            reporter.add_trade(t)
        m = reporter.compute_metrics()
        assert m.total_fees == 12  # 5 + 3 + 4

    def test_best_worst_trade(self, reporter, sample_trades):
        for t in sample_trades:
            reporter.add_trade(t)
        m = reporter.compute_metrics()
        assert m.best_trade == 100
        assert m.worst_trade == -100
