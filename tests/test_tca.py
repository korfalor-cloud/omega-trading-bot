"""Tests for transaction cost analysis."""
from __future__ import annotations

import pytest

from tradingbot.data.tca import TransactionCostAnalyzer, TCAResult


class TestTransactionCostAnalyzer:
    @pytest.fixture
    def tca(self):
        return TransactionCostAnalyzer()

    def test_pre_trade_estimate(self, tca):
        result = tca.pre_trade_estimate(50000, 1.0, 1000.0, 0.02, 2.0, "buy")
        assert isinstance(result, TCAResult)
        assert result.total_cost > 0
        assert result.market_impact > 0
        assert result.spread_cost > 0
        assert result.commission > 0

    def test_pre_trade_zero_volume(self, tca):
        result = tca.pre_trade_estimate(50000, 1.0, 0, 0.02, 2.0, "buy")
        assert result.market_impact == 0

    def test_post_trade_buy(self, tca):
        result = tca.post_trade_analysis(
            decision_price=50000, avg_exec_price=50025, quantity=1.0,
            vwap=50010, arrival_price=50005, side="buy", commission=50,
        )
        assert result.total_cost > 0
        assert result.vs_vwap > 0  # Paid more than VWAP

    def test_post_trade_sell(self, tca):
        result = tca.post_trade_analysis(
            decision_price=50000, avg_exec_price=49975, quantity=1.0,
            vwap=49990, arrival_price=49995, side="sell", commission=50,
        )
        assert result.total_cost > 0

    def test_analyze_fills(self, tca):
        fills = [
            {"price": 50010, "quantity": 0.5, "fee": 25},
            {"price": 50020, "quantity": 0.5, "fee": 25},
        ]
        result = tca.analyze_fills(fills, 50000, "buy", 50015)
        assert result.total_cost > 0
        assert result.commission == 50

    def test_analyze_fills_empty(self, tca):
        result = tca.analyze_fills([], 50000, "buy")
        assert result.total_cost == 0

    def test_implementation_shortfall_pct(self, tca):
        result = tca.post_trade_analysis(
            decision_price=50000, avg_exec_price=50050, quantity=1.0,
            vwap=50025, arrival_price=50020, side="buy", commission=50,
        )
        assert result.implementation_shortfall > 0
        assert result.implementation_shortfall < 0.01  # < 1%
