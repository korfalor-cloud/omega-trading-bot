"""Tests for slippage models."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.execution.slippage import SlippageEstimate, SlippageModel


class TestSlippageModel:
    @pytest.fixture
    def model(self):
        return SlippageModel(config={"impact_coeff": 0.1, "spread_bps": 2.0})

    def test_linear_slippage_buy(self, model):
        result = model.linear_slippage(50000, 1.0, 100.0, "buy")
        assert result.slippage_pct > 0
        assert result.fill_price > 50000
        assert result.method == "linear"

    def test_linear_slippage_sell(self, model):
        result = model.linear_slippage(50000, 1.0, 100.0, "sell")
        assert result.fill_price < 50000

    def test_larger_order_more_slippage(self, model):
        small = model.linear_slippage(50000, 1.0, 1000.0, "buy")
        large = model.linear_slippage(50000, 10.0, 1000.0, "buy")
        assert large.slippage_pct > small.slippage_pct

    def test_sqrt_impact(self, model):
        result = model.sqrt_impact(50000, 1.0, 100.0, 0.02, "buy")
        assert result.slippage_pct > 0
        assert result.method == "sqrt"

    def test_sqrt_zero_volume(self, model):
        result = model.sqrt_impact(50000, 1.0, 0, 0.02, "buy")
        assert result.fill_price == 50000

    def test_historical_slippage(self, model):
        expected = np.array([100, 101, 102, 103, 104])
        actual = np.array([100.1, 101.2, 101.9, 103.1, 103.8])
        sides = np.array(["buy", "buy", "sell", "buy", "sell"])

        result = model.historical_slippage(expected, actual, sides)
        assert result["n_samples"] == 5
        assert "avg_slippage" in result
        assert "median_slippage" in result

    def test_estimate_cost(self, model):
        result = model.estimate_cost(50000, 1.0, 100.0, 0.02, "buy", 0.001)
        assert result["notional"] == 50000
        assert result["total_cost"] > 0
        assert result["cost_pct"] > 0

    def test_slippage_estimate_fields(self, model):
        result = model.linear_slippage(50000, 1.0, 100.0, "buy")
        assert isinstance(result, SlippageEstimate)
        assert hasattr(result, "market_impact")
        assert hasattr(result, "spread_cost")
