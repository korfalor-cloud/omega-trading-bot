"""Tests for stress testing."""
from __future__ import annotations

import pytest

from tradingbot.risk.stress_testing import StressTester, StressResult


class TestStressTester:
    @pytest.fixture
    def tester(self):
        return StressTester()

    @pytest.fixture
    def portfolio(self):
        positions = {"BTC": 0.5, "ETH": 5.0}
        prices = {"BTC": 60000, "ETH": 3000}
        return positions, prices

    def test_run_scenario(self, tester, portfolio):
        positions, prices = portfolio
        result = tester.run_scenario("market_crash", positions, prices)
        assert isinstance(result, StressResult)
        assert result.portfolio_loss < 0
        assert result.scenario_name == "market_crash"

    def test_run_all_scenarios(self, tester, portfolio):
        positions, prices = portfolio
        results = tester.run_all_scenarios(positions, prices)
        assert len(results) >= 5
        assert all(isinstance(r, StressResult) for r in results)

    def test_worst_case(self, tester, portfolio):
        positions, prices = portfolio
        worst = tester.worst_case(positions, prices)
        assert worst.portfolio_loss < 0
        assert worst.scenario_name == "crypto_winter" or worst.scenario_name == "black_swan"

    def test_sensitivity_analysis(self, tester, portfolio):
        positions, prices = portfolio
        result = tester.sensitivity_analysis(positions, prices)
        assert "shocks" in result
        assert "results" in result
        assert "portfolio" in result["results"]

    def test_add_scenario(self, tester, portfolio):
        positions, prices = portfolio
        tester.add_scenario("custom", {"equity_shock": -0.15})
        result = tester.run_scenario("custom", positions, prices)
        assert result.portfolio_loss < 0

    def test_empty_portfolio(self, tester):
        result = tester.run_scenario("market_crash", {}, {})
        assert result.portfolio_loss == 0

    def test_worst_position(self, tester, portfolio):
        positions, prices = portfolio
        result = tester.run_scenario("market_crash", positions, prices)
        assert result.worst_position in positions

    def test_loss_pct(self, tester, portfolio):
        positions, prices = portfolio
        result = tester.run_scenario("market_crash", positions, prices)
        assert -1 <= result.portfolio_loss_pct <= 0
