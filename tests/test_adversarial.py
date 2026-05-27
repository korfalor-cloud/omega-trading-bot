"""Tests for adversarial stress testing."""
from __future__ import annotations

import pytest

from tradingbot.adversarial.stress_tester import (
    AdversarialTester,
    StressResult,
    StressScenario,
)
from tradingbot.genome.strategy_genome import create_random_genome


class TestAdversarialTester:
    @pytest.fixture
    def tester(self):
        return AdversarialTester({"initial_capital": 100000.0})

    def test_scenario_generation(self, tester):
        for scenario in tester.scenarios:
            bars = tester.generate_scenario_data(scenario)
            assert len(bars) == scenario.duration_bars
            assert all(b.close > 0 for b in bars)

    def test_flash_crash(self, tester):
        scenario = next(s for s in tester.scenarios if s.name == "flash_crash")
        bars = tester.generate_scenario_data(scenario)
        # Should have a significant drop
        prices = [b.close for b in bars]
        max_drop = max((prices[i] - prices[i+1]) / prices[i] for i in range(len(prices)-1))
        assert max_drop > 0.1  # At least 10% drop somewhere

    def test_prolonged_bear(self, tester):
        scenario = next(s for s in tester.scenarios if s.name == "prolonged_bear")
        bars = tester.generate_scenario_data(scenario)
        # End price should be lower than start
        assert bars[-1].close < bars[0].close

    def test_black_swan(self, tester):
        scenario = next(s for s in tester.scenarios if s.name == "black_swan")
        bars = tester.generate_scenario_data(scenario)
        prices = [b.close for b in bars]
        # Should have a single large drop
        min_price = min(prices)
        max_price = max(prices[:len(prices)//2])
        drop = (max_price - min_price) / max_price
        assert drop > 0.2

    @pytest.mark.asyncio
    async def test_stress_test_result(self, tester):
        genome = create_random_genome("stress_test")
        scenario = tester.scenarios[0]
        result = await tester.run_stress_test(genome, scenario)

        assert isinstance(result, StressResult)
        assert result.scenario == scenario.name
        assert 0 <= result.max_drawdown <= 1
        assert isinstance(result.survived, bool)

    @pytest.mark.asyncio
    async def test_run_all_scenarios(self, tester):
        genome = create_random_genome("full_test")
        results = await tester.run_all_scenarios(genome)
        assert len(results) == len(tester.scenarios)
        assert all(isinstance(r, StressResult) for r in results)

    def test_report_generation(self, tester):
        results = [
            StressResult(
                scenario="test",
                max_drawdown=0.15,
                total_return=0.05,
                sharpe_ratio=1.2,
                max_loss_streak=3,
                recovery_time_bars=50,
                survived=True,
            ),
        ]
        report = tester.generate_report(results)
        assert "STRESS TEST REPORT" in report
        assert "test" in report
        assert "PASS" in report
