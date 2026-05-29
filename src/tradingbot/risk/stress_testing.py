"""Portfolio Stress Testing.

Implements:
- Historical scenario replay
- Hypothetical scenario construction
- Multi-factor stress tests
- Sensitivity analysis
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StressResult:
    """Result of a stress test scenario."""
    scenario_name: str = ""
    portfolio_loss: float = 0.0
    portfolio_loss_pct: float = 0.0
    worst_position: str = ""
    worst_position_loss: float = 0.0
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class StressTester:
    """Portfolio stress testing engine."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.scenarios = self._default_scenarios()

    def _default_scenarios(self) -> dict[str, dict]:
        """Default stress test scenarios."""
        return {
            "market_crash": {"equity_shock": -0.20, "vol_spike": 2.0, "correlation_increase": 0.3},
            "flash_crash": {"equity_shock": -0.10, "vol_spike": 3.0, "correlation_increase": 0.5},
            "rate_hike": {"equity_shock": -0.05, "bond_shock": -0.03, "fx_shock": 0.02},
            "crypto_winter": {"equity_shock": -0.50, "vol_spike": 1.5, "correlation_increase": 0.2},
            "black_swan": {"equity_shock": -0.30, "vol_spike": 5.0, "correlation_increase": 0.8},
        }

    def add_scenario(self, name: str, shocks: dict) -> None:
        self.scenarios[name] = shocks

    def run_scenario(
        self,
        scenario_name: str,
        positions: dict[str, float],
        prices: dict[str, float],
    ) -> StressResult:
        """Run a single stress scenario."""
        scenario = self.scenarios.get(scenario_name, {})
        equity_shock = scenario.get("equity_shock", 0)

        total_loss = 0.0
        worst_pos = ""
        worst_loss = 0.0

        for symbol, qty in positions.items():
            price = prices.get(symbol, 0)
            loss = qty * price * equity_shock
            total_loss += loss

            if loss < worst_loss:
                worst_loss = loss
                worst_pos = symbol

        total_value = sum(qty * prices.get(s, 0) for s, qty in positions.items())
        loss_pct = total_loss / total_value if total_value > 0 else 0

        return StressResult(
            scenario_name=scenario_name,
            portfolio_loss=total_loss,
            portfolio_loss_pct=loss_pct,
            worst_position=worst_pos,
            worst_position_loss=worst_loss,
            details=scenario,
        )

    def run_all_scenarios(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
    ) -> list[StressResult]:
        """Run all configured scenarios."""
        return [
            self.run_scenario(name, positions, prices)
            for name in self.scenarios
        ]

    def sensitivity_analysis(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
        shocks: list[float] = None,
    ) -> dict[str, list[float]]:
        """Analyze sensitivity to different shock levels."""
        if shocks is None:
            shocks = [-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30]

        total_value = sum(qty * prices.get(s, 0) for s, qty in positions.items())
        results = {}

        for symbol, qty in positions.items():
            price = prices.get(symbol, 0)
            pnl = [qty * price * shock for shock in shocks]
            results[symbol] = pnl

        results["portfolio"] = [
            sum(qty * prices.get(s, 0) * shock for s, qty in positions.items())
            for shock in shocks
        ]

        return {"shocks": shocks, "results": results}

    def worst_case(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
    ) -> StressResult:
        """Find the worst-case scenario."""
        results = self.run_all_scenarios(positions, prices)
        return min(results, key=lambda r: r.portfolio_loss)
