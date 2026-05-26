"""Market Simulator — The Dream Engine.

Simulates future market scenarios before the agent acts.
Like dreaming about the future to make better decisions.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """A simulated market scenario."""
    id: str
    paths: dict[str, np.ndarray]  # symbol → price path
    probabilities: dict[str, float]  # regime probabilities
    expected_returns: dict[str, float]
    volatilities: dict[str, float]
    correlations: np.ndarray
    horizon_days: int


class MarketSimulator:
    """Generates realistic future market scenarios.

    Methods:
    - Geometric Brownian Motion (basic)
    - Regime-switching model (advanced)
    - GAN-based generator (learned)
    - Monte Carlo with fat tails (Student-t)
    """

    def __init__(self, config: dict):
        self.scenario_count = config.get("scenario_count", 10000)
        self.horizon_days = config.get("simulation_horizon_days", 30)
        self._historical_returns: dict[str, np.ndarray] = {}
        self._historical_vols: dict[str, np.ndarray] = {}

    async def fit(self, price_history: dict[str, np.ndarray]) -> None:
        """Fit simulator on historical price data."""
        for symbol, prices in price_history.items():
            if len(prices) < 2:
                continue
            returns = np.diff(np.log(prices))
            self._historical_returns[symbol] = returns

            # Rolling volatility
            vol_window = 20
            vols = np.array([
                np.std(returns[max(0, i - vol_window):i])
                for i in range(1, len(returns) + 1)
            ])
            self._historical_vols[symbol] = vols

        logger.info(f"Market simulator fitted on {len(price_history)} symbols")

    async def simulate_scenarios(
        self,
        current_prices: dict[str, float],
        horizon_days: Optional[int] = None,
        n_scenarios: Optional[int] = None,
    ) -> list[Scenario]:
        """Generate multiple future scenarios."""
        horizon = horizon_days or self.horizon_days
        n = n_scenarios or self.scenario_count

        scenarios = []
        symbols = list(current_prices.keys())

        for i in range(n):
            scenario = await self._generate_scenario(
                symbols, current_prices, horizon, scenario_id=i
            )
            scenarios.append(scenario)

        return scenarios

    async def _generate_scenario(
        self,
        symbols: list[str],
        current_prices: dict[str, float],
        horizon: int,
        scenario_id: int,
    ) -> Scenario:
        """Generate a single scenario."""
        paths = {}

        for symbol in symbols:
            if symbol not in self._historical_returns:
                # No history, assume flat
                paths[symbol] = np.full(horizon, current_prices.get(symbol, 100))
                continue

            returns = self._historical_returns[symbol]
            vols = self._historical_vols[symbol]

            # Current volatility
            current_vol = vols[-1] if len(vols) > 0 else np.std(returns)

            # Generate returns with fat tails (Student-t)
            df = 5  # degrees of freedom for Student-t
            daily_returns = np.random.standard_t(df, size=horizon) * current_vol

            # Add drift (mean return)
            mean_return = np.mean(returns[-60:]) if len(returns) > 60 else np.mean(returns)
            daily_returns += mean_return

            # Generate price path
            price = current_prices.get(symbol, 100)
            price_path = np.zeros(horizon)
            for t in range(horizon):
                price *= math.exp(daily_returns[t])
                price_path[t] = price

            paths[symbol] = price_path

        # Calculate expected returns and volatilities
        expected_returns = {}
        volatilities = {}
        for symbol in symbols:
            if symbol in paths:
                final_price = paths[symbol][-1]
                start_price = current_prices.get(symbol, 100)
                expected_returns[symbol] = (final_price - start_price) / start_price
                volatilities[symbol] = np.std(np.diff(np.log(paths[symbol])))

        # Correlation matrix
        if len(symbols) > 1:
            returns_matrix = np.column_stack([
                np.diff(np.log(paths[s])) for s in symbols if s in paths
            ])
            if returns_matrix.shape[0] > 1:
                corr = np.corrcoef(returns_matrix.T)
            else:
                corr = np.eye(len(symbols))
        else:
            corr = np.array([[1.0]])

        return Scenario(
            id=f"scenario_{scenario_id}",
            paths=paths,
            probabilities={"base": 1.0 / self.scenario_count},
            expected_returns=expected_returns,
            volatilities=volatilities,
            correlations=corr,
            horizon_days=horizon,
        )

    async def simulate_order_impact(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
        order_book_depth: float,
    ) -> dict:
        """Simulate the market impact of an order."""
        # Almgren-Chriss simplified impact model
        # Temporary impact: g(v) = η * v
        # Permanent impact: h(v) = γ * v

        eta = 0.01  # Temporary impact coefficient
        gamma = 0.001  # Permanent impact coefficient

        participation_rate = quantity / max(order_book_depth, 1)

        temp_impact = eta * participation_rate
        perm_impact = gamma * participation_rate

        total_impact_bps = (temp_impact + perm_impact) * 10000

        slippage_price = current_price * (1 + total_impact_bps / 10000 if side == "buy" else 1 - total_impact_bps / 10000)

        return {
            "temporary_impact_bps": temp_impact * 10000,
            "permanent_impact_bps": perm_impact * 10000,
            "total_impact_bps": total_impact_bps,
            "expected_fill_price": slippage_price,
            "participation_rate": participation_rate,
        }

    async def monte_carlo_var(
        self,
        portfolio_returns: np.ndarray,
        confidence: float = 0.95,
        horizon_days: int = 1,
        n_simulations: int = 10000,
    ) -> dict:
        """Monte Carlo VaR calculation."""
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)

        # Generate scenarios
        simulated_returns = np.random.normal(
            mean_return * horizon_days,
            std_return * math.sqrt(horizon_days),
            n_simulations,
        )

        # Calculate VaR
        var_percentile = np.percentile(simulated_returns, (1 - confidence) * 100)
        cvar = np.mean(simulated_returns[simulated_returns <= var_percentile])

        return {
            "var": float(var_percentile),
            "cvar": float(cvar),
            "mean": float(np.mean(simulated_returns)),
            "std": float(np.std(simulated_returns)),
            "worst_case": float(np.min(simulated_returns)),
            "best_case": float(np.max(simulated_returns)),
        }
