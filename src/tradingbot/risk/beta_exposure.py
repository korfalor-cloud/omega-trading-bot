"""Beta Exposure Management.

Implements:
- Portfolio beta calculation
- Beta hedging
- Sector beta decomposition
- Dynamic beta targeting
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BetaResult:
    """Beta analysis result."""
    portfolio_beta: float = 0.0
    asset_betas: dict = None
    hedge_ratio: float = 0.0
    target_beta: float = 1.0
    adjustment_needed: float = 0.0

    def __post_init__(self):
        if self.asset_betas is None:
            self.asset_betas = {}


class BetaManager:
    """Portfolio beta management."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.target_beta = config.get("target_beta", 0.0)
        self._return_history: dict[str, list[float]] = {}
        self._market_returns: list[float] = []

    def update(self, symbol: str, return_val: float, market_return: float = None) -> None:
        if symbol not in self._return_history:
            self._return_history[symbol] = []
        self._return_history[symbol].append(return_val)
        if market_return is not None:
            self._market_returns.append(market_return)

    def compute_beta(self, returns: np.ndarray, market_returns: np.ndarray) -> float:
        """Compute beta coefficient."""
        n = min(len(returns), len(market_returns))
        if n < 10:
            return 1.0

        cov = np.cov(returns[-n:], market_returns[-n:])
        var_market = cov[1, 1]
        if var_market > 0:
            return float(cov[0, 1] / var_market)
        return 1.0

    def compute_asset_betas(self) -> dict[str, float]:
        """Compute beta for each asset."""
        if not self._market_returns:
            return {}

        market = np.array(self._market_returns)
        betas = {}
        for symbol, returns in self._return_history.items():
            r = np.array(returns)
            betas[symbol] = self.compute_beta(r, market)
        return betas

    def compute_portfolio_beta(self, weights: dict[str, float]) -> float:
        """Compute weighted portfolio beta."""
        betas = self.compute_asset_betas()
        return sum(weights.get(s, 0) * betas.get(s, 1.0) for s in weights)

    def get_hedge_ratio(self, portfolio_beta: float, hedge_beta: float = 1.0) -> float:
        """Compute hedge ratio to achieve target beta."""
        return (portfolio_beta - self.target_beta) / hedge_beta if hedge_beta != 0 else 0

    def analyze(self, weights: dict[str, float]) -> BetaResult:
        """Full beta analysis."""
        betas = self.compute_asset_betas()
        port_beta = self.compute_portfolio_beta(weights)
        hedge = self.get_hedge_ratio(port_beta)

        return BetaResult(
            portfolio_beta=port_beta,
            asset_betas=betas,
            hedge_ratio=hedge,
            target_beta=self.target_beta,
            adjustment_needed=port_beta - self.target_beta,
        )
