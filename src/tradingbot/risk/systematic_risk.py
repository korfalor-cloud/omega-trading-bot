"""Systematic Risk — beta exposure and market risk management.

Implements:
- Market beta calculation for individual assets and portfolio
- Systematic risk decomposition (market vs idiosyncratic)
- Hedge ratio computation to target desired beta
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AssetBeta:
    """Beta analysis for a single asset."""
    symbol: str = ""
    beta: float = 0.0
    r_squared: float = 0.0
    systematic_risk_pct: float = 0.0
    idiosyncratic_risk_pct: float = 0.0
    volatility: float = 0.0


@dataclass
class SystematicRiskReport:
    """Systematic risk decomposition report."""
    portfolio_beta: float = 0.0
    portfolio_systematic_var: float = 0.0
    portfolio_total_var: float = 0.0
    systematic_risk_pct: float = 0.0
    idiosyncratic_risk_pct: float = 0.0
    asset_betas: list[AssetBeta] = field(default_factory=list)
    hedge_ratio: float = 0.0
    target_beta: float = 0.0
    hedge_notional: float = 0.0
    risk_level: str = "low"


class SystematicRiskManager:
    """Beta exposure and systematic risk management."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.target_beta = config.get("target_beta", 0.0)
        self.risk_free_rate = config.get("risk_free_rate", 0.04)  # annualized
        self._return_history: dict[str, list[float]] = {}
        self._market_returns: list[float] = []
        self._positions: dict[str, float] = {}  # symbol -> notional weight

    def update(self, symbol: str, asset_return: float, market_return: float) -> None:
        """Update with new return data."""
        if symbol not in self._return_history:
            self._return_history[symbol] = []
        self._return_history[symbol].append(asset_return)
        self._market_returns.append(market_return)

    def set_position(self, symbol: str, weight: float) -> None:
        """Set position weight for a symbol."""
        self._positions[symbol] = weight

    def compute_beta(self, returns: np.ndarray, market_returns: np.ndarray) -> tuple[float, float]:
        """Compute beta and R-squared for an asset.

        Returns: (beta, r_squared)
        """
        n = min(len(returns), len(market_returns))
        if n < 20:
            return 1.0, 0.0

        r = returns[-n:]
        m = market_returns[-n:]

        cov_matrix = np.cov(r, m)
        market_var = cov_matrix[1, 1]

        if market_var == 0:
            return 1.0, 0.0

        beta = cov_matrix[0, 1] / market_var

        # R-squared: proportion of variance explained by market
        total_var = cov_matrix[0, 0]
        systematic_var = beta ** 2 * market_var
        r_squared = systematic_var / total_var if total_var > 0 else 0.0

        return float(beta), float(np.clip(r_squared, 0.0, 1.0))

    def compute_asset_betas(self) -> list[AssetBeta]:
        """Compute beta for all tracked assets."""
        if not self._market_returns:
            return []

        market = np.array(self._market_returns)
        results = []

        for symbol, returns in self._return_history.items():
            r = np.array(returns)
            beta, r_sq = self.compute_beta(r, market)

            # Decompose risk
            total_vol = float(np.std(r)) if len(r) > 1 else 0.0
            systematic_pct = r_sq
            idiosyncratic_pct = 1.0 - r_sq

            results.append(AssetBeta(
                symbol=symbol,
                beta=round(beta, 4),
                r_squared=round(r_sq, 4),
                systematic_risk_pct=round(systematic_pct, 4),
                idiosyncratic_risk_pct=round(idiosyncratic_pct, 4),
                volatility=round(total_vol, 6),
            ))

        return results

    def compute_portfolio_beta(self) -> float:
        """Compute weighted portfolio beta."""
        betas = {ab.symbol: ab.beta for ab in self.compute_asset_betas()}
        return sum(self._positions.get(s, 0) * betas.get(s, 1.0) for s in self._positions)

    def compute_hedge_ratio(self) -> tuple[float, float]:
        """Compute hedge ratio and hedge notional to achieve target beta.

        Returns: (hedge_ratio, hedge_notional)
        """
        port_beta = self.compute_portfolio_beta()
        # hedge_ratio = (portfolio_beta - target_beta) / hedge_beta
        # Assuming hedge instrument has beta = 1.0
        hedge_ratio = port_beta - self.target_beta
        total_notional = sum(abs(v) for v in self._positions.values())
        hedge_notional = hedge_ratio * total_notional

        return float(hedge_ratio), float(hedge_notional)

    def analyze(self) -> SystematicRiskReport:
        """Run full systematic risk analysis."""
        asset_betas = self.compute_asset_betas()
        port_beta = self.compute_portfolio_beta()
        hedge_ratio, hedge_notional = self.compute_hedge_ratio()

        # Portfolio-level risk decomposition
        if not self._market_returns:
            return SystematicRiskReport(
                portfolio_beta=round(port_beta, 4),
                target_beta=self.target_beta,
                asset_betas=asset_betas,
            )

        market = np.array(self._market_returns)
        market_var = float(np.var(market))
        market_vol = float(np.std(market))

        # Systematic variance = beta^2 * market_var
        systematic_var = port_beta ** 2 * market_var

        # Total portfolio variance (approximate from individual volatilities)
        total_var = 0.0
        for ab in asset_betas:
            w = self._positions.get(ab.symbol, 0)
            total_var += (w * ab.volatility) ** 2

        sys_pct = systematic_var / total_var if total_var > 0 else 0.0
        idio_pct = 1.0 - sys_pct

        # Risk level
        if abs(port_beta) > 1.5 or sys_pct > 0.8:
            risk_level = "high"
        elif abs(port_beta) > 1.0 or sys_pct > 0.6:
            risk_level = "medium"
        else:
            risk_level = "low"

        return SystematicRiskReport(
            portfolio_beta=round(port_beta, 4),
            portfolio_systematic_var=round(systematic_var, 8),
            portfolio_total_var=round(total_var, 8),
            systematic_risk_pct=round(sys_pct, 4),
            idiosyncratic_risk_pct=round(idio_pct, 4),
            asset_betas=asset_betas,
            hedge_ratio=round(hedge_ratio, 4),
            target_beta=self.target_beta,
            hedge_notional=round(hedge_notional, 2),
            risk_level=risk_level,
        )
