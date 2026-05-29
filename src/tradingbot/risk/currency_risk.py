"""Currency Risk — multi-currency exposure management.

Implements:
- Currency exposure tracking across holdings
- FX risk calculation (volatility-adjusted)
- Hedging recommendations for concentrated FX exposures
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CurrencyExposure:
    """Single currency exposure record."""
    currency: str = ""
    notional: float = 0.0
    weight: float = 0.0
    fx_vol: float = 0.0
    risk_contribution: float = 0.0


@dataclass
class CurrencyRiskReport:
    """Full currency risk report."""
    total_notional: float = 0.0
    exposures: list[CurrencyExposure] = field(default_factory=list)
    portfolio_fx_var: float = 0.0
    max_single_exposure: float = 0.0
    concentration_ratio: float = 0.0
    hedges_needed: list[dict] = field(default_factory=list)
    risk_level: str = "low"


class CurrencyRiskManager:
    """Multi-currency exposure management."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.max_exposure_pct = config.get("max_exposure_pct", 0.30)
        self.confidence_level = config.get("confidence_level", 0.95)
        self._positions: dict[str, float] = {}  # currency -> notional in base
        self._fx_vols: dict[str, float] = {}  # currency -> annualized FX vol
        self._fx_returns: dict[str, list[float]] = {}

    def update_position(self, currency: str, notional: float) -> None:
        """Update position for a given currency."""
        self._positions[currency] = notional

    def set_fx_vol(self, currency: str, vol: float) -> None:
        """Set annualized FX volatility for a currency."""
        self._fx_vols[currency] = vol

    def update_fx_return(self, currency: str, return_val: float) -> None:
        """Track FX return history for volatility estimation."""
        if currency not in self._fx_returns:
            self._fx_returns[currency] = []
        self._fx_returns[currency].append(return_val)

    def estimate_fx_vol(self, currency: str, lookback: int = 60) -> float:
        """Estimate FX volatility from return history."""
        if currency in self._fx_vols and self._fx_vols[currency] > 0:
            return self._fx_vols[currency]

        returns = self._fx_returns.get(currency, [])
        if len(returns) < lookback:
            lookback = len(returns)

        if lookback < 10:
            return 0.10  # default 10% annualized vol

        window = np.array(returns[-lookback:])
        daily_vol = np.std(window)
        annual_vol = daily_vol * np.sqrt(252)
        return float(annual_vol)

    def compute_exposures(self) -> list[CurrencyExposure]:
        """Compute currency exposures with risk metrics."""
        total = sum(abs(v) for v in self._positions.values())
        if total == 0:
            return []

        exposures = []
        for currency, notional in sorted(self._positions.items(), key=lambda x: abs(x[1]), reverse=True):
            weight = abs(notional) / total
            vol = self.estimate_fx_vol(currency)
            risk_contrib = weight * vol

            exposures.append(CurrencyExposure(
                currency=currency,
                notional=notional,
                weight=round(weight, 4),
                fx_vol=round(vol, 4),
                risk_contribution=round(risk_contrib, 4),
            ))

        return exposures

    def compute_portfolio_fx_var(self) -> float:
        """Compute portfolio-level FX VaR using variance-covariance method.

        Assumes independent FX factors (conservative approximation).
        """
        exposures = self.compute_exposures()
        if not exposures:
            return 0.0

        total = sum(abs(v) for v in self._positions.values())
        z_score = 1.645  # 95% confidence

        # Sum of squared risk contributions (independent assumption)
        var_squared = sum((e.weight * e.fx_vol * total) ** 2 for e in exposures)
        return float(z_score * np.sqrt(var_squared))

    def get_hedges(self) -> list[dict]:
        """Generate hedging recommendations for concentrated exposures."""
        exposures = self.compute_exposures()
        hedges = []

        for exp in exposures:
            if exp.weight > self.max_exposure_pct:
                excess_weight = exp.weight - self.max_exposure_pct
                hedge_notional = excess_weight * sum(abs(v) for v in self._positions.values())
                hedges.append({
                    "currency": exp.currency,
                    "current_weight": exp.weight,
                    "target_weight": self.max_exposure_pct,
                    "hedge_notional": round(hedge_notional, 2),
                    "method": "fx_forward" if abs(hedge_notional) > 10_000 else "options",
                    "priority": "high" if exp.weight > self.max_exposure_pct * 1.5 else "medium",
                })

        return hedges

    def analyze(self) -> CurrencyRiskReport:
        """Run full currency risk analysis."""
        exposures = self.compute_exposures()
        total = sum(abs(v) for v in self._positions.values())
        max_weight = max((e.weight for e in exposures), default=0.0)
        hedges = self.get_hedges()
        fx_var = self.compute_portfolio_fx_var()

        # Concentration ratio (HHI over weights)
        hhi = sum(e.weight ** 2 for e in exposures)

        # Risk level
        if max_weight > 0.50 or hhi > 0.30:
            risk_level = "high"
        elif max_weight > self.max_exposure_pct or hhi > 0.20:
            risk_level = "medium"
        else:
            risk_level = "low"

        return CurrencyRiskReport(
            total_notional=round(total, 2),
            exposures=exposures,
            portfolio_fx_var=round(fx_var, 2),
            max_single_exposure=round(max_weight, 4),
            concentration_ratio=round(hhi, 4),
            hedges_needed=hedges,
            risk_level=risk_level,
        )
