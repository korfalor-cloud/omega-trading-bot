"""Interest Rate Risk — funding rate sensitivity analysis.

Implements:
- Funding rate impact calculation for leveraged positions
- Cost projection over holding horizons
- Rate regime detection (high/low/normal funding)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FundingCostProjection:
    """Projected funding costs for a position."""
    position_notional: float = 0.0
    current_rate: float = 0.0
    daily_cost: float = 0.0
    weekly_cost: float = 0.0
    monthly_cost: float = 0.0
    annualized_rate: float = 0.0
    breakeven_days: float = 0.0


@dataclass
class RateRegime:
    """Detected funding rate regime."""
    regime: str = "normal"  # extreme_positive, positive, normal, negative, extreme_negative
    current_rate: float = 0.0
    mean_rate: float = 0.0
    std_rate: float = 0.0
    percentile: float = 0.5
    signal: str = "neutral"


@dataclass
class InterestRateReport:
    """Full interest rate risk report."""
    projections: list[FundingCostProjection] = None
    regime: RateRegime = None
    total_daily_cost: float = 0.0
    total_monthly_cost: float = 0.0
    risk_level: str = "low"

    def __post_init__(self):
        if self.projections is None:
            self.projections = []
        if self.regime is None:
            self.regime = RateRegime()


class InterestRateRiskManager:
    """Funding rate sensitivity analysis."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.funding_periods_per_day = config.get("funding_periods_per_day", 3)
        self.extreme_percentile = config.get("extreme_percentile", 0.90)
        self._rate_history: list[float] = []
        self._positions: list[dict] = []

    def update_rate(self, rate: float) -> None:
        """Record a new funding rate observation."""
        self._rate_history.append(rate)

    def add_position(self, symbol: str, notional: float, side: str, leverage: float = 1.0) -> None:
        """Add a position for interest rate analysis."""
        self._positions.append({
            "symbol": symbol,
            "notional": abs(notional),
            "side": side,
            "leverage": leverage,
        })

    def clear_positions(self) -> None:
        """Clear all tracked positions."""
        self._positions.clear()

    def compute_daily_cost(self, notional: float, rate: float, side: str) -> float:
        """Compute daily funding cost for a position.

        Longs pay when funding is positive; shorts receive.
        """
        cost = notional * abs(rate) * self.funding_periods_per_day
        # Positive funding: longs pay, shorts receive
        if rate > 0 and side == "long":
            return cost
        elif rate < 0 and side == "short":
            return cost
        return -cost  # position receives funding

    def project_costs(self, notional: float, side: str, leverage: float = 1.0) -> FundingCostProjection:
        """Project funding costs over multiple horizons."""
        if not self._rate_history:
            return FundingCostProjection(position_notional=notional)

        # Use recent average rate for projection
        lookback = min(len(self._rate_history), 30)
        avg_rate = float(np.mean(self._rate_history[-lookback:]))
        current_rate = self._rate_history[-1]

        effective_notional = notional * leverage
        daily = self.compute_daily_cost(effective_notional, current_rate, side)

        # Annualized rate
        annualized = current_rate * self.funding_periods_per_day * 365

        # Breakeven: days for cumulative funding to equal 1% of notional
        threshold = effective_notional * 0.01
        breakeven = threshold / abs(daily) if abs(daily) > 0 else float("inf")

        return FundingCostProjection(
            position_notional=round(effective_notional, 2),
            current_rate=round(current_rate, 6),
            daily_cost=round(daily, 2),
            weekly_cost=round(daily * 7, 2),
            monthly_cost=round(daily * 30, 2),
            annualized_rate=round(annualized, 4),
            breakeven_days=round(breakeven, 1),
        )

    def detect_regime(self) -> RateRegime:
        """Detect current funding rate regime."""
        if len(self._rate_history) < 30:
            current = self._rate_history[-1] if self._rate_history else 0.0
            return RateRegime(current_rate=current, regime="normal")

        rates = np.array(self._rate_history)
        current = rates[-1]
        mean = float(np.mean(rates))
        std = float(np.std(rates))

        # Percentile of current rate in historical distribution
        percentile = float(np.sum(rates <= current) / len(rates))

        # Classify regime
        if percentile >= self.extreme_percentile:
            regime = "extreme_positive"
        elif percentile >= 0.65:
            regime = "positive"
        elif percentile <= 1 - self.extreme_percentile:
            regime = "extreme_negative"
        elif percentile <= 0.35:
            regime = "negative"
        else:
            regime = "normal"

        # Signal: high positive funding -> shorts pay less -> bearish setup
        # High negative funding -> longs receive -> bullish setup
        if regime in ("extreme_positive", "positive"):
            signal = "bearish"
        elif regime in ("extreme_negative", "negative"):
            signal = "bullish"
        else:
            signal = "neutral"

        return RateRegime(
            regime=regime,
            current_rate=round(current, 6),
            mean_rate=round(mean, 6),
            std_rate=round(std, 6),
            percentile=round(percentile, 3),
            signal=signal,
        )

    def analyze(self) -> InterestRateReport:
        """Run full interest rate risk analysis."""
        regime = self.detect_regime()
        projections = []

        for pos in self._positions:
            proj = self.project_costs(pos["notional"], pos["side"], pos["leverage"])
            projections.append(proj)

        total_daily = sum(p.daily_cost for p in projections)
        total_monthly = sum(p.monthly_cost for p in projections)

        # Risk level based on total cost relative to total notional
        total_notional = sum(p.position_notional for p in projections)
        cost_ratio = abs(total_monthly) / total_notional if total_notional > 0 else 0

        if cost_ratio > 0.05 or regime.regime.startswith("extreme"):
            risk_level = "high"
        elif cost_ratio > 0.02 or regime.regime in ("positive", "negative"):
            risk_level = "medium"
        else:
            risk_level = "low"

        return InterestRateReport(
            projections=projections,
            regime=regime,
            total_daily_cost=round(total_daily, 2),
            total_monthly_cost=round(total_monthly, 2),
            risk_level=risk_level,
        )
