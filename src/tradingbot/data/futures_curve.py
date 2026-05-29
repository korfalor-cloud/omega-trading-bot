"""Futures Curve Analysis.

Implements:
- Term structure construction
- Contango/backwardation detection
- Roll yield calculation
- Curve slope analysis
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CurveState:
    """Futures curve state."""
    regime: str = "neutral"  # contango, backwardation, flat
    slope: float = 0.0
    roll_yield: float = 0.0
    curvature: float = 0.0
    front_price: float = 0.0
    back_price: float = 0.0


class FuturesCurveAnalyzer:
    """Analyze futures term structure."""

    def __init__(self, config: dict = None):
        config = config or {}
        self._curve_history: list[dict] = {}

    def analyze(self, contracts: dict[str, float]) -> CurveState:
        """Analyze futures curve from contract prices.

        Args:
            contracts: {expiry: price} e.g. {"202403": 50000, "202406": 51000}
        """
        if len(contracts) < 2:
            return CurveState()

        sorted_contracts = sorted(contracts.items())
        prices = np.array([p for _, p in sorted_contracts])
        expiries = np.arange(len(prices))

        # Slope (linear regression)
        if len(prices) >= 2:
            slope = (prices[-1] - prices[0]) / (len(prices) - 1)
        else:
            slope = 0

        # Regime
        pct_diff = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
        if pct_diff > 0.01:
            regime = "contango"
        elif pct_diff < -0.01:
            regime = "backwardation"
        else:
            regime = "flat"

        # Roll yield (annualized)
        front_price = prices[0]
        back_price = prices[-1]
        roll_yield = (back_price - front_price) / front_price * 4 if front_price > 0 else 0  # Quarterly roll

        # Curvature
        curvature = 0
        if len(prices) >= 3:
            mid = prices[len(prices) // 2]
            expected_mid = (prices[0] + prices[-1]) / 2
            curvature = (mid - expected_mid) / expected_mid if expected_mid > 0 else 0

        return CurveState(
            regime=regime,
            slope=float(slope),
            roll_yield=float(roll_yield),
            curvature=float(curvature),
            front_price=float(front_price),
            back_price=float(back_price),
        )

    def detect_regime_change(self, current: CurveState, previous: CurveState) -> bool:
        """Detect regime change."""
        return current.regime != previous.regime

    def get_carry_signal(self, state: CurveState) -> str:
        """Get carry trade signal."""
        if state.regime == "contango" and state.roll_yield > 0.05:
            return "short_futures_long_spot"
        elif state.regime == "backwardation" and state.roll_yield < -0.05:
            return "long_futures_short_spot"
        return "neutral"
