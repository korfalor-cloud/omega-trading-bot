"""Open Interest Analysis.

Implements:
- OI tracking
- OI change detection
- OI-price divergence
- Liquidation monitoring
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OIState:
    """Open interest state."""
    current_oi: float = 0.0
    oi_change: float = 0.0
    oi_change_pct: float = 0.0
    price_change_pct: float = 0.0
    divergence: str = ""  # bullish, bearish, neutral
    signal: str = ""


class OpenInterestAnalyzer:
    """Analyze open interest for trading signals."""

    def __init__(self, config: dict = None):
        config = config or {}
        self._oi_history: list[float] = []
        self._price_history: list[float] = []

    def update(self, oi: float, price: float) -> None:
        self._oi_history.append(oi)
        self._price_history.append(price)

    def analyze(self) -> OIState:
        """Analyze current OI state."""
        if len(self._oi_history) < 2:
            return OIState()

        current_oi = self._oi_history[-1]
        prev_oi = self._oi_history[-2]
        current_price = self._price_history[-1]
        prev_price = self._price_history[-2]

        oi_change = current_oi - prev_oi
        oi_change_pct = oi_change / prev_oi if prev_oi > 0 else 0
        price_change_pct = (current_price - prev_price) / prev_price if prev_price > 0 else 0

        # Divergence detection
        divergence = "neutral"
        signal = ""

        if oi_change_pct > 0.01 and price_change_pct > 0.01:
            divergence = "bullish"
            signal = "long"
        elif oi_change_pct > 0.01 and price_change_pct < -0.01:
            divergence = "bearish"
            signal = "short"
        elif oi_change_pct < -0.01 and price_change_pct > 0.01:
            divergence = "bearish_cover"
            signal = "cautious_long"
        elif oi_change_pct < -0.01 and price_change_pct < -0.01:
            divergence = "bullish_liquidation"
            signal = "cautious_short"

        return OIState(
            current_oi=current_oi,
            oi_change=oi_change,
            oi_change_pct=oi_change_pct,
            price_change_pct=price_change_pct,
            divergence=divergence,
            signal=signal,
        )

    def get_liquidation_risk(self, current_oi: float, avg_oi: float) -> str:
        """Assess liquidation risk based on OI."""
        ratio = current_oi / avg_oi if avg_oi > 0 else 1
        if ratio > 1.5:
            return "high"
        elif ratio > 1.2:
            return "medium"
        return "low"
