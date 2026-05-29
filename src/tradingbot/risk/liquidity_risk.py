"""Liquidity Risk — market impact and liquidity modeling.

Implements:
- Bid-ask spread monitoring
- Market depth analysis
- Liquidity scoring
- Liquidation risk assessment
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LiquidityScore:
    """Liquidity assessment."""
    score: float = 0.0  # 0-1, higher = more liquid
    spread_bps: float = 0.0
    depth_usd: float = 0.0
    impact_1pct: float = 0.0
    time_to_exit_hours: float = 0.0
    risk_level: str = "low"


class LiquidityRiskAnalyzer:
    """Analyze liquidity risk for positions."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.min_depth = config.get("min_depth", 10000)
        self.max_spread_bps = config.get("max_spread_bps", 10)

    def assess(
        self,
        position_size: float,
        avg_volume: float,
        bid: float,
        ask: float,
        depth_usd: float = 0,
    ) -> LiquidityScore:
        """Assess liquidity for a position."""
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 1
        spread_bps = (ask - bid) / mid * 10000 if mid > 0 else 0

        # Market impact estimate
        notional = position_size * mid
        impact_1pct = notional / (avg_volume * mid) if avg_volume * mid > 0 else 1

        # Time to exit (assuming 10% of avg volume per hour)
        exit_rate = avg_volume * 0.1 if avg_volume > 0 else 1
        time_to_exit = position_size / exit_rate if exit_rate > 0 else 999

        # Liquidity score
        spread_score = max(0, 1 - spread_bps / 100)
        depth_score = min(1, depth_usd / self.min_depth) if self.min_depth > 0 else 1
        volume_score = min(1, avg_volume / 1000) if avg_volume > 0 else 0

        score = (spread_score * 0.3 + depth_score * 0.3 + volume_score * 0.4)

        if score < 0.3:
            risk = "high"
        elif score < 0.6:
            risk = "medium"
        else:
            risk = "low"

        return LiquidityScore(
            score=score,
            spread_bps=spread_bps,
            depth_usd=depth_usd,
            impact_1pct=impact_1pct,
            time_to_exit_hours=time_to_exit,
            risk_level=risk,
        )

    def check_liquidation_risk(
        self,
        position_size: float,
        liquidation_price: float,
        current_price: float,
        avg_volume: float,
    ) -> dict:
        """Check if position can be liquidated before reaching liq price."""
        distance_pct = abs(current_price - liquidation_price) / current_price if current_price > 0 else 0
        exit_time = position_size / (avg_volume * 0.1) if avg_volume > 0 else 999

        return {
            "distance_to_liquidation": distance_pct,
            "exit_time_hours": exit_time,
            "can_exit_before_liq": exit_time < distance_pct * 100,
            "risk_level": "high" if distance_pct < 0.05 else "medium" if distance_pct < 0.10 else "low",
        }
