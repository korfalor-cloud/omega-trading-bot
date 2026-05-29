"""Counterparty Risk — exchange risk scoring.

Implements:
- Exchange risk scoring
- Exposure tracking per exchange
- Diversification metrics
- Risk alerts
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CounterpartyScore:
    """Exchange risk score."""
    exchange: str = ""
    risk_score: float = 0.0  # 0-1, higher = riskier
    exposure_usd: float = 0.0
    exposure_pct: float = 0.0
    risk_factors: dict = None

    def __post_init__(self):
        if self.risk_factors is None:
            self.risk_factors = {}


class CounterpartyRiskManager:
    """Manage counterparty risk across exchanges."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.max_exposure_pct = config.get("max_exposure_pct", 0.40)
        self.max_single_exchange = config.get("max_single_exchange", 0.50)
        self._exposures: dict[str, float] = {}
        self._risk_scores: dict[str, float] = {}

    def set_risk_score(self, exchange: str, score: float) -> None:
        self._risk_scores[exchange] = score

    def update_exposure(self, exchange: str, amount: float) -> None:
        self._exposures[exchange] = amount

    def get_total_exposure(self) -> float:
        return sum(self._exposures.values())

    def analyze(self) -> list[CounterpartyScore]:
        """Analyze counterparty risk."""
        total = self.get_total_exposure()
        scores = []

        for exchange, exposure in self._exposures.items():
            risk_score = self._risk_scores.get(exposure, 0.5)
            exposure_pct = exposure / total if total > 0 else 0

            risk_factors = {}
            if exposure_pct > self.max_single_exchange:
                risk_factors["concentration"] = "high"
            if risk_score > 0.7:
                risk_factors["exchange_risk"] = "high"

            scores.append(CounterpartyScore(
                exchange=exchange,
                risk_score=risk_score,
                exposure_usd=exposure,
                exposure_pct=exposure_pct,
                risk_factors=risk_factors,
            ))

        return scores

    def get_diversification_score(self) -> float:
        """Get exposure diversification score (0-1)."""
        if not self._exposures:
            return 1.0

        total = self.get_total_exposure()
        if total == 0:
            return 1.0

        weights = [e / total for e in self._exposures.values()]
        hhi = sum(w ** 2 for w in weights)
        n = len(weights)
        return max(0, 1 - (hhi - 1 / n) / (1 - 1 / n)) if n > 1 else 0

    def should_rebalance(self) -> bool:
        """Check if rebalancing is needed."""
        total = self.get_total_exposure()
        for exposure in self._exposures.values():
            if exposure / total > self.max_single_exchange:
                return True
        return False
