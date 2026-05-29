"""Omega Ratio — risk-adjusted return metric.

Implements:
- Omega ratio calculation
- Omega optimization
- Threshold parameter tuning
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class OmegaRatioCalculator:
    """Omega ratio computation and optimization."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def compute(self, returns: np.ndarray, threshold: float = None) -> float:
        """Compute Omega ratio.

        Omega = E[max(R - threshold, 0)] / E[max(threshold - R, 0)]
        """
        t = threshold if threshold is not None else self.threshold
        gains = np.sum(np.maximum(returns - t, 0))
        losses = np.sum(np.maximum(t - returns, 0))

        return float(gains / losses) if losses > 0 else float("inf")

    def compute_profile(self, returns: np.ndarray, thresholds: np.ndarray = None) -> dict[float, float]:
        """Compute Omega ratio for multiple thresholds."""
        if thresholds is None:
            thresholds = np.linspace(-0.05, 0.05, 21)

        return {float(t): self.compute(returns, t) for t in thresholds}

    def optimize(self, returns: np.ndarray) -> float:
        """Find optimal threshold that maximizes Omega."""
        thresholds = np.linspace(-0.10, 0.10, 101)
        best_t = 0.0
        best_omega = 0.0

        for t in thresholds:
            omega = self.compute(returns, t)
            if omega > best_omega and omega < float("inf"):
                best_omega = omega
                best_t = t

        return best_t

    def rank_assets(self, returns_dict: dict[str, np.ndarray]) -> list[tuple[str, float]]:
        """Rank assets by Omega ratio."""
        omegas = []
        for asset, returns in returns_dict.items():
            omega = self.compute(returns)
            omegas.append((asset, omega))
        return sorted(omegas, key=lambda x: x[1], reverse=True)
