"""Correlation Monitoring and Breakdown Detection.

Implements:
- Real-time correlation tracking
- Correlation breakdown alerts
- Correlation regime shifts
- Diversification score
- Portfolio correlation heat map data
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CorrelationAlert:
    """Alert for correlation change."""
    asset_a: str = ""
    asset_b: str = ""
    old_corr: float = 0.0
    new_corr: float = 0.0
    change: float = 0.0
    alert_type: str = ""  # breakdown, surge, regime_shift
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CorrelationMonitor:
    """Monitor and alert on correlation changes.

    Tracks rolling correlations between assets and detects
    significant changes that may affect portfolio risk.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.window = config.get("window", 60)
        self.breakdown_threshold = config.get("breakdown_threshold", 0.3)
        self.surge_threshold = config.get("surge_threshold", 0.3)
        self._correlation_history: dict[str, list[float]] = {}
        self._alerts: list[CorrelationAlert] = []

    def update(
        self,
        returns: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Update correlations and detect changes.

        Returns current correlation matrix (flattened pairs).
        """
        names = sorted(returns.keys())
        current_corrs = {}

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                pair_key = f"{a}_{b}"

                ra = returns[a]
                rb = returns[b]
                n = min(len(ra), len(rb))

                if n < self.window:
                    continue

                corr = float(np.corrcoef(ra[-self.window:], rb[-self.window:])[0, 1])
                current_corrs[pair_key] = corr

                # Check for changes
                history = self._correlation_history.get(pair_key, [])
                if history:
                    old_corr = history[-1]
                    change = corr - old_corr

                    if abs(change) > self.breakdown_threshold:
                        alert_type = "breakdown" if abs(corr) < abs(old_corr) else "surge"
                        self._alerts.append(CorrelationAlert(
                            asset_a=a, asset_b=b,
                            old_corr=old_corr, new_corr=corr,
                            change=change, alert_type=alert_type,
                        ))

                if pair_key not in self._correlation_history:
                    self._correlation_history[pair_key] = []
                self._correlation_history[pair_key].append(corr)

        return current_corrs

    def get_correlation_matrix(
        self,
        returns: dict[str, np.ndarray],
    ) -> tuple[list[str], np.ndarray]:
        """Compute current correlation matrix."""
        names = sorted(returns.keys())
        n = len(names)
        matrix = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                ra = returns[names[i]]
                rb = returns[names[j]]
                length = min(len(ra), len(rb))

                if length < self.window:
                    continue

                corr = float(np.corrcoef(ra[-self.window:], rb[-self.window:])[0, 1])
                matrix[i, j] = corr
                matrix[j, i] = corr

        return names, matrix

    def diversification_score(
        self,
        returns: dict[str, np.ndarray],
    ) -> float:
        """Compute portfolio diversification score (0 = none, 1 = perfect).

        Based on average pairwise correlation.
        """
        names, matrix = self.get_correlation_matrix(returns)
        n = len(names)
        if n < 2:
            return 1.0

        off_diag = []
        for i in range(n):
            for j in range(i + 1, n):
                off_diag.append(abs(matrix[i, j]))

        avg_corr = np.mean(off_diag) if off_diag else 0
        return float(1 - avg_corr)

    def get_alerts(self, limit: int = 50) -> list[CorrelationAlert]:
        return self._alerts[-limit:]

    def get_pair_correlation(self, asset_a: str, asset_b: str) -> Optional[float]:
        """Get latest correlation for a specific pair."""
        key1 = f"{asset_a}_{asset_b}"
        key2 = f"{asset_b}_{asset_a}"

        history = self._correlation_history.get(key1) or self._correlation_history.get(key2)
        if history:
            return history[-1]
        return None

    def correlation_trend(
        self,
        asset_a: str,
        asset_b: str,
        lookback: int = 10,
    ) -> str:
        """Check if correlation is trending up, down, or stable."""
        key1 = f"{asset_a}_{asset_b}"
        key2 = f"{asset_b}_{asset_a}"

        history = self._correlation_history.get(key1) or self._correlation_history.get(key2)
        if not history or len(history) < lookback:
            return "stable"

        recent = history[-lookback:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]

        if trend > 0.01:
            return "increasing"
        elif trend < -0.01:
            return "decreasing"
        return "stable"
