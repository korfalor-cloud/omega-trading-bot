"""Correlation Breakdown Detector.

Implements:
- Real-time correlation monitoring
- Breakdown detection (correlation drops)
- Crisis indicator
- Portfolio risk adjustment
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BreakdownAlert:
    """Correlation breakdown alert."""
    pair: str = ""
    current_corr: float = 0.0
    historical_corr: float = 0.0
    change: float = 0.0
    severity: str = "low"
    timestamp: float = 0.0


class CorrelationBreakdownDetector:
    """Detect correlation breakdowns during market stress."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.window = config.get("window", 30)
        self.threshold = config.get("threshold", 0.3)
        self._return_history: dict[str, list[float]] = {}

    def update(self, symbol: str, return_val: float) -> None:
        if symbol not in self._return_history:
            self._return_history[symbol] = []
        self._return_history[symbol].append(return_val)
        if len(self._return_history[symbol]) > self.window * 3:
            self._return_history[symbol] = self._return_history[symbol][-self.window * 2:]

    def detect(self) -> list[BreakdownAlert]:
        """Detect correlation breakdowns."""
        alerts = []
        symbols = list(self._return_history.keys())

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                s1, s2 = symbols[i], symbols[j]
                r1 = self._return_history.get(s1, [])
                r2 = self._return_history.get(s2, [])

                if len(r1) < self.window * 2 or len(r2) < self.window * 2:
                    continue

                n = min(len(r1), len(r2))

                # Historical correlation
                hist_corr = np.corrcoef(r1[:n - self.window], r2[:n - self.window])[0, 1]

                # Recent correlation
                recent_corr = np.corrcoef(r1[-self.window:], r2[-self.window:])[0, 1]

                change = abs(recent_corr - hist_corr)

                if change > self.threshold:
                    severity = "high" if change > 0.5 else "medium" if change > 0.3 else "low"
                    alerts.append(BreakdownAlert(
                        pair=f"{s1}_{s2}",
                        current_corr=recent_corr,
                        historical_corr=hist_corr,
                        change=change,
                        severity=severity,
                    ))

        return alerts

    def get_crisis_indicator(self) -> float:
        """Get overall crisis indicator (0-1)."""
        alerts = self.detect()
        if not alerts:
            return 0.0

        avg_change = np.mean([a.change for a in alerts])
        return min(1.0, avg_change / 0.5)

    def should_reduce_exposure(self) -> bool:
        """Check if exposure should be reduced."""
        return self.get_crisis_indicator() > 0.5
