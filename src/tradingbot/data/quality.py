"""Data Quality — validation, gap detection, anomaly detection.

Implements:
- OHLCV data validation
- Missing data/gap detection
- Price anomaly detection (spikes, stale data)
- Volume anomaly detection
- Data completeness scoring
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Data quality report."""
    total_bars: int = 0
    valid_bars: int = 0
    completeness: float = 0.0
    gaps: int = 0
    anomalies: int = 0
    stale_bars: int = 0
    issues: list[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class DataQualityChecker:
    """Market data quality validation."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.max_price_change_pct = config.get("max_price_change_pct", 0.10)
        self.stale_threshold_bars = config.get("stale_threshold_bars", 5)
        self.min_volume = config.get("min_volume", 0)
        self.max_gap_seconds = config.get("max_gap_seconds", 3600)

    def validate_bars(self, bars: list[dict]) -> QualityReport:
        """Validate a list of OHLCV bars.

        Each bar dict should have: timestamp, open, high, low, close, volume
        """
        if not bars:
            return QualityReport(issues=["Empty data"])

        issues = []
        valid = 0
        anomalies = 0
        stale = 0
        gaps = 0

        for i, bar in enumerate(bars):
            # Check required fields
            if not all(k in bar for k in ("open", "high", "low", "close", "volume")):
                issues.append(f"Bar {i}: missing fields")
                continue

            o, h, l, c, v = bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"]

            # Basic OHLC validation
            if h < l or h < o or h < c or l > o or l > c:
                issues.append(f"Bar {i}: invalid OHLC (O={o} H={h} L={l} C={c})")
                anomalies += 1
                continue

            # Volume check
            if v < self.min_volume:
                stale += 1

            # Price spike detection
            if i > 0 and "close" in bars[i - 1]:
                prev_c = bars[i - 1]["close"]
                if prev_c > 0:
                    change = abs(c - prev_c) / prev_c
                    if change > self.max_price_change_pct:
                        anomalies += 1
                        issues.append(f"Bar {i}: price spike {change:.1%}")

            # Gap detection
            if i > 0 and "timestamp" in bar and "timestamp" in bars[i - 1]:
                try:
                    dt = (bar["timestamp"] - bars[i - 1]["timestamp"]).total_seconds()
                    if dt > self.max_gap_seconds:
                        gaps += 1
                except (TypeError, AttributeError):
                    pass

            # Stale data (same price for N bars)
            if i >= self.stale_threshold_bars:
                recent = [bars[j]["close"] for j in range(i - self.stale_threshold_bars, i + 1) if "close" in bars[j]]
                if len(recent) == self.stale_threshold_bars + 1 and len(set(recent)) == 1:
                    stale += 1

            valid += 1

        completeness = valid / len(bars) if bars else 0

        return QualityReport(
            total_bars=len(bars),
            valid_bars=valid,
            completeness=completeness,
            gaps=gaps,
            anomalies=anomalies,
            stale_bars=stale,
            issues=issues[:50],
        )

    def detect_spikes(
        self,
        prices: np.ndarray,
        threshold: float = 3.0,
    ) -> list[int]:
        """Detect price spikes using z-score.

        Returns indices of anomalous bars.
        """
        if len(prices) < 10:
            return []

        returns = np.diff(prices) / prices[:-1]
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        if std_ret < 1e-10:
            return []

        zscores = np.abs((returns - mean_ret) / std_ret)
        return [i + 1 for i, z in enumerate(zscores) if z > threshold]

    def detect_stale(
        self,
        prices: np.ndarray,
        min_changes: int = 3,
        window: int = 10,
    ) -> list[int]:
        """Detect stale data (insufficient price movement)."""
        stale_indices = []
        for i in range(window, len(prices)):
            segment = prices[i - window:i]
            n_changes = np.sum(np.abs(np.diff(segment)) > 0)
            if n_changes < min_changes:
                stale_indices.append(i)
        return stale_indices

    def fill_gaps(
        self,
        bars: list[dict],
        method: str = "forward",
    ) -> list[dict]:
        """Fill gaps in data.

        Methods: forward, linear, zero
        """
        if len(bars) < 2:
            return bars

        filled = [bars[0]]
        for i in range(1, len(bars)):
            filled.append(bars[i])

            if "timestamp" in bars[i] and "timestamp" in bars[i - 1]:
                try:
                    gap = (bars[i]["timestamp"] - bars[i - 1]["timestamp"]).total_seconds()
                    if gap > self.max_gap_seconds * 2:
                        # Insert interpolated bars
                        n_fill = int(gap / self.max_gap_seconds) - 1
                        for j in range(1, n_fill + 1):
                            frac = j / (n_fill + 1)
                            if method == "linear":
                                interp = {
                                    "open": bars[i - 1]["close"] + (bars[i]["open"] - bars[i - 1]["close"]) * frac,
                                    "high": bars[i - 1]["close"] + (bars[i]["high"] - bars[i - 1]["close"]) * frac,
                                    "low": bars[i - 1]["close"] + (bars[i]["low"] - bars[i - 1]["close"]) * frac,
                                    "close": bars[i - 1]["close"] + (bars[i]["close"] - bars[i - 1]["close"]) * frac,
                                    "volume": 0,
                                }
                            else:  # forward fill
                                interp = dict(bars[i - 1])
                                interp["volume"] = 0
                            filled.append(interp)
                except (TypeError, AttributeError):
                    pass

        return filled
