"""Market Data Normalizer.

Implements:
- Multi-exchange data normalization
- Timestamp alignment
- Price normalization (different quote currencies)
- Volume normalization
- Data resampling
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NormalizedBar:
    """Normalized OHLCV bar."""
    timestamp: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    source: str = ""


class DataNormalizer:
    """Normalize market data from multiple sources."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.base_currency = config.get("base_currency", "USDT")
        self.timestamp_unit = config.get("timestamp_unit", "seconds")

    def normalize_bars(self, bars: list[dict], source: str = "") -> list[NormalizedBar]:
        """Normalize raw bars to standard format."""
        result = []
        for bar in bars:
            ts = bar.get("timestamp", 0)
            if hasattr(ts, "timestamp"):
                ts = ts.timestamp()
            elif isinstance(ts, str):
                from datetime import datetime
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()

            result.append(NormalizedBar(
                timestamp=float(ts),
                open=float(bar.get("open", 0)),
                high=float(bar.get("high", 0)),
                low=float(bar.get("low", 0)),
                close=float(bar.get("close", 0)),
                volume=float(bar.get("volume", 0)),
                source=source,
            ))
        return result

    def align_timestamps(
        self,
        bars_a: list[NormalizedBar],
        bars_b: list[NormalizedBar],
        tolerance: float = 60.0,
    ) -> tuple[list[NormalizedBar], list[NormalizedBar]]:
        """Align two bar series by timestamp."""
        aligned_a = []
        aligned_b = []

        i, j = 0, 0
        while i < len(bars_a) and j < len(bars_b):
            diff = abs(bars_a[i].timestamp - bars_b[j].timestamp)
            if diff <= tolerance:
                aligned_a.append(bars_a[i])
                aligned_b.append(bars_b[j])
                i += 1
                j += 1
            elif bars_a[i].timestamp < bars_b[j].timestamp:
                i += 1
            else:
                j += 1

        return aligned_a, aligned_b

    def resample(
        self,
        bars: list[NormalizedBar],
        target_period: float,
    ) -> list[NormalizedBar]:
        """Resample bars to a different timeframe."""
        if not bars:
            return []

        result = []
        current = NormalizedBar(
            timestamp=bars[0].timestamp,
            open=bars[0].open,
            high=bars[0].high,
            low=bars[0].low,
            close=bars[0].close,
            volume=bars[0].volume,
            source=bars[0].source,
        )
        period_start = bars[0].timestamp

        for bar in bars[1:]:
            if bar.timestamp - period_start >= target_period:
                result.append(current)
                period_start = bar.timestamp
                current = NormalizedBar(
                    timestamp=bar.timestamp,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    source=bar.source,
                )
            else:
                current.high = max(current.high, bar.high)
                current.low = min(current.low, bar.low)
                current.close = bar.close
                current.volume += bar.volume

        result.append(current)
        return result

    def normalize_volume(self, volumes: np.ndarray, method: str = "zscore") -> np.ndarray:
        """Normalize volume data."""
        if method == "zscore":
            mean = np.mean(volumes)
            std = np.std(volumes)
            return (volumes - mean) / std if std > 0 else np.zeros_like(volumes)
        elif method == "minmax":
            vmin, vmax = np.min(volumes), np.max(volumes)
            return (volumes - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(volumes)
        elif method == "log":
            return np.log1p(volumes)
        return volumes

    def detect_outliers(self, prices: np.ndarray, threshold: float = 3.0) -> list[int]:
        """Detect price outliers using z-score."""
        returns = np.diff(prices) / prices[:-1]
        zscores = np.abs((returns - np.mean(returns)) / np.std(returns)) if np.std(returns) > 0 else np.zeros_like(returns)
        return [i + 1 for i, z in enumerate(zscores) if z > threshold]
