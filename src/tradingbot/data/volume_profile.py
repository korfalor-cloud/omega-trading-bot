"""Volume Profile Analysis.

Implements:
- Volume-at-price distribution
- POC (Point of Control) detection
- Value Area High/Low (VAH/VAL)
- Volume-weighted support/resistance levels
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfileResult:
    """Volume profile analysis result."""
    poc: float = 0.0  # Point of Control (highest volume price)
    vah: float = 0.0  # Value Area High
    val: float = 0.0  # Value Area Low
    total_volume: float = 0.0
    price_levels: np.ndarray = None
    volume_at_price: np.ndarray = None

    def __post_init__(self):
        if self.price_levels is None:
            self.price_levels = np.array([])
        if self.volume_at_price is None:
            self.volume_at_price = np.array([])


class VolumeProfileAnalyzer:
    """Volume profile analysis engine."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.n_bins = config.get("n_bins", 50)
        self.value_area_pct = config.get("value_area_pct", 0.70)

    def analyze(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        highs: np.ndarray | None = None,
        lows: np.ndarray | None = None,
    ) -> VolumeProfileResult:
        """Build and analyze volume profile."""
        if len(prices) == 0 or len(volumes) == 0:
            return VolumeProfileResult()

        # Build volume-at-price
        price_min = np.min(lows) if lows is not None else np.min(prices)
        price_max = np.max(highs) if highs is not None else np.max(prices)

        if price_min == price_max:
            return VolumeProfileResult(poc=price_min, total_volume=float(np.sum(volumes)))

        bins = np.linspace(price_min, price_max, self.n_bins + 1)
        vol_at_price = np.zeros(self.n_bins)

        # Distribute volume across price range for each bar
        if highs is not None and lows is not None:
            for i in range(len(prices)):
                low, high, vol = lows[i], highs[i], volumes[i]
                mask = (bins[:-1] >= low) & (bins[:-1] <= high)
                n_active = np.sum(mask)
                if n_active > 0:
                    vol_at_price[mask] += vol / n_active
        else:
            # Use close prices only
            bin_indices = np.digitize(prices, bins) - 1
            bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
            for i, idx in enumerate(bin_indices):
                vol_at_price[idx] += volumes[i]

        # POC
        poc_idx = np.argmax(vol_at_price)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2

        # Value Area (70% of volume)
        total_vol = np.sum(vol_at_price)
        target_vol = total_vol * self.value_area_pct

        # Expand outward from POC
        cumulative = vol_at_price[poc_idx]
        low_idx = poc_idx
        high_idx = poc_idx

        while cumulative < target_vol and (low_idx > 0 or high_idx < self.n_bins - 1):
            expand_low = vol_at_price[low_idx - 1] if low_idx > 0 else 0
            expand_high = vol_at_price[high_idx + 1] if high_idx < self.n_bins - 1 else 0

            if expand_low >= expand_high and low_idx > 0:
                low_idx -= 1
                cumulative += vol_at_price[low_idx]
            elif high_idx < self.n_bins - 1:
                high_idx += 1
                cumulative += vol_at_price[high_idx]
            else:
                low_idx -= 1
                cumulative += vol_at_price[low_idx]

        val = bins[low_idx]
        vah = bins[min(high_idx + 1, self.n_bins)]

        return VolumeProfileResult(
            poc=poc,
            vah=vah,
            val=val,
            total_volume=float(total_vol),
            price_levels=(bins[:-1] + bins[1:]) / 2,
            volume_at_price=vol_at_price,
        )

    def find_support_resistance(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        n_levels: int = 3,
    ) -> list[float]:
        """Find high-volume support/resistance levels."""
        profile = self.analyze(prices, volumes)
        if len(profile.volume_at_price) == 0:
            return []

        # Find peaks in volume profile
        vol = profile.volume_at_price
        levels = []
        for i in range(1, len(vol) - 1):
            if vol[i] > vol[i - 1] and vol[i] > vol[i + 1]:
                levels.append((vol[i], profile.price_levels[i]))

        levels.sort(reverse=True)
        return [lvl for _, lvl in levels[:n_levels]]
