"""Order Book Reconstruction and Analysis.

Implements:
- Order book snapshot construction from raw data
- Level aggregation and normalization
- Order flow analysis
- Multi-level depth features
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.types import OrderBookLevel, OrderBookSnapshot

logger = logging.getLogger(__name__)


@dataclass
class OrderBookFeatures:
    """Features extracted from order book data."""
    mid_price: float = 0.0
    spread_bps: float = 0.0
    bid_depth_total: float = 0.0
    ask_depth_total: float = 0.0
    imbalance: float = 0.0  # bid_depth - ask_depth / total
    weighted_mid_price: float = 0.0
    micro_price: float = 0.0
    bid_slope: float = 0.0  # How quickly depth decreases away from mid
    ask_slope: float = 0.0
    bid_volume_imbalance_levels: list[float] = field(default_factory=list)
    kyle_lambda: float = 0.0
    pressure_ratio: float = 0.0


class OrderBookAnalyzer:
    """Order book feature extraction and analysis.

    Computes microstructural features from L2 order book data
    for use in trading signals and market making.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.n_levels = config.get("n_levels", 10)
        self.depth_percentiles = config.get("depth_percentiles", [0.25, 0.5, 0.75])

    def extract_features(self, book: OrderBookSnapshot) -> OrderBookFeatures:
        """Extract all features from an order book snapshot."""
        if not book.bids or not book.asks:
            return OrderBookFeatures()

        best_bid = book.bids[0].price
        best_ask = book.asks[0].price
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_bps = (spread / mid * 10000) if mid > 0 else 0

        bid_depth = sum(l.quantity for l in book.bids[:self.n_levels])
        ask_depth = sum(l.quantity for l in book.asks[:self.n_levels])
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0

        # Weighted mid price (volume-weighted)
        if bid_depth > 0 and ask_depth > 0:
            weighted_mid = (best_bid * ask_depth + best_ask * bid_depth) / total_depth
        else:
            weighted_mid = mid

        # Micro price (queue-position weighted)
        best_bid_qty = book.bids[0].quantity
        best_ask_qty = book.asks[0].quantity
        if best_bid_qty + best_ask_qty > 0:
            micro_price = (best_bid * best_ask_qty + best_ask * best_bid_qty) / (best_bid_qty + best_ask_qty)
        else:
            micro_price = mid

        # Depth slope (how fast depth decreases from mid)
        bid_slope = self._compute_slope(book.bids, mid, is_bid=True)
        ask_slope = self._compute_slope(book.asks, mid, is_bid=False)

        # Per-level imbalance
        level_imbalances = []
        for i in range(min(self.n_levels, len(book.bids), len(book.asks))):
            b = book.bids[i].quantity
            a = book.asks[i].quantity
            total = b + a
            level_imbalances.append((b - a) / total if total > 0 else 0)

        # Pressure ratio (weighted by proximity to mid)
        pressure = 0.0
        for i, level in enumerate(book.bids[:self.n_levels]):
            weight = 1.0 / (i + 1)
            pressure += level.quantity * weight
        for i, level in enumerate(book.asks[:self.n_levels]):
            weight = 1.0 / (i + 1)
            pressure -= level.quantity * weight

        return OrderBookFeatures(
            mid_price=mid,
            spread_bps=spread_bps,
            bid_depth_total=bid_depth,
            ask_depth_total=ask_depth,
            imbalance=imbalance,
            weighted_mid_price=weighted_mid,
            micro_price=micro_price,
            bid_slope=bid_slope,
            ask_slope=ask_slope,
            bid_volume_imbalance_levels=level_imbalances,
            pressure_ratio=pressure / total_depth if total_depth > 0 else 0,
        )

    def compute_depth_features(
        self,
        book: OrderBookSnapshot,
        n_levels: int = 10,
    ) -> np.ndarray:
        """Compute depth features as a fixed-size numpy array."""
        features = []

        for side_levels in [book.bids[:n_levels], book.asks[:n_levels]]:
            quantities = [l.quantity for l in side_levels]
            prices = [l.price for l in side_levels]

            # Pad to n_levels
            while len(quantities) < n_levels:
                quantities.append(0)
                prices.append(prices[-1] if prices else 0)

            quantities = np.array(quantities[:n_levels])
            prices = np.array(prices[:n_levels])

            features.extend([
                np.sum(quantities),  # Total depth
                np.mean(quantities),  # Average level size
                np.std(quantities),  # Depth dispersion
                np.max(quantities),  # Max level size
            ])

        return np.array(features)

    def _compute_slope(
        self,
        levels: list[OrderBookLevel],
        mid_price: float,
        is_bid: bool,
        n_levels: int = 5,
    ) -> float:
        """Compute the slope of depth away from mid price."""
        if len(levels) < 2:
            return 0.0

        distances = []
        quantities = []
        for i, level in enumerate(levels[:n_levels]):
            dist = abs(level.price - mid_price)
            if mid_price > 0:
                dist_bps = dist / mid_price * 10000
            else:
                dist_bps = 0
            distances.append(dist_bps)
            quantities.append(level.quantity)

        if len(distances) < 2:
            return 0.0

        # Simple linear regression
        x = np.array(distances)
        y = np.array(quantities)
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator < 1e-10:
            return 0.0

        return float(numerator / denominator)
