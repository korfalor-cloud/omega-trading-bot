"""Market Microstructure Analysis.

Implements:
- VPIN (Volume-synchronized Probability of Informed Trading)
- Kyle's Lambda (price impact)
- Order flow imbalance
- Trade classification (Lee-Ready algorithm)
- Effective spread estimation
- Market impact models
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...core.types import OHLCVBar, OrderBookSnapshot, Tick

logger = logging.getLogger(__name__)


@dataclass
class MicrostructureMetrics:
    """Aggregated microstructure metrics."""
    vpin: float = 0.0
    kyle_lambda: float = 0.0
    effective_spread_bps: float = 0.0
    order_flow_imbalance: float = 0.0
    trade_intensity: float = 0.0
    toxicity: float = 0.0  # Overall toxicity metric


class MicrostructureAnalyzer:
    """Market microstructure analysis engine.

    Analyzes order flow, trade classification, and market impact
    from tick-level and order book data.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.vpin_buckets = config.get("vpin_buckets", 50)
        self.vpin_bucket_size = config.get("vpin_bucket_size", 0)

    def compute_vpin(
        self,
        volumes: np.ndarray,
        price_changes: np.ndarray,
        bucket_size: int = 0,
    ) -> float:
        """VPIN — Volume-Synchronized Probability of Informed Trading.

        Classifies trades as buy/sell using the bulk volume classification
        method and computes the order imbalance ratio.

        Args:
            volumes: Array of trade volumes
            price_changes: Array of price changes
            bucket_size: Volume per bucket (0 = auto from median)
        """
        if len(volumes) < 20:
            return 0.0

        if bucket_size <= 0:
            bucket_size = float(np.median(volumes) * 50)

        # Bulk Volume Classification
        sigma = np.std(price_changes) if np.std(price_changes) > 0 else 1e-8
        z = price_changes / sigma

        # Standard normal CDF approximation
        buy_volume_pct = 0.5 * (1 + np.tanh(z * 0.7978))  # Approximation of Phi(z)

        buy_vol = volumes * buy_volume_pct
        sell_vol = volumes * (1 - buy_volume_pct)

        # Aggregate into buckets
        cumulative_vol = np.cumsum(volumes)
        n_buckets = max(1, int(cumulative_vol[-1] / bucket_size)) if bucket_size > 0 else 1

        if n_buckets < 5:
            # Not enough data for meaningful VPIN
            total_buy = np.sum(buy_vol)
            total_sell = np.sum(sell_vol)
            total = total_buy + total_sell
            if total > 0:
                return abs(total_buy - total_sell) / total
            return 0.0

        bucket_imbalances = []
        for b in range(n_buckets):
            vol_low = b * bucket_size
            vol_high = (b + 1) * bucket_size
            mask = (cumulative_vol >= vol_low) & (cumulative_vol < vol_high)
            if np.any(mask):
                bucket_buy = np.sum(buy_vol[mask])
                bucket_sell = np.sum(sell_vol[mask])
                bucket_total = bucket_buy + bucket_sell
                if bucket_total > 0:
                    bucket_imbalances.append(abs(bucket_buy - bucket_sell) / bucket_total)

        if not bucket_imbalances:
            return 0.0

        return float(np.mean(bucket_imbalances))

    def compute_kyle_lambda(
        self,
        price_changes: np.ndarray,
        signed_volumes: np.ndarray,
    ) -> float:
        """Kyle's Lambda — price impact coefficient.

        Measures the price impact per unit of order flow.
        Lambda = Cov(delta_p, V) / Var(V)

        Higher lambda = more informed trading / less liquidity.
        """
        if len(price_changes) < 20 or len(signed_volumes) < 20:
            return 0.0

        n = min(len(price_changes), len(signed_volumes))
        dp = price_changes[:n]
        sv = signed_volumes[:n]

        var_v = np.var(sv)
        if var_v < 1e-12:
            return 0.0

        cov_pv = np.cov(dp, sv)[0, 1]
        return float(cov_pv / var_v)

    def classify_trade(
        self,
        price: float,
        bid: float,
        ask: float,
        volume: float,
    ) -> tuple[str, float]:
        """Lee-Ready trade classification.

        Returns:
            side: 'buy' or 'sell'
            signed_volume: positive for buy, negative for sell
        """
        mid = (bid + ask) / 2 if (bid + ask) > 0 else price

        if price > mid:
            return "buy", volume
        elif price < mid:
            return "sell", -volume
        else:
            # At the midpoint — use tick rule (compare to previous price)
            return "buy", volume  # Default to buy at midpoint

    def compute_effective_spread(
        self,
        trade_price: float,
        mid_price: float,
        volume: float,
    ) -> float:
        """Effective spread in basis points.

        Effective spread = 2 * |trade_price - mid_price| / mid_price
        """
        if mid_price == 0:
            return 0.0
        return 2.0 * abs(trade_price - mid_price) / mid_price * 10000

    def compute_order_flow_imbalance(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        depth_levels: int = 5,
    ) -> float:
        """Order flow imbalance from order book.

        Returns value in [-1, 1]:
            +1 = heavy bid pressure (bullish)
            -1 = heavy ask pressure (bearish)
        """
        bid_vol = sum(qty for _, qty in bids[:depth_levels])
        ask_vol = sum(qty for _, qty in asks[:depth_levels])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def compute_trade_intensity(
        self,
        timestamps: list[float],
        window_seconds: float = 60.0,
    ) -> float:
        """Trade intensity — trades per second in recent window."""
        if len(timestamps) < 2:
            return 0.0

        now = timestamps[-1]
        recent = [t for t in timestamps if now - t <= window_seconds]
        if len(recent) < 2:
            return 0.0

        duration = recent[-1] - recent[0]
        if duration <= 0:
            return 0.0

        return (len(recent) - 1) / duration

    def estimate_market_impact(
        self,
        quantity: float,
        adv: float,
        volatility: float,
        spread_bps: float = 0.0,
    ) -> float:
        """Estimate market impact using a square-root model.

        Impact = sigma * (Q / ADV)^0.5 + spread/2

        Args:
            quantity: Order quantity
            adv: Average daily volume
            volatility: Daily volatility
            spread_bps: Bid-ask spread in bps
        """
        if adv <= 0:
            return 0.0

        participation = quantity / adv
        temporary_impact = volatility * np.sqrt(participation) * 10000  # bps
        spread_cost = spread_bps / 2

        return float(temporary_impact + spread_cost)

    def analyze_tick_data(
        self,
        ticks: list[Tick],
        order_book: OrderBookSnapshot | None = None,
    ) -> MicrostructureMetrics:
        """Full microstructure analysis from tick data."""
        if len(ticks) < 20:
            return MicrostructureMetrics()

        prices = np.array([t.price for t in ticks])
        volumes = np.array([t.quantity for t in ticks])
        price_changes = np.diff(prices)

        # VPIN
        vpin = self.compute_vpin(volumes[1:], price_changes)

        # Kyle's Lambda
        # Estimate signed volume using tick rule
        signed_vol = np.zeros(len(price_changes))
        for i in range(len(price_changes)):
            signed_vol[i] = volumes[i + 1] if price_changes[i] > 0 else -volumes[i + 1]
        kyle_lambda = self.compute_kyle_lambda(price_changes, signed_vol)

        # Effective spread
        effective_spread = 0.0
        if order_book and order_book.mid_price:
            last_price = ticks[-1].price
            effective_spread = self.compute_effective_spread(
                last_price, order_book.mid_price, ticks[-1].quantity
            )

        # Order flow imbalance
        ofi = 0.0
        if order_book:
            bids = [(l.price, l.quantity) for l in order_book.bids]
            asks = [(l.price, l.quantity) for l in order_book.asks]
            ofi = self.compute_order_flow_imbalance(bids, asks)

        # Trade intensity
        timestamps = [t.timestamp.timestamp() for t in ticks]
        trade_intensity = self.compute_trade_intensity(timestamps)

        # Overall toxicity score (composite)
        toxicity = 0.4 * vpin + 0.3 * min(1.0, kyle_lambda * 1000) + 0.3 * abs(ofi)

        return MicrostructureMetrics(
            vpin=vpin,
            kyle_lambda=kyle_lambda,
            effective_spread_bps=effective_spread,
            order_flow_imbalance=ofi,
            trade_intensity=trade_intensity,
            toxicity=toxicity,
        )
