"""Trade Data Aggregation.

Aggregates raw trades into bars at various timeframes and computes
volume profiles, VWAP, and trade flow metrics.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from ..core.enums import Timeframe
from ..core.types import OHLCVBar, Tick

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfile:
    """Volume profile at specific price levels."""
    price_levels: np.ndarray = field(default_factory=lambda: np.array([]))
    buy_volumes: np.ndarray = field(default_factory=lambda: np.array([]))
    sell_volumes: np.ndarray = field(default_factory=lambda: np.array([]))
    total_volumes: np.ndarray = field(default_factory=lambda: np.array([]))
    poc_price: float = 0.0  # Point of Control (highest volume price)
    value_area_high: float = 0.0
    value_area_low: float = 0.0


class TradeAggregator:
    """Aggregates raw trades into OHLCV bars and computes volume profiles.

    Supports:
    - Time-based aggregation (any timeframe)
    - Volume-based aggregation (fixed volume bars)
    - Tick-based aggregation (fixed tick count bars)
    - Volume profile construction
    - VWAP computation
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.tz = config.get("timezone", "UTC")
        self._current_bars: dict[str, dict] = {}  # symbol -> current bar state
        self._completed_bars: dict[str, list[OHLCVBar]] = defaultdict(list)
        self._ticks: dict[str, list[Tick]] = defaultdict(list)

    def aggregate_time_bar(
        self,
        tick: Tick,
        timeframe: Timeframe = Timeframe.M1,
    ) -> Optional[OHLCVBar]:
        """Aggregate tick into time-based OHLCV bar.

        Returns completed bar when timeframe boundary is crossed.
        """
        symbol = tick.symbol
        tf_seconds = timeframe.seconds

        if symbol not in self._current_bars:
            self._current_bars[symbol] = {
                "open": tick.price,
                "high": tick.price,
                "low": tick.price,
                "close": tick.price,
                "volume": tick.quantity,
                "trades": 1,
                "vwap_num": tick.price * tick.quantity,
                "vwap_den": tick.quantity,
                "start_time": self._floor_timestamp(tick.timestamp, tf_seconds),
            }
            self._ticks[symbol].append(tick)
            return None

        bar = self._current_bars[symbol]
        self._ticks[symbol].append(tick)

        # Check if we crossed a timeframe boundary
        current_start = self._floor_timestamp(tick.timestamp, tf_seconds)
        if current_start > bar["start_time"]:
            # Complete the current bar
            completed = OHLCVBar(
                timestamp=bar["start_time"],
                symbol=symbol,
                timeframe=timeframe,
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
                volume=bar["volume"],
                exchange=tick.exchange,
                trades_count=bar["trades"],
                vwap=bar["vwap_num"] / bar["vwap_den"] if bar["vwap_den"] > 0 else 0,
            )
            self._completed_bars[symbol].append(completed)

            # Start new bar
            self._current_bars[symbol] = {
                "open": tick.price,
                "high": tick.price,
                "low": tick.price,
                "close": tick.price,
                "volume": tick.quantity,
                "trades": 1,
                "vwap_num": tick.price * tick.quantity,
                "vwap_den": tick.quantity,
                "start_time": current_start,
            }
            self._ticks[symbol] = [tick]
            return completed

        # Update current bar
        bar["high"] = max(bar["high"], tick.price)
        bar["low"] = min(bar["low"], tick.price)
        bar["close"] = tick.price
        bar["volume"] += tick.quantity
        bar["trades"] += 1
        bar["vwap_num"] += tick.price * tick.quantity
        bar["vwap_den"] += tick.quantity

        return None

    def aggregate_volume_bar(
        self,
        tick: Tick,
        target_volume: float = 100.0,
    ) -> Optional[OHLCVBar]:
        """Aggregate into fixed-volume bars."""
        symbol = tick.symbol

        if symbol not in self._current_bars:
            self._current_bars[symbol] = {
                "open": tick.price,
                "high": tick.price,
                "low": tick.price,
                "close": tick.price,
                "volume": tick.quantity,
                "trades": 1,
                "start_time": tick.timestamp,
            }
            return None

        bar = self._current_bars[symbol]
        bar["high"] = max(bar["high"], tick.price)
        bar["low"] = min(bar["low"], tick.price)
        bar["close"] = tick.price
        bar["volume"] += tick.quantity
        bar["trades"] += 1

        if bar["volume"] >= target_volume:
            completed = OHLCVBar(
                timestamp=bar["start_time"],
                symbol=symbol,
                timeframe=Timeframe.TICK,
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
                volume=bar["volume"],
                exchange=tick.exchange,
                trades_count=bar["trades"],
            )
            self._completed_bars[symbol].append(completed)

            self._current_bars[symbol] = {
                "open": tick.price,
                "high": tick.price,
                "low": tick.price,
                "close": tick.price,
                "volume": 0.0,
                "trades": 0,
                "start_time": tick.timestamp,
            }
            return completed

        return None

    def compute_vwap(
        self,
        ticks: list[Tick],
        anchor_time: datetime | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute running VWAP from ticks.

        Returns:
            timestamps: Array of timestamps
            vwap_values: Array of VWAP values
        """
        if not ticks:
            return np.array([]), np.array([])

        timestamps = []
        vwap_values = []
        cum_volume = 0.0
        cum_pv = 0.0

        for tick in ticks:
            cum_pv += tick.price * tick.quantity
            cum_volume += tick.quantity
            timestamps.append(tick.timestamp)
            vwap_values.append(cum_pv / cum_volume if cum_volume > 0 else tick.price)

        return np.array(timestamps), np.array(vwap_values)

    def build_volume_profile(
        self,
        bars: list[OHLCVBar],
        n_bins: int = 50,
    ) -> VolumeProfile:
        """Build volume profile from OHLCV bars."""
        if not bars:
            return VolumeProfile()

        prices = np.array([b.close for b in bars])
        volumes = np.array([b.volume for b in bars])

        price_min = np.min([b.low for b in bars])
        price_max = np.max([b.high for b in bars])

        if price_min == price_max:
            return VolumeProfile(
                price_levels=np.array([price_min]),
                total_volumes=np.array([np.sum(volumes)]),
                poc_price=price_min,
            )

        # Create price bins
        bins = np.linspace(price_min, price_max, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        buy_volumes = np.zeros(n_bins)
        sell_volumes = np.zeros(n_bins)

        for bar in bars:
            bin_idx = np.digitize(bar.close, bins) - 1
            bin_idx = max(0, min(n_bins - 1, bin_idx))

            if bar.close > bar.open:
                buy_volumes[bin_idx] += bar.volume
            else:
                sell_volumes[bin_idx] += bar.volume

        total_volumes = buy_volumes + sell_volumes

        # Point of Control
        poc_idx = np.argmax(total_volumes)
        poc_price = bin_centers[poc_idx]

        # Value Area (70% of volume)
        sorted_indices = np.argsort(total_volumes)[::-1]
        cumulative = 0.0
        total_vol = np.sum(total_volumes)
        va_indices = []
        for idx in sorted_indices:
            va_indices.append(idx)
            cumulative += total_volumes[idx]
            if cumulative >= 0.7 * total_vol:
                break

        va_high = bin_centers[max(va_indices)]
        va_low = bin_centers[min(va_indices)]

        return VolumeProfile(
            price_levels=bin_centers,
            buy_volumes=buy_volumes,
            sell_volumes=sell_volumes,
            total_volumes=total_volumes,
            poc_price=poc_price,
            value_area_high=va_high,
            value_area_low=va_low,
        )

    def get_completed_bars(self, symbol: str) -> list[OHLCVBar]:
        return list(self._completed_bars.get(symbol, []))

    def get_tick_count(self, symbol: str) -> int:
        return len(self._ticks.get(symbol, []))

    @staticmethod
    def _floor_timestamp(ts: datetime, seconds: int) -> datetime:
        """Floor timestamp to the nearest interval."""
        epoch = ts.timestamp()
        floored = epoch - (epoch % seconds)
        return datetime.fromtimestamp(floored, tz=ts.tzinfo)
