"""Async data feed manager.

Handles multiple concurrent data streams (ticks, bars, order books) from
multiple exchanges with:
- Automatic bar aggregation from tick streams
- Inline data quality checks (stale data, NaN, price spikes)
- Automatic reconnection on websocket / REST failures
- Fan-out distribution of bars/ticks to strategy callbacks via EventBus
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from ..core.enums import Timeframe
from ..core.events import Event, EventBus
from ..core.interfaces import ExchangeAdapter
from ..core.types import OHLCVBar, Tick

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------


@dataclass
class DataQualityReport:
    """Result of an inline data quality check."""
    passed: bool = True
    stale: bool = False
    nan_detected: bool = False
    price_spike: bool = False
    gap_detected: bool = False
    message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DataQualityChecker:
    """Inline quality gate for every incoming bar / tick."""

    def __init__(
        self,
        max_staleness_seconds: int = 300,
        max_price_change_pct: float = 0.10,  # 10% spike threshold
        gap_tolerance_seconds: int = 60,
    ) -> None:
        self._max_staleness = max_staleness_seconds
        self._max_price_change = max_price_change_pct
        self._gap_tolerance = gap_tolerance_seconds
        self._last_timestamp: dict[str, datetime] = {}
        self._last_price: dict[str, float] = {}

    def check_bar(self, bar: OHLCVBar) -> DataQualityReport:
        report = DataQualityReport(timestamp=bar.timestamp)
        key = f"{bar.symbol}:{bar.exchange}"

        # NaN check
        for attr in ("open", "high", "low", "close", "volume"):
            if math.isnan(getattr(bar, attr)):
                report.nan_detected = True
                report.passed = False
                report.message = f"NaN in {attr}"
                return report

        # Staleness check
        if key in self._last_timestamp:
            delta = (bar.timestamp - self._last_timestamp[key]).total_seconds()
            if delta > self._max_staleness:
                report.stale = True
                report.passed = False
                report.message = f"Stale: {delta:.0f}s since last bar"
                return report

        # Gap check
        if key in self._last_timestamp:
            expected_gap = bar.timestamp - self._last_timestamp[key]
            if expected_gap > timedelta(seconds=bar.timeframe.seconds + self._gap_tolerance):
                report.gap_detected = True
                # Gap is a warning but does not fail the bar
                report.message = f"Gap detected: {expected_gap}"

        # Price spike check
        if key in self._last_price and self._last_price[key] > 0:
            change = abs(bar.close - self._last_price[key]) / self._last_price[key]
            if change > self._max_price_change:
                report.price_spike = True
                report.passed = False
                report.message = f"Price spike: {change:.2%}"
                return report

        # Update state
        self._last_timestamp[key] = bar.timestamp
        self._last_price[key] = bar.close
        return report

    def check_tick(self, tick: Tick) -> DataQualityReport:
        report = DataQualityReport(timestamp=tick.timestamp)
        key = f"{tick.symbol}:{tick.exchange}"

        if math.isnan(tick.price) or math.isnan(tick.quantity):
            report.nan_detected = True
            report.passed = False
            report.message = "NaN in tick"
            return report

        if tick.price <= 0:
            report.passed = False
            report.message = f"Non-positive price: {tick.price}"
            return report

        if key in self._last_price and self._last_price[key] > 0:
            change = abs(tick.price - self._last_price[key]) / self._last_price[key]
            if change > self._max_price_change:
                report.price_spike = True
                report.passed = False
                report.message = f"Tick price spike: {change:.2%}"
                return report

        self._last_price[key] = tick.price
        return report


# ---------------------------------------------------------------------------
# Bar aggregator (tick -> bar)
# ---------------------------------------------------------------------------

class BarAggregator:
    """Aggregates ticks into OHLCV bars for each (symbol, timeframe) pair."""

    def __init__(self) -> None:
        # key: (symbol, exchange, timeframe_seconds) -> partial bar state
        self._bars: dict[tuple[str, str, int], dict[str, Any]] = {}

    def process_tick(self, tick: Tick, timeframe: Timeframe) -> Optional[OHLCVBar]:
        """Feed a tick; returns a completed bar when the interval closes."""
        tf_seconds = timeframe.seconds
        if tf_seconds <= 0:
            return None  # can't aggregate tick timeframe

        key = (tick.symbol, tick.exchange, tf_seconds)
        bar_start = _floor_timestamp(tick.timestamp, tf_seconds)
        state = self._bars.get(key)

        if state is not None and state["start"] != bar_start:
            # Previous bar is complete
            completed = _state_to_bar(state, tick.symbol, timeframe, tick.exchange)
            # Start new bar
            self._bars[key] = {
                "start": bar_start,
                "open": tick.price,
                "high": tick.price,
                "low": tick.price,
                "close": tick.price,
                "volume": tick.quantity,
                "trades": 1,
            }
            return completed

        if state is None:
            self._bars[key] = {
                "start": bar_start,
                "open": tick.price,
                "high": tick.price,
                "low": tick.price,
                "close": tick.price,
                "volume": tick.quantity,
                "trades": 1,
            }
            return None

        # Update partial bar
        state["high"] = max(state["high"], tick.price)
        state["low"] = min(state["low"], tick.price)
        state["close"] = tick.price
        state["volume"] += tick.quantity
        state["trades"] += 1
        return None


def _floor_timestamp(ts: datetime, interval_seconds: int) -> datetime:
    epoch = int(ts.timestamp())
    floored = epoch - (epoch % interval_seconds)
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def _state_to_bar(
    state: dict[str, Any],
    symbol: str,
    timeframe: Timeframe,
    exchange: str,
) -> OHLCVBar:
    return OHLCVBar(
        timestamp=state["start"],
        symbol=symbol,
        timeframe=timeframe,
        open=state["open"],
        high=state["high"],
        low=state["low"],
        close=state["close"],
        volume=state["volume"],
        exchange=exchange,
        trades_count=state.get("trades", 0),
    )


# ---------------------------------------------------------------------------
# DataFeedManager
# ---------------------------------------------------------------------------

# Reconnect back-off settings
_INITIAL_BACKOFF = 1.0
_MAX_BACKOFF = 120.0
_BACKOFF_MULTIPLIER = 2.0


class DataFeedManager:
    """Manages concurrent data streams from multiple exchanges.

    Supports three stream types:
    * Candle streams  -> direct bar delivery
    * Tick streams    -> aggregated into bars via ``BarAggregator``
    * Order book streams -> snapshots delivered via EventBus

    All data passes through ``DataQualityChecker`` before distribution.
    """

    def __init__(
        self,
        exchanges: dict[str, ExchangeAdapter],
        event_bus: EventBus,
        symbols: list[str],
        tick_aggregation_timeframes: Optional[list[Timeframe]] = None,
    ) -> None:
        self._exchanges = exchanges
        self._event_bus = event_bus
        self._symbols = symbols

        self._tick_aggregation_tfs = tick_aggregation_timeframes or [
            Timeframe.M1, Timeframe.M5, Timeframe.M15,
        ]

        self._quality_checker = DataQualityChecker()
        self._aggregator = BarAggregator()

        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._active_streams = 0
        self._reconnect_count = 0

    @property
    def active_stream_count(self) -> int:
        return self._active_streams

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all configured data streams."""
        self._running = True

        for exchange_name, exchange in self._exchanges.items():
            for symbol in self._symbols:
                # Candle stream (primary)
                self._tasks.append(
                    asyncio.create_task(
                        self._run_candle_stream(exchange, symbol, Timeframe.M1, exchange_name),
                        name=f"candle:{exchange_name}:{symbol}",
                    )
                )
                # Tick stream -> bar aggregation
                self._tasks.append(
                    asyncio.create_task(
                        self._run_tick_stream(exchange, symbol, exchange_name),
                        name=f"tick:{exchange_name}:{symbol}",
                    )
                )
                # Order book stream
                self._tasks.append(
                    asyncio.create_task(
                        self._run_orderbook_stream(exchange, symbol, exchange_name),
                        name=f"book:{exchange_name}:{symbol}",
                    )
                )

        logger.info("DataFeedManager started %d stream tasks", len(self._tasks))
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Cancel all stream tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("DataFeedManager stopped")

    # ------------------------------------------------------------------
    # Stream runners (with reconnection)
    # ------------------------------------------------------------------

    async def _run_candle_stream(
        self,
        exchange: ExchangeAdapter,
        symbol: str,
        timeframe: Timeframe,
        exchange_name: str,
    ) -> None:
        backoff = _INITIAL_BACKOFF
        while self._running:
            try:
                self._active_streams += 1
                async for bar in exchange.watch_candles(symbol, timeframe):
                    if not self._running:
                        break
                    qr = self._quality_checker.check_bar(bar)
                    if not qr.passed:
                        logger.warning("DQ fail candle %s/%s: %s", exchange_name, symbol, qr.message)
                        await self._event_bus.publish(Event.DATA_GAP_DETECTED, qr)
                        continue
                    if qr.gap_detected:
                        await self._event_bus.publish(Event.DATA_GAP_DETECTED, qr)
                    await self._event_bus.publish(Event.BAR_CLOSED, bar)
            except asyncio.CancelledError:
                break
            except Exception:
                self._active_streams -= 1
                self._reconnect_count += 1
                logger.exception(
                    "Candle stream %s/%s lost, reconnecting in %.1fs",
                    exchange_name, symbol, backoff,
                )
                lost_info = {
                    "stream": "candle",
                    "exchange": exchange_name,
                    "symbol": symbol,
                }
                await self._event_bus.publish(Event.CONNECTION_LOST, lost_info)
                await asyncio.sleep(backoff)
                backoff = min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF)
                await self._event_bus.publish(Event.CONNECTION_RESTORED, lost_info)
            else:
                self._active_streams -= 1

    async def _run_tick_stream(
        self,
        exchange: ExchangeAdapter,
        symbol: str,
        exchange_name: str,
    ) -> None:
        backoff = _INITIAL_BACKOFF
        while self._running:
            try:
                self._active_streams += 1
                async for tick in exchange.watch_trades(symbol):
                    if not self._running:
                        break
                    qr = self._quality_checker.check_tick(tick)
                    if not qr.passed:
                        logger.warning(
                            "DQ fail tick %s/%s: %s",
                            exchange_name, symbol, qr.message,
                        )
                        continue

                    await self._event_bus.publish(Event.TICK_RECEIVED, tick)

                    # Aggregate into bars for each configured timeframe
                    for tf in self._tick_aggregation_tfs:
                        bar = self._aggregator.process_tick(tick, tf)
                        if bar is not None:
                            bar_qr = self._quality_checker.check_bar(bar)
                            if bar_qr.passed:
                                await self._event_bus.publish(Event.BAR_CLOSED, bar)
            except asyncio.CancelledError:
                break
            except Exception:
                self._active_streams -= 1
                self._reconnect_count += 1
                logger.exception(
                    "Tick stream %s/%s lost, reconnecting in %.1fs",
                    exchange_name, symbol, backoff,
                )
                lost_info = {
                    "stream": "tick",
                    "exchange": exchange_name,
                    "symbol": symbol,
                }
                await self._event_bus.publish(Event.CONNECTION_LOST, lost_info)
                await asyncio.sleep(backoff)
                backoff = min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF)
                await self._event_bus.publish(Event.CONNECTION_RESTORED, lost_info)
            else:
                self._active_streams -= 1

    async def _run_orderbook_stream(
        self,
        exchange: ExchangeAdapter,
        symbol: str,
        exchange_name: str,
    ) -> None:
        backoff = _INITIAL_BACKOFF
        while self._running:
            try:
                self._active_streams += 1
                async for snapshot in exchange.watch_order_book(symbol):
                    if not self._running:
                        break
                    await self._event_bus.publish(Event.ORDER_BOOK_UPDATE, snapshot)
            except asyncio.CancelledError:
                break
            except Exception:
                self._active_streams -= 1
                self._reconnect_count += 1
                logger.exception(
                    "Orderbook stream %s/%s lost, reconnecting in %.1fs",
                    exchange_name, symbol, backoff,
                )
                lost_info = {
                    "stream": "orderbook",
                    "exchange": exchange_name,
                    "symbol": symbol,
                }
                await self._event_bus.publish(Event.CONNECTION_LOST, lost_info)
                await asyncio.sleep(backoff)
                backoff = min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF)
                await self._event_bus.publish(Event.CONNECTION_RESTORED, lost_info)
            else:
                self._active_streams -= 1
