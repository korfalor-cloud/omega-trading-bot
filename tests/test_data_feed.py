"""Tests for DataFeedManager — bar aggregation, data quality checks, stream management."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from tradingbot.core.enums import Timeframe
from tradingbot.core.types import OHLCVBar, Tick, Side
from tradingbot.engine.data_feed import (
    BarAggregator,
    DataQualityChecker,
    DataQualityReport,
    DataFeedManager,
    _floor_timestamp,
)


# ---------------------------------------------------------------------------
# DataQualityChecker
# ---------------------------------------------------------------------------

class TestDataQualityChecker:
    @pytest.fixture
    def checker(self):
        return DataQualityChecker(
            max_staleness_seconds=300,
            max_price_change_pct=0.10,
            gap_tolerance_seconds=60,
        )

    def _make_bar(self, close=50000.0, ts=None, symbol="BTC/USDT"):
        ts = ts or datetime.now(timezone.utc)
        return OHLCVBar(
            timestamp=ts, symbol=symbol, timeframe=Timeframe.H1,
            open=close - 100, high=close + 200, low=close - 200,
            close=close, volume=100, exchange="binance",
        )

    def test_valid_bar_passes(self, checker):
        bar = self._make_bar()
        report = checker.check_bar(bar)
        assert report.passed is True

    def test_nan_detected(self, checker):
        bar = self._make_bar()
        # Manually create bar with NaN
        bad = OHLCVBar(
            timestamp=datetime.now(timezone.utc), symbol="BTC/USDT", timeframe=Timeframe.H1,
            open=float("nan"), high=51000, low=49000, close=50000, volume=100, exchange="binance",
        )
        report = checker.check_bar(bad)
        assert report.passed is False
        assert report.nan_detected is True

    def test_price_spike_detected(self, checker):
        ts = datetime.now(timezone.utc)
        bar1 = self._make_bar(close=50000.0, ts=ts)
        checker.check_bar(bar1)
        # 15% spike exceeds 10% threshold
        bar2 = self._make_bar(close=57500.0, ts=ts + timedelta(minutes=1))
        report = checker.check_bar(bar2)
        assert report.passed is False
        assert report.price_spike is True

    def test_normal_price_change_passes(self, checker):
        ts = datetime.now(timezone.utc)
        bar1 = self._make_bar(close=50000.0, ts=ts)
        checker.check_bar(bar1)
        bar2 = self._make_bar(close=52000.0, ts=ts + timedelta(minutes=1))
        report = checker.check_bar(bar2)
        assert report.passed is True

    def test_tick_nan_detected(self, checker):
        tick = Tick(
            timestamp=datetime.now(timezone.utc), symbol="BTC/USDT",
            price=float("nan"), quantity=1.0, side=Side.BUY, exchange="binance",
        )
        report = checker.check_tick(tick)
        assert report.passed is False
        assert report.nan_detected is True

    def test_tick_negative_price(self, checker):
        tick = Tick(
            timestamp=datetime.now(timezone.utc), symbol="BTC/USDT",
            price=-1.0, quantity=1.0, side=Side.BUY, exchange="binance",
        )
        report = checker.check_tick(tick)
        assert report.passed is False

    def test_tick_spike_detected(self, checker):
        ts = datetime.now(timezone.utc)
        t1 = Tick(timestamp=ts, symbol="BTC/USDT", price=50000.0, quantity=1.0, side=Side.BUY, exchange="binance")
        checker.check_tick(t1)
        t2 = Tick(timestamp=ts, symbol="BTC/USDT", price=60000.0, quantity=1.0, side=Side.BUY, exchange="binance")
        report = checker.check_tick(t2)
        assert report.passed is False
        assert report.price_spike is True


# ---------------------------------------------------------------------------
# BarAggregator
# ---------------------------------------------------------------------------

class TestBarAggregator:
    def test_first_tick_returns_none(self):
        agg = BarAggregator()
        tick = Tick(
            timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            symbol="BTC/USDT", price=50000.0, quantity=1.0, side=Side.BUY, exchange="binance",
        )
        result = agg.process_tick(tick, Timeframe.M1)
        assert result is None

    def test_tick_aggregation_same_bar(self):
        agg = BarAggregator()
        ts1 = datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 1, 0, 0, 15, tzinfo=timezone.utc)

        t1 = Tick(timestamp=ts1, symbol="BTC/USDT", price=50000.0, quantity=1.0, side=Side.BUY, exchange="binance")
        t2 = Tick(timestamp=ts2, symbol="BTC/USDT", price=50100.0, quantity=2.0, side=Side.BUY, exchange="binance")

        assert agg.process_tick(t1, Timeframe.M1) is None
        result = agg.process_tick(t2, Timeframe.M1)
        assert result is None  # same minute

    def test_bar_completes_on_new_interval(self):
        agg = BarAggregator()
        ts1 = datetime(2024, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 1, 0, 1, 5, tzinfo=timezone.utc)

        t1 = Tick(timestamp=ts1, symbol="BTC/USDT", price=50000.0, quantity=1.0, side=Side.BUY, exchange="binance")
        t2 = Tick(timestamp=ts2, symbol="BTC/USDT", price=50100.0, quantity=2.0, side=Side.BUY, exchange="binance")

        agg.process_tick(t1, Timeframe.M1)
        bar = agg.process_tick(t2, Timeframe.M1)

        assert bar is not None
        assert bar.open == 50000.0
        assert bar.close == 50000.0  # close of first interval
        assert bar.high == 50000.0
        assert bar.low == 50000.0

    def test_tick_timeframe_returns_none(self):
        agg = BarAggregator()
        tick = Tick(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            symbol="BTC/USDT", price=50000.0, quantity=1.0, side=Side.BUY, exchange="binance",
        )
        assert agg.process_tick(tick, Timeframe.TICK) is None


# ---------------------------------------------------------------------------
# _floor_timestamp
# ---------------------------------------------------------------------------

class TestFloorTimestamp:
    def test_floors_to_minute(self):
        ts = datetime(2024, 1, 1, 0, 1, 30, tzinfo=timezone.utc)
        floored = _floor_timestamp(ts, 60)
        assert floored.second == 0
        assert floored.minute == 1

    def test_floors_to_hour(self):
        ts = datetime(2024, 1, 1, 1, 30, 0, tzinfo=timezone.utc)
        floored = _floor_timestamp(ts, 3600)
        assert floored.minute == 0
        assert floored.hour == 1


# ---------------------------------------------------------------------------
# DataQualityReport
# ---------------------------------------------------------------------------

class TestDataQualityReport:
    def test_defaults(self):
        report = DataQualityReport()
        assert report.passed is True
        assert report.stale is False
        assert report.nan_detected is False
        assert report.price_spike is False


# ---------------------------------------------------------------------------
# DataFeedManager
# ---------------------------------------------------------------------------

class TestDataFeedManager:
    def test_active_stream_count_initial(self):
        bus = __import__("tradingbot.core.events", fromlist=["EventBus"]).EventBus()
        manager = DataFeedManager(exchanges={}, event_bus=bus, symbols=["BTC/USDT"])
        assert manager.active_stream_count == 0
