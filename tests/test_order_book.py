"""Tests for order book analysis and trade aggregation."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from tradingbot.data.order_book import OrderBookAnalyzer, OrderBookFeatures
from tradingbot.data.trade_aggregator import TradeAggregator
from tradingbot.core.types import OrderBookLevel, OrderBookSnapshot, Tick
from tradingbot.core.enums import Side, Timeframe


class TestOrderBookAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return OrderBookAnalyzer()

    @pytest.fixture
    def book(self):
        return OrderBookSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            exchange="test",
            bids=[
                OrderBookLevel(price=49900, quantity=2.0),
                OrderBookLevel(price=49800, quantity=3.0),
                OrderBookLevel(price=49700, quantity=5.0),
            ],
            asks=[
                OrderBookLevel(price=50100, quantity=1.5),
                OrderBookLevel(price=50200, quantity=2.5),
                OrderBookLevel(price=50300, quantity=4.0),
            ],
        )

    def test_extract_features(self, analyzer, book):
        features = analyzer.extract_features(book)
        assert isinstance(features, OrderBookFeatures)
        assert features.mid_price == 50000.0
        assert features.spread_bps > 0
        assert features.bid_depth_total == 10.0
        assert features.ask_depth_total == 8.0
        assert features.imbalance > 0  # More bid volume

    def test_weighted_mid_price(self, analyzer, book):
        features = analyzer.extract_features(book)
        # Weighted mid should be between best bid and best ask
        assert 49900 < features.weighted_mid_price < 50100

    def test_micro_price(self, analyzer, book):
        features = analyzer.extract_features(book)
        assert 49900 < features.micro_price < 50100

    def test_depth_features(self, analyzer, book):
        features = analyzer.compute_depth_features(book, n_levels=3)
        assert len(features) == 8  # 4 per side
        assert all(v >= 0 for v in features[:4])  # Bid features

    def test_empty_book(self, analyzer):
        empty = OrderBookSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT", exchange="test",
            bids=[], asks=[],
        )
        features = analyzer.extract_features(empty)
        assert features.mid_price == 0


class TestTradeAggregator:
    @pytest.fixture
    def aggregator(self):
        return TradeAggregator()

    def test_time_bar_aggregation(self, aggregator):
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        # Send ticks within same minute
        tick1 = Tick(timestamp=base_time, symbol="BTC/USDT", price=50000, quantity=1.0, side=Side.BUY, exchange="test")
        tick2 = Tick(timestamp=base_time.replace(second=30), symbol="BTC/USDT", price=50100, quantity=0.5, side=Side.SELL, exchange="test")

        result1 = aggregator.aggregate_time_bar(tick1, Timeframe.M1)
        assert result1 is None  # No bar completed yet

        result2 = aggregator.aggregate_time_bar(tick2, Timeframe.M1)
        assert result2 is None  # Still same minute

    def test_volume_bar_aggregation(self, aggregator):
        tick1 = Tick(timestamp=datetime.now(timezone.utc), symbol="BTC/USDT", price=50000, quantity=60, side=Side.BUY, exchange="test")
        tick2 = Tick(timestamp=datetime.now(timezone.utc), symbol="BTC/USDT", price=50100, quantity=50, side=Side.SELL, exchange="test")

        result1 = aggregator.aggregate_volume_bar(tick1, target_volume=100)
        assert result1 is None  # Not enough volume yet

        result2 = aggregator.aggregate_volume_bar(tick2, target_volume=100)
        assert result2 is not None  # Volume exceeded 100
        assert result2.open == 50000
        assert result2.close == 50100
        assert result2.high == 50100

    def test_vwap_computation(self, aggregator):
        ticks = [
            Tick(timestamp=datetime.now(timezone.utc), symbol="BTC/USDT", price=50000, quantity=10, side=Side.BUY, exchange="test"),
            Tick(timestamp=datetime.now(timezone.utc), symbol="BTC/USDT", price=50100, quantity=5, side=Side.SELL, exchange="test"),
            Tick(timestamp=datetime.now(timezone.utc), symbol="BTC/USDT", price=50050, quantity=15, side=Side.BUY, exchange="test"),
        ]
        timestamps, vwap = aggregator.compute_vwap(ticks)
        assert len(vwap) == 3
        # VWAP should be volume-weighted
        expected = (50000*10 + 50100*5 + 50050*15) / 30
        assert abs(vwap[-1] - expected) < 0.01

    def test_volume_profile(self, aggregator):
        from tradingbot.core.types import OHLCVBar
        bars = [
            OHLCVBar(timestamp=datetime.now(timezone.utc), symbol="BTC/USDT", timeframe=Timeframe.H1,
                     open=50000, high=50500, low=49500, close=50200, volume=100, exchange="test"),
            OHLCVBar(timestamp=datetime.now(timezone.utc), symbol="BTC/USDT", timeframe=Timeframe.H1,
                     open=50200, high=50800, low=50000, close=50600, volume=150, exchange="test"),
        ]
        vp = aggregator.build_volume_profile(bars, n_bins=10)
        assert vp.poc_price > 0
        assert vp.value_area_high > vp.value_area_low
