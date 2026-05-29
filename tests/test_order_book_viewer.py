"""Tests for OrderBookViewer — live order book display."""
from __future__ import annotations

import pytest

from tradingbot.monitoring.order_book_viewer import OrderBookViewer, OrderBookSnapshot


class TestOrderBookViewer:
    @pytest.fixture
    def viewer(self):
        return OrderBookViewer()

    @pytest.fixture
    def sample_bids(self):
        return [(49900.0, 2.0), (49800.0, 3.0), (49700.0, 5.0)]

    @pytest.fixture
    def sample_asks(self):
        return [(50100.0, 1.5), (50200.0, 2.5), (50300.0, 4.0)]

    def test_update_creates_snapshot(self, viewer, sample_bids, sample_asks):
        snapshot = viewer.update("BTC/USDT", sample_bids, sample_asks)
        assert isinstance(snapshot, OrderBookSnapshot)
        assert snapshot.symbol == "BTC/USDT"

    def test_mid_price(self, viewer, sample_bids, sample_asks):
        snapshot = viewer.update("BTC/USDT", sample_bids, sample_asks)
        assert snapshot.mid_price == pytest.approx((49900 + 50100) / 2)

    def test_spread(self, viewer, sample_bids, sample_asks):
        snapshot = viewer.update("BTC/USDT", sample_bids, sample_asks)
        assert snapshot.spread == pytest.approx(200.0)

    def test_spread_bps(self, viewer, sample_bids, sample_asks):
        snapshot = viewer.update("BTC/USDT", sample_bids, sample_asks)
        expected_bps = 200.0 / snapshot.mid_price * 10000
        assert snapshot.spread_bps == pytest.approx(expected_bps, rel=0.01)

    def test_bid_depth(self, viewer, sample_bids, sample_asks):
        snapshot = viewer.update("BTC/USDT", sample_bids, sample_asks)
        assert snapshot.bid_depth == pytest.approx(10.0)

    def test_ask_depth(self, viewer, sample_bids, sample_asks):
        snapshot = viewer.update("BTC/USDT", sample_bids, sample_asks)
        assert snapshot.ask_depth == pytest.approx(8.0)

    def test_imbalance(self, viewer, sample_bids, sample_asks):
        snapshot = viewer.update("BTC/USDT", sample_bids, sample_asks)
        # More bid volume than ask volume -> positive imbalance
        assert snapshot.imbalance > 0
        expected = (10.0 - 8.0) / (10.0 + 8.0)
        assert snapshot.imbalance == pytest.approx(expected)

    def test_bids_sorted_descending(self, viewer, sample_bids, sample_asks):
        snapshot = viewer.update("BTC/USDT", sample_bids, sample_asks)
        prices = [p for p, _ in snapshot.bids]
        assert prices == sorted(prices, reverse=True)

    def test_asks_sorted_ascending(self, viewer, sample_bids, sample_asks):
        snapshot = viewer.update("BTC/USDT", sample_bids, sample_asks)
        prices = [p for p, _ in snapshot.asks]
        assert prices == sorted(prices)

    def test_get_snapshot(self, viewer, sample_bids, sample_asks):
        viewer.update("BTC/USDT", sample_bids, sample_asks)
        snapshot = viewer.get_snapshot("BTC/USDT")
        assert snapshot.symbol == "BTC/USDT"
        assert snapshot.mid_price > 0

    def test_get_snapshot_unknown(self, viewer):
        snapshot = viewer.get_snapshot("UNKNOWN/USDT")
        assert snapshot.mid_price == 0

    def test_format_book(self, viewer, sample_bids, sample_asks):
        viewer.update("BTC/USDT", sample_bids, sample_asks)
        text = viewer.format_book("BTC/USDT")
        assert "BTC/USDT" in text
        assert "Mid" in text
        assert "Imbalance" in text

    def test_format_book_no_data(self, viewer):
        text = viewer.format_book("UNKNOWN/USDT")
        assert text == "No data"

    def test_empty_order_book(self, viewer):
        snapshot = viewer.update("BTC/USDT", [], [])
        assert snapshot.mid_price == 0
        assert snapshot.spread == 0
        assert snapshot.imbalance == 0
