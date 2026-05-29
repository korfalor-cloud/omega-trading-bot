"""Tests for TradeBlotter — real-time trade display."""
from __future__ import annotations

import pytest

from tradingbot.monitoring.trade_blotter import BlotterEntry, TradeBlotter


class TestTradeBlotter:
    @pytest.fixture
    def blotter(self):
        return TradeBlotter()

    @pytest.fixture
    def filled_blotter(self):
        b = TradeBlotter()
        b.add_trade("BTC/USDT", "buy", 50000.0, 0.1, fee=5.0, strategy_id="s1", pnl=100.0)
        b.add_trade("ETH/USDT", "sell", 3000.0, 1.0, fee=3.0, strategy_id="s2", pnl=-50.0)
        b.add_trade("BTC/USDT", "sell", 51000.0, 0.1, fee=5.0, strategy_id="s1", pnl=200.0)
        return b

    def test_add_trade(self, blotter):
        blotter.add_trade("BTC/USDT", "buy", 50000.0, 0.1)
        entries = blotter.get_entries()
        assert len(entries) == 1
        assert entries[0].symbol == "BTC/USDT"
        assert entries[0].side == "buy"
        assert entries[0].price == 50000.0
        assert entries[0].quantity == 0.1

    def test_add_entry(self, blotter):
        entry = BlotterEntry(symbol="ETH/USDT", side="sell", price=3000.0, quantity=1.0, value=3000.0)
        blotter.add(entry)
        assert len(blotter.get_entries()) == 1

    def test_value_computed(self, blotter):
        blotter.add_trade("BTC/USDT", "buy", 50000.0, 0.2)
        entry = blotter.get_entries()[0]
        assert entry.value == 50000.0 * 0.2

    def test_get_entries_filter_symbol(self, filled_blotter):
        btc = filled_blotter.get_entries(symbol="BTC/USDT")
        assert len(btc) == 2
        for e in btc:
            assert e.symbol == "BTC/USDT"

    def test_get_entries_filter_strategy(self, filled_blotter):
        s1 = filled_blotter.get_entries(strategy_id="s1")
        assert len(s1) == 2
        for e in s1:
            assert e.strategy_id == "s1"

    def test_get_entries_limit(self, filled_blotter):
        entries = filled_blotter.get_entries(limit=2)
        assert len(entries) == 2

    def test_get_summary(self, filled_blotter):
        summary = filled_blotter.get_summary()
        assert summary["total_trades"] == 3
        assert summary["total_pnl"] == 250.0
        assert summary["total_fees"] == 13.0
        assert summary["win_rate"] > 0

    def test_empty_summary(self, blotter):
        summary = blotter.get_summary()
        assert summary["total_trades"] == 0

    def test_clear(self, filled_blotter):
        filled_blotter.clear()
        assert len(filled_blotter.get_entries()) == 0

    def test_max_entries_eviction(self):
        blotter = TradeBlotter({"max_entries": 5})
        for i in range(10):
            blotter.add_trade("BTC/USDT", "buy", 50000.0 + i, 0.1)
        entries = blotter.get_entries()
        assert len(entries) == 5
        # Should keep the most recent entries
        assert entries[-1].price == 50009.0
