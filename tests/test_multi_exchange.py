"""Tests for multi-exchange router."""
from __future__ import annotations

import pytest

from tradingbot.exchanges.multi_exchange import (
    MultiExchangeRouter,
    RoutingDecision,
    VenueQuote,
)


class TestMultiExchangeRouter:
    @pytest.fixture
    def router(self):
        return MultiExchangeRouter(config={"max_latency_ms": 500, "min_liquidity": 0.01})

    @pytest.fixture
    def quotes(self):
        return [
            VenueQuote(exchange="binance", bid=50000, ask=50010, bid_size=1.5, ask_size=1.5, fee_rate=0.001, latency_ms=50),
            VenueQuote(exchange="coinbase", bid=49995, ask=50005, bid_size=0.8, ask_size=0.8, fee_rate=0.0015, latency_ms=100),
            VenueQuote(exchange="kraken", bid=50002, ask=50012, bid_size=2.0, ask_size=2.0, fee_rate=0.002, latency_ms=200),
        ]

    def test_update_quote(self, router, quotes):
        for q in quotes:
            router.update_quote(q)
        assert len(router.get_venues()) == 3

    def test_best_ask(self, router, quotes):
        for q in quotes:
            router.update_quote(q)
        best = router.best_ask()
        # Binance ask=50010 + 0.001*50010 = 50060; Coinbase ask=50005 + 0.0015*50005 = 50080
        assert best.exchange == "binance"  # Best fee-adjusted ask

    def test_best_bid(self, router, quotes):
        for q in quotes:
            router.update_quote(q)
        best = router.best_bid()
        assert best is not None

    def test_route_buy(self, router, quotes):
        for q in quotes:
            router.update_quote(q)
        decisions = router.route_buy(0.5)
        assert len(decisions) > 0
        assert decisions[0].exchange == "binance"  # Best fee-adjusted ask

    def test_route_sell(self, router, quotes):
        for q in quotes:
            router.update_quote(q)
        decisions = router.route_sell(0.5)
        assert len(decisions) > 0

    def test_route_buy_split(self, router, quotes):
        for q in quotes:
            router.update_quote(q)
        # Route 2 units — exceeds single venue liquidity
        decisions = router.route_buy(2.0)
        assert len(decisions) > 1

    def test_empty_router(self, router):
        assert router.best_ask() is None
        assert router.best_bid() is None
        assert router.route_buy(1.0) == []

    def test_latency_filter(self, router):
        router.update_quote(VenueQuote(exchange="slow", bid=50000, ask=50005, bid_size=10, ask_size=10, latency_ms=1000))
        router.update_quote(VenueQuote(exchange="fast", bid=49990, ask=50010, bid_size=10, ask_size=10, latency_ms=50))
        decisions = router.route_buy(1.0)
        # slow venue filtered out
        assert decisions[0].exchange == "fast"

    def test_spread_comparison(self, router, quotes):
        for q in quotes:
            router.update_quote(q)
        spreads = router.get_spread_comparison()
        assert "binance" in spreads
        assert spreads["binance"] == 10  # 50010 - 50000
