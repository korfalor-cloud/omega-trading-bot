"""Tests for market microstructure analysis."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

import numpy as np

from tradingbot.execution.microstructure.analysis import (
    MicrostructureAnalyzer,
    MicrostructureMetrics,
)
from tradingbot.core.types import OrderBookLevel, OrderBookSnapshot, Tick
from tradingbot.core.enums import Side


class TestMicrostructureAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return MicrostructureAnalyzer()

    @pytest.fixture
    def ticks(self):
        from datetime import timedelta
        rng = np.random.default_rng(42)
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ticks = []
        price = 50000.0
        for i in range(100):
            price += rng.normal(0, 50)
            ticks.append(Tick(
                timestamp=base_time + timedelta(seconds=i),
                symbol="BTC/USDT",
                price=price,
                quantity=rng.uniform(0.01, 1.0),
                side=Side.BUY if rng.random() > 0.5 else Side.SELL,
                exchange="test",
            ))
        return ticks

    @pytest.fixture
    def order_book(self):
        return OrderBookSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            exchange="test",
            bids=[
                OrderBookLevel(price=49900, quantity=1.5),
                OrderBookLevel(price=49800, quantity=2.0),
                OrderBookLevel(price=49700, quantity=3.0),
            ],
            asks=[
                OrderBookLevel(price=50100, quantity=1.0),
                OrderBookLevel(price=50200, quantity=2.5),
                OrderBookLevel(price=50300, quantity=1.5),
            ],
        )

    def test_vpin(self, analyzer):
        volumes = np.random.uniform(0.1, 10.0, 100)
        price_changes = np.random.normal(0, 0.01, 100)
        vpin = analyzer.compute_vpin(volumes, price_changes)
        assert 0 <= vpin <= 1

    def test_kyle_lambda(self, analyzer):
        price_changes = np.random.normal(0, 0.01, 100)
        signed_volumes = np.random.normal(0, 5.0, 100)
        lam = analyzer.compute_kyle_lambda(price_changes, signed_volumes)
        assert isinstance(lam, float)

    def test_classify_trade(self, analyzer):
        side, signed_vol = analyzer.classify_trade(50050, 50000, 50100, 1.0)
        assert side == "buy"
        assert signed_vol > 0

        side, signed_vol = analyzer.classify_trade(49950, 50000, 50100, 1.0)
        assert side == "sell"
        assert signed_vol < 0

    def test_effective_spread(self, analyzer):
        spread = analyzer.compute_effective_spread(50050, 50000, 1.0)
        assert spread > 0
        # Should be 2 * 50 / 50000 * 10000 = 20 bps
        assert abs(spread - 20.0) < 1.0

    def test_order_flow_imbalance(self, analyzer):
        bids = [(49900, 2.0), (49800, 3.0)]
        asks = [(50100, 1.0), (50200, 1.0)]
        imbalance = analyzer.compute_order_flow_imbalance(bids, asks)
        assert imbalance > 0  # More bid volume
        assert -1 <= imbalance <= 1

    def test_order_flow_balanced(self, analyzer):
        bids = [(49900, 1.0)]
        asks = [(50100, 1.0)]
        imbalance = analyzer.compute_order_flow_imbalance(bids, asks)
        assert abs(imbalance) < 0.01

    def test_trade_intensity(self, analyzer):
        timestamps = [float(i) for i in range(100)]
        intensity = analyzer.compute_trade_intensity(timestamps, window_seconds=60)
        assert intensity > 0

    def test_market_impact(self, analyzer):
        impact = analyzer.estimate_market_impact(
            quantity=10.0, adv=1000.0, volatility=0.03, spread_bps=5.0
        )
        assert impact > 0

    def test_market_impact_zero_adv(self, analyzer):
        impact = analyzer.estimate_market_impact(10.0, 0, 0.03)
        assert impact == 0

    def test_analyze_tick_data(self, analyzer, ticks):
        metrics = analyzer.analyze_tick_data(ticks)
        assert isinstance(metrics, MicrostructureMetrics)
        assert metrics.trade_intensity > 0

    def test_analyze_with_order_book(self, analyzer, ticks, order_book):
        metrics = analyzer.analyze_tick_data(ticks, order_book)
        assert isinstance(metrics, MicrostructureMetrics)
        assert metrics.effective_spread_bps > 0

    def test_insufficient_ticks(self, analyzer):
        ticks = [Tick(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT", price=50000, quantity=1.0,
            side=Side.BUY, exchange="test",
        )]
        metrics = analyzer.analyze_tick_data(ticks)
        assert metrics.vpin == 0.0
