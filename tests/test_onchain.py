"""Tests for on-chain analytics."""
from __future__ import annotations

import pytest
import numpy as np
from datetime import datetime, timezone

from tradingbot.alternative.onchain import (
    OnChainAnalyzer,
    OnChainSignal,
    WhaleAlert,
)


class TestOnChainAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return OnChainAnalyzer(config={"whale_threshold_usd": 100_000})

    def test_exchange_flows_bullish(self, analyzer):
        # Net outflow = bullish
        inflows = np.array([100, 200, 150])
        outflows = np.array([500, 600, 400])
        result = analyzer.analyze_exchange_flows(inflows, outflows)
        assert isinstance(result, OnChainSignal)
        assert result.metric == "exchange_flow"
        assert result.value > 0  # Net outflow

    def test_exchange_flows_bearish(self, analyzer):
        # Net inflow = bearish
        inflows = np.array([500, 600, 400])
        outflows = np.array([100, 200, 150])
        result = analyzer.analyze_exchange_flows(inflows, outflows)
        assert result.value < 0  # Net inflow

    def test_whale_detection(self, analyzer):
        transactions = [
            {"amount_usd": 50_000, "from": "wallet_a", "to": "wallet_b"},
            {"amount_usd": 500_000, "from": "whale_wallet", "to": "binance_exchange"},
            {"amount_usd": 1_000_000, "from": "coinbase_exchange", "to": "whale_wallet"},
        ]
        alerts = analyzer.detect_whale_activity(transactions)
        assert len(alerts) == 2  # Only whale txns
        assert alerts[0].direction == "to_exchange"
        assert alerts[0].signal == "bearish"
        assert alerts[1].direction == "from_exchange"
        assert alerts[1].signal == "bullish"

    def test_whale_below_threshold(self, analyzer):
        transactions = [
            {"amount_usd": 1000, "from": "a", "to": "b"},
        ]
        alerts = analyzer.detect_whale_activity(transactions)
        assert len(alerts) == 0

    def test_nvt_ratio(self, analyzer):
        result = analyzer.nvt_ratio(market_cap=1e12, daily_transaction_volume=50e9)
        assert isinstance(result, OnChainSignal)
        assert result.metric == "nvt_ratio"
        assert result.value == pytest.approx(20.0)

    def test_nvt_high_overvalued(self, analyzer):
        # Build history
        for _ in range(20):
            analyzer.nvt_ratio(1e12, 50e9)
        # High NVT = overvalued = bearish
        result = analyzer.nvt_ratio(1e12, 10e9)
        assert result.value > 50

    def test_active_address_signal(self, analyzer):
        addresses = np.arange(1000, 1200, 10).astype(float)
        result = analyzer.active_address_signal(addresses)
        assert isinstance(result, OnChainSignal)
        assert result.metric == "active_addresses"
        # Growing addresses = bullish
        assert result.signal in ("bullish", "neutral")

    def test_active_address_decline(self, analyzer):
        addresses = np.arange(1200, 1000, -10).astype(float)
        result = analyzer.active_address_signal(addresses)
        assert result.signal in ("bearish", "neutral")

    def test_hash_rate_signal(self, analyzer):
        hash_rate = np.linspace(100, 150, 60)
        result = analyzer.hash_rate_signal(hash_rate)
        assert result.metric == "hash_rate"
        assert result.signal in ("bullish", "neutral")

    def test_hash_rate_decline(self, analyzer):
        hash_rate = np.linspace(150, 100, 60)
        result = analyzer.hash_rate_signal(hash_rate)
        assert result.signal in ("bearish", "neutral")

    def test_aggregate_signals(self, analyzer):
        signals = [
            OnChainSignal(metric="flow", signal="bullish", strength=0.8),
            OnChainSignal(metric="nvt", signal="bearish", strength=0.3),
            OnChainSignal(metric="addresses", signal="bullish", strength=0.5),
        ]
        result = analyzer.aggregate_signals(signals)
        assert result["n_signals"] == 3
        assert result["direction"] in ("bullish", "bearish", "neutral")
        assert isinstance(result["composite_score"], float)

    def test_aggregate_empty(self, analyzer):
        result = analyzer.aggregate_signals([])
        assert result["direction"] == "neutral"

    def test_active_address_short_data(self, analyzer):
        result = analyzer.active_address_signal(np.array([100.0]))
        assert result.metric == "active_addresses"

    def test_hash_rate_short_data(self, analyzer):
        result = analyzer.hash_rate_signal(np.array([100.0, 110.0]))
        assert result.metric == "hash_rate"
