"""Tests for multi-strategy orchestrator."""
from __future__ import annotations

import pytest

from tradingbot.strategies.multi_strategy.orchestrator import (
    OrchestratorSignal,
    StrategyAllocation,
    StrategyOrchestrator,
)


class TestStrategyOrchestrator:
    @pytest.fixture
    def orch(self):
        return StrategyOrchestrator(config={"total_capital": 100000})

    def test_register_strategy(self, orch):
        alloc = orch.register_strategy("trend", 0.3)
        assert alloc.strategy_id == "trend"
        assert alloc.weight == 0.3
        assert alloc.allocated_capital == 30000

    def test_register_max_strategies(self, orch):
        orch.max_strategies = 3
        orch.register_strategy("a", 0.33)
        orch.register_strategy("b", 0.33)
        orch.register_strategy("c", 0.34)
        with pytest.raises(ValueError):
            orch.register_strategy("d", 0.1)

    def test_remove_strategy(self, orch):
        orch.register_strategy("trend", 0.5)
        orch.register_strategy("mean_rev", 0.5)
        result = orch.remove_strategy("trend")
        assert result is True
        assert len(orch.get_all_allocations()) == 1

    def test_remove_nonexistent(self, orch):
        assert orch.remove_strategy("nonexistent") is False

    def test_update_performance(self, orch):
        orch.register_strategy("trend", 0.5)
        orch.update_performance("trend", pnl=1000, sharpe=1.5, win_rate=0.6)
        alloc = orch.get_allocation("trend")
        assert alloc.current_pnl == 1000
        assert alloc.sharpe == 1.5

    def test_compute_signal_same_direction(self, orch):
        orch.register_strategy("trend", 0.5)
        orch.register_strategy("mean_rev", 0.5)

        signals = [
            OrchestratorSignal(strategy_id="trend", symbol="BTC/USDT", side="buy", strength=0.8, confidence=0.7),
            OrchestratorSignal(strategy_id="mean_rev", symbol="BTC/USDT", side="buy", strength=0.6, confidence=0.8),
        ]
        result = orch.compute_signal(signals)
        assert "BTC/USDT" in result
        assert result["BTC/USDT"].side == "buy"

    def test_compute_signal_opposing(self, orch):
        orch.register_strategy("trend", 0.5)
        orch.register_strategy("mean_rev", 0.5)

        signals = [
            OrchestratorSignal(strategy_id="trend", symbol="BTC/USDT", side="buy", strength=0.8, confidence=0.7),
            OrchestratorSignal(strategy_id="mean_rev", symbol="BTC/USDT", side="sell", strength=0.8, confidence=0.7),
        ]
        result = orch.compute_signal(signals)
        # Equal opposing signals should cancel
        assert len(result) == 0

    def test_compute_signal_partial_cancel(self, orch):
        orch.register_strategy("trend", 0.5)
        orch.register_strategy("mean_rev", 0.5)

        signals = [
            OrchestratorSignal(strategy_id="trend", symbol="BTC/USDT", side="buy", strength=1.0, confidence=0.7),
            OrchestratorSignal(strategy_id="mean_rev", symbol="BTC/USDT", side="sell", strength=0.3, confidence=0.7),
        ]
        result = orch.compute_signal(signals)
        assert "BTC/USDT" in result
        assert result["BTC/USDT"].side == "buy"

    def test_rebalance_weights(self, orch):
        orch.register_strategy("trend", 0.5)
        orch.register_strategy("mean_rev", 0.5)
        orch.update_performance("trend", pnl=1000, sharpe=2.0)
        orch.update_performance("mean_rev", pnl=500, sharpe=0.5)

        weights = orch.rebalance_weights()
        assert weights["trend"] > weights["mean_rev"]

    def test_get_portfolio_summary(self, orch):
        orch.register_strategy("trend", 0.5)
        orch.register_strategy("mean_rev", 0.5)
        orch.update_performance("trend", pnl=1000)
        orch.update_performance("mean_rev", pnl=500)

        summary = orch.get_portfolio_summary()
        assert summary["n_strategies"] == 2
        assert summary["total_pnl"] == 1500

    def test_weight_normalization(self, orch):
        orch.register_strategy("a", 0.3)
        orch.register_strategy("b", 0.3)
        orch.register_strategy("c", 0.3)
        orch._normalize_weights()
        total = sum(a.weight for a in orch.get_all_allocations())
        assert abs(total - 1.0) < 0.01

    def test_min_max_weight_clamp(self, orch):
        orch.min_weight = 0.1
        orch.max_weight = 0.5
        orch.register_strategy("a", 0.01)  # Below min
        orch.register_strategy("b", 0.99)  # Above max
        allocs = orch.get_all_allocations()
        assert all(a.weight >= 0.1 for a in allocs)
        assert all(a.weight <= 0.5 for a in allocs)
