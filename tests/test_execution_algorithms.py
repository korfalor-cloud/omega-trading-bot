"""Tests for execution algorithms: TWAP, VWAP, Implementation Shortfall."""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from tradingbot.execution.algorithms.twap import TWAPAlgorithm
from tradingbot.execution.algorithms.vwap import VWAPAlgorithm
from tradingbot.execution.algorithms.implementation_shortfall import ImplementationShortfallAlgorithm
from tradingbot.core.enums import Side


class TestTWAP:
    @pytest.fixture
    def algo(self):
        return TWAPAlgorithm({"num_slices": 5, "duration_minutes": 10})

    def test_create_twap(self, algo):
        state = algo.create_twap_order("BTC/USDT", Side.BUY, 1.0)
        assert state.total_quantity == 1.0
        assert len(state.slices) == 5
        assert not state.completed

    def test_slice_quantities_sum(self, algo):
        state = algo.create_twap_order("BTC/USDT", Side.BUY, 1.0)
        total = sum(s.order.quantity for s in state.slices)
        assert abs(total - 1.0) < 0.01

    def test_get_pending_slices(self, algo):
        start = datetime.utcnow() - timedelta(minutes=5)
        state = algo.create_twap_order("BTC/USDT", Side.BUY, 1.0, start_time=start)
        pending = algo.get_pending_slices(state.parent_id)
        # Should have some pending slices since we started 5 min ago
        assert len(pending) > 0

    def test_record_fill(self, algo):
        state = algo.create_twap_order("BTC/USDT", Side.BUY, 1.0)
        algo.record_fill(state.parent_id, 0, 50000.0, 0.2)
        assert state.filled_quantity == 0.2
        assert state.vwap_executed == 50000.0

    def test_completion(self, algo):
        state = algo.create_twap_order("BTC/USDT", Side.BUY, 1.0)
        for i, s in enumerate(state.slices):
            algo.record_fill(state.parent_id, i, 50000.0, s.order.quantity)
        assert state.completed

    def test_slippage_bps(self, algo):
        state = algo.create_twap_order("BTC/USDT", Side.BUY, 1.0)
        algo.record_fill(state.parent_id, 0, 50100.0, 1.0)
        slippage = algo.get_slippage_bps(state.parent_id, 50000.0)
        assert slippage > 0  # Bought at higher price

    def test_cancel(self, algo):
        state = algo.create_twap_order("BTC/USDT", Side.BUY, 1.0)
        algo.cancel(state.parent_id)
        assert state.completed

    def test_get_state(self, algo):
        state = algo.create_twap_order("BTC/USDT", Side.BUY, 1.0)
        assert algo.get_state(state.parent_id) is state
        assert algo.get_state("nonexistent") is None


class TestVWAP:
    @pytest.fixture
    def algo(self):
        return VWAPAlgorithm({"num_buckets": 6})

    def test_create_vwap(self, algo):
        state = algo.create_vwap_order("BTC/USDT", Side.BUY, 1.0)
        assert state.total_quantity == 1.0
        assert len(state.buckets) == 6

    def test_bucket_quantities_sum(self, algo):
        state = algo.create_vwap_order("BTC/USDT", Side.BUY, 1.0)
        total = sum(b.target_quantity for b in state.buckets)
        assert abs(total - 1.0) < 0.01

    def test_volume_profile(self):
        algo = VWAPAlgorithm()
        profile = algo._default_volume_profile(10)
        assert len(profile) == 10
        assert abs(sum(profile) - 1.0) < 0.001

    def test_set_volume_profile(self, algo):
        import numpy as np
        algo.set_volume_profile(np.array([1, 2, 3, 4, 5, 6]))
        state = algo.create_vwap_order("BTC/USDT", Side.BUY, 1.0)
        # Later buckets should have more volume
        assert state.buckets[-1].volume_pct > state.buckets[0].volume_pct

    def test_record_fill(self, algo):
        state = algo.create_vwap_order("BTC/USDT", Side.BUY, 1.0)
        algo.record_fill(state.parent_id, 0, 50000.0, state.buckets[0].target_quantity)
        assert state.buckets[0].completed

    def test_slippage(self, algo):
        state = algo.create_vwap_order("BTC/USDT", Side.SELL, 1.0)
        algo.record_fill(state.parent_id, 0, 50100.0, 1.0)
        slippage = algo.get_slippage_bps(state.parent_id, 50000.0)
        # Sold at 50100 vs market VWAP 50000 — negative slippage (favorable)
        assert slippage < 0

    def test_cancel(self, algo):
        state = algo.create_vwap_order("BTC/USDT", Side.BUY, 1.0)
        algo.cancel(state.parent_id)
        assert state.completed


class TestImplementationShortfall:
    @pytest.fixture
    def algo(self):
        return ImplementationShortfallAlgorithm({"num_slices": 5, "duration_minutes": 30})

    def test_create_is(self, algo):
        state = algo.create_is_order("BTC/USDT", Side.BUY, 1.0, 50000.0)
        assert state.total_quantity == 1.0
        assert state.decision_price == 50000.0
        assert len(state.trajectory) == 6  # n+1 trajectory points

    def test_trajectory_quantity_sum(self, algo):
        state = algo.create_is_order("BTC/USDT", Side.BUY, 1.0, 50000.0)
        total = sum(t.trade_quantity for t in state.trajectory)
        assert abs(total - 1.0) < 0.01

    def test_optimal_trajectory_decays(self, algo):
        state = algo.create_is_order("BTC/USDT", Side.BUY, 1.0, 50000.0)
        # Remaining quantity should decrease over time
        remainings = [t.remaining_quantity for t in state.trajectory]
        for i in range(1, len(remainings)):
            assert remainings[i] <= remainings[i - 1] + 0.001

    def test_record_fill(self, algo):
        state = algo.create_is_order("BTC/USDT", Side.BUY, 1.0, 50000.0)
        algo.record_fill(state.parent_id, 0, 50100.0, 0.5)
        assert state.filled_quantity == 0.5
        assert state.total_cost == 50100.0 * 0.5

    def test_implementation_shortfall_calculation(self, algo):
        state = algo.create_is_order("BTC/USDT", Side.BUY, 1.0, 50000.0)
        algo.record_fill(state.parent_id, 0, 50100.0, 0.5)
        algo.record_fill(state.parent_id, 1, 50200.0, 0.5)
        # IS should be positive (bought at higher price than decision)
        assert state.implementation_shortfall > 0

    def test_completion(self, algo):
        state = algo.create_is_order("BTC/USDT", Side.BUY, 1.0, 50000.0)
        for i, t in enumerate(state.trajectory):
            if t.trade_quantity > 0:
                algo.record_fill(state.parent_id, i, 50000.0, t.trade_quantity)
        assert state.completed

    def test_cancel(self, algo):
        state = algo.create_is_order("BTC/USDT", Side.BUY, 1.0, 50000.0)
        algo.cancel(state.parent_id)
        assert state.completed

    def test_risk_aversion_affects_trajectory(self):
        # Low risk aversion = more backloaded
        algo_low = ImplementationShortfallAlgorithm({"risk_aversion": 1e-8, "num_slices": 5})
        state_low = algo_low.create_is_order("BTC/USDT", Side.BUY, 1.0, 50000.0)

        # High risk aversion = more frontloaded
        algo_high = ImplementationShortfallAlgorithm({"risk_aversion": 1e-4, "num_slices": 5})
        state_high = algo_high.create_is_order("BTC/USDT", Side.BUY, 1.0, 50000.0)

        # First slice should be larger for high risk aversion
        first_low = state_low.trajectory[0].trade_quantity
        first_high = state_high.trajectory[0].trade_quantity
        assert first_high >= first_low
