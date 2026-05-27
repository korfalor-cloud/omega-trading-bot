"""Tests for regime detection."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.regime.detector import RegimeDetector, RegimeState


class TestRegimeDetector:
    @pytest.fixture
    def detector(self):
        return RegimeDetector()

    @pytest.fixture
    def trending_returns(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal(200) * 0.01 + 0.001  # Slight positive drift

    @pytest.fixture
    def volatile_returns(self):
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(200) * 0.01
        returns[100:] *= 5  # High volatility regime
        return returns

    def test_detect_returns_regime_state(self, detector, trending_returns):
        state = detector.detect(trending_returns)
        assert isinstance(state, RegimeState)
        assert state.confidence >= 0
        assert state.volatility_regime in ("low", "medium", "high")
        assert state.trend_regime in ("trending_up", "trending_down", "ranging")

    def test_detect_short_series(self, detector):
        returns = np.array([0.01, -0.01, 0.005])
        state = detector.detect(returns)
        assert isinstance(state, RegimeState)

    def test_volatility_regime_change(self, detector, volatile_returns):
        # Build history with low vol data
        for i in range(20, 100, 10):
            detector.detect(volatile_returns[:i])
        # Now feed high vol data
        for i in range(110, len(volatile_returns), 10):
            state = detector.detect(volatile_returns[:i])
        # After enough high-vol data, should classify as high
        assert state.volatility_regime in ("medium", "high")

    def test_regime_duration_tracking(self, detector, trending_returns):
        state1 = detector.detect(trending_returns)
        state2 = detector.detect(trending_returns)
        # Same regime should increment duration
        if state1.regime_id == state2.regime_id:
            assert state2.duration >= state1.duration

    def test_transition_probabilities(self, detector, volatile_returns):
        # Run multiple detections to build history
        for i in range(20, len(volatile_returns), 10):
            detector.detect(volatile_returns[:i])
        state = detector.detect(volatile_returns)
        # Should have some transition probabilities
        assert isinstance(state.transition_probs, dict)

    def test_dominant_regime(self, detector, trending_returns):
        for i in range(50, len(trending_returns), 10):
            detector.detect(trending_returns[:i])
        dominant = detector.dominant_regime()
        assert dominant is not None
        assert isinstance(dominant, int)

    def test_dominant_regime_empty(self, detector):
        assert detector.dominant_regime() is None

    def test_regime_stats(self, detector, trending_returns):
        for i in range(50, len(trending_returns), 10):
            detector.detect(trending_returns[:i])
        state = detector.detect(trending_returns)
        stats = detector.get_regime_stats(trending_returns, state.regime_id)
        assert isinstance(stats, dict)
        if stats:  # May be empty if not enough data
            assert "mean_return" in stats
            assert "volatility" in stats
            assert "sharpe" in stats

    def test_custom_config(self):
        detector = RegimeDetector(config={
            "n_regimes": 2,
            "vol_lookback": 10,
            "trend_lookback": 30,
        })
        assert detector.vol_lookback == 10
        assert detector.trend_lookback == 30

    def test_regime_name(self, detector, trending_returns):
        state = detector.detect(trending_returns)
        assert state.regime_name != "unknown" or state.regime_id == 0
