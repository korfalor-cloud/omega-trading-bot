"""Tests for volume profile analysis."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.data.volume_profile import VolumeProfileAnalyzer, VolumeProfileResult


class TestVolumeProfileAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return VolumeProfileAnalyzer(config={"n_bins": 20, "value_area_pct": 0.70})

    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(42)
        n = 100
        prices = 50000 + rng.normal(0, 500, n)
        volumes = rng.uniform(100, 1000, n)
        highs = prices + rng.uniform(0, 200, n)
        lows = prices - rng.uniform(0, 200, n)
        return prices, volumes, highs, lows

    def test_analyze_basic(self, analyzer, sample_data):
        prices, volumes, highs, lows = sample_data
        result = analyzer.analyze(prices, volumes)
        assert isinstance(result, VolumeProfileResult)
        assert result.poc > 0
        assert result.total_volume > 0

    def test_analyze_with_hl(self, analyzer, sample_data):
        prices, volumes, highs, lows = sample_data
        result = analyzer.analyze(prices, volumes, highs, lows)
        assert result.poc > 0
        assert result.vah > result.val

    def test_poc_near_center(self, analyzer, sample_data):
        prices, volumes, _, _ = sample_data
        result = analyzer.analyze(prices, volumes)
        mean_price = np.mean(prices)
        assert abs(result.poc - mean_price) < 2000

    def test_value_area(self, analyzer, sample_data):
        prices, volumes, highs, lows = sample_data
        result = analyzer.analyze(prices, volumes, highs, lows)
        assert result.val <= result.poc <= result.vah

    def test_empty_data(self, analyzer):
        result = analyzer.analyze(np.array([]), np.array([]))
        assert result.poc == 0

    def test_single_price(self, analyzer):
        result = analyzer.analyze(np.array([50000]), np.array([100]))
        assert result.poc == 50000

    def test_price_levels(self, analyzer, sample_data):
        prices, volumes, _, _ = sample_data
        result = analyzer.analyze(prices, volumes)
        assert len(result.price_levels) > 0
        assert len(result.volume_at_price) > 0

    def test_support_resistance(self, analyzer, sample_data):
        prices, volumes, _, _ = sample_data
        levels = analyzer.find_support_resistance(prices, volumes, n_levels=3)
        assert len(levels) <= 3
        for lvl in levels:
            assert lvl > 0

    def test_support_resistance_empty(self, analyzer):
        levels = analyzer.find_support_resistance(np.array([]), np.array([]))
        assert levels == []
