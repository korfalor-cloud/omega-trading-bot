"""Tests for data normalizer."""
from __future__ import annotations

import pytest
import numpy as np
from datetime import datetime, timezone

from tradingbot.data.normalizer import DataNormalizer, NormalizedBar


class TestDataNormalizer:
    @pytest.fixture
    def normalizer(self):
        return DataNormalizer()

    def test_normalize_bars(self, normalizer):
        raw = [
            {"timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc), "open": 100, "high": 105, "low": 99, "close": 103, "volume": 1000},
            {"timestamp": datetime(2024, 1, 2, tzinfo=timezone.utc), "open": 103, "high": 108, "low": 102, "close": 107, "volume": 1200},
        ]
        result = normalizer.normalize_bars(raw, source="test")
        assert len(result) == 2
        assert all(isinstance(b, NormalizedBar) for b in result)
        assert result[0].source == "test"

    def test_normalize_timestamp_float(self, normalizer):
        raw = [{"timestamp": 1704067200.0, "open": 100, "high": 105, "low": 99, "close": 103, "volume": 1000}]
        result = normalizer.normalize_bars(raw)
        assert result[0].timestamp == 1704067200.0

    def test_align_timestamps(self, normalizer):
        bars_a = [
            NormalizedBar(timestamp=1.0, open=100, high=105, low=99, close=103, volume=100),
            NormalizedBar(timestamp=2.0, open=103, high=108, low=102, close=107, volume=100),
        ]
        bars_b = [
            NormalizedBar(timestamp=1.1, open=200, high=205, low=199, close=203, volume=100),
            NormalizedBar(timestamp=2.1, open=203, high=208, low=202, close=207, volume=100),
        ]
        a, b = normalizer.align_timestamps(bars_a, bars_b, tolerance=0.5)
        assert len(a) == 2
        assert len(b) == 2

    def test_resample(self, normalizer):
        bars = [
            NormalizedBar(timestamp=0, open=100, high=105, low=99, close=103, volume=100),
            NormalizedBar(timestamp=30, open=103, high=108, low=102, close=107, volume=100),
            NormalizedBar(timestamp=60, open=107, high=110, low=105, close=109, volume=100),
            NormalizedBar(timestamp=90, open=109, high=112, low=108, close=111, volume=100),
        ]
        result = normalizer.resample(bars, target_period=60)
        assert len(result) == 2
        assert result[0].high == 108
        assert result[0].volume == 200

    def test_normalize_volume_zscore(self, normalizer):
        volumes = np.array([100, 200, 300, 400, 500], dtype=float)
        result = normalizer.normalize_volume(volumes, "zscore")
        assert abs(np.mean(result)) < 0.01

    def test_normalize_volume_minmax(self, normalizer):
        volumes = np.array([100, 200, 300, 400, 500], dtype=float)
        result = normalizer.normalize_volume(volumes, "minmax")
        assert result[0] == 0
        assert result[-1] == 1

    def test_normalize_volume_log(self, normalizer):
        volumes = np.array([100, 200, 300, 400, 500], dtype=float)
        result = normalizer.normalize_volume(volumes, "log")
        assert all(r > 0 for r in result)

    def test_detect_outliers(self, normalizer):
        prices = np.array([100, 101, 102, 103, 200, 105, 106, 107, 108, 109], dtype=float)
        outliers = normalizer.detect_outliers(prices, threshold=2.0)
        assert 4 in outliers

    def test_detect_no_outliers(self, normalizer):
        prices = np.arange(100, 110, dtype=float)
        outliers = normalizer.detect_outliers(prices, threshold=3.0)
        assert len(outliers) == 0

    def test_empty_bars(self, normalizer):
        result = normalizer.normalize_bars([])
        assert result == []
