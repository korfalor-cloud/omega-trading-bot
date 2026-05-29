"""Tests for data quality checker."""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta

import numpy as np

from tradingbot.data.quality import DataQualityChecker, QualityReport


class TestDataQualityChecker:
    @pytest.fixture
    def checker(self):
        return DataQualityChecker()

    @pytest.fixture
    def valid_bars(self):
        return [
            {"open": 100, "high": 105, "low": 99, "close": 103, "volume": 1000},
            {"open": 103, "high": 108, "low": 102, "close": 107, "volume": 1200},
            {"open": 107, "high": 110, "low": 105, "close": 109, "volume": 800},
        ]

    def test_valid_bars(self, checker, valid_bars):
        report = checker.validate_bars(valid_bars)
        assert report.total_bars == 3
        assert report.valid_bars == 3
        assert report.completeness == 1.0

    def test_invalid_ohlc(self, checker):
        bars = [{"open": 100, "high": 90, "low": 105, "close": 103, "volume": 1000}]
        report = checker.validate_bars(bars)
        assert report.anomalies > 0

    def test_empty_data(self, checker):
        report = checker.validate_bars([])
        assert report.total_bars == 0
        assert "Empty data" in report.issues

    def test_missing_fields(self, checker):
        bars = [{"open": 100, "high": 105}]
        report = checker.validate_bars(bars)
        assert report.valid_bars == 0

    def test_detect_spikes(self, checker):
        prices = np.array([100, 101, 102, 103, 200, 105, 106, 107, 108, 109])
        spikes = checker.detect_spikes(prices, threshold=2.0)
        assert 4 in spikes

    def test_detect_no_spikes(self, checker):
        prices = np.arange(100, 110, dtype=float)
        spikes = checker.detect_spikes(prices, threshold=3.0)
        assert len(spikes) == 0

    def test_detect_stale(self, checker):
        prices = np.array([100] * 20)
        stale = checker.detect_stale(prices, min_changes=3, window=10)
        assert len(stale) > 0

    def test_detect_not_stale(self, checker):
        prices = np.arange(100, 120, dtype=float)
        stale = checker.detect_stale(prices, min_changes=3, window=10)
        assert len(stale) == 0

    def test_price_spike_detection_in_bars(self, checker):
        bars = [
            {"open": 100, "high": 105, "low": 99, "close": 103, "volume": 1000},
            {"open": 103, "high": 200, "low": 102, "close": 195, "volume": 1200},
        ]
        report = checker.validate_bars(bars)
        assert report.anomalies > 0

    def test_quality_report_fields(self, checker, valid_bars):
        report = checker.validate_bars(valid_bars)
        assert isinstance(report, QualityReport)
        assert hasattr(report, "completeness")
        assert hasattr(report, "gaps")
        assert hasattr(report, "anomalies")
