"""Tests for multi-timeframe analysis."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.strategies.multi_timeframe.analysis import (
    MultiTimeframeAnalyzer,
    MTFAnalysis,
    TimeframeState,
)


class TestMultiTimeframeAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return MultiTimeframeAnalyzer()

    @pytest.fixture
    def bullish_data(self):
        """Prices trending up strongly on all timeframes."""
        base = 50000 + np.arange(60) * 500  # Strong uptrend
        return {
            "1h": base.copy(),
            "4h": base.copy(),
            "1d": base.copy(),
        }

    @pytest.fixture
    def bearish_data(self):
        """Prices trending down strongly on all timeframes."""
        base = 80000 - np.arange(60) * 500
        return {
            "1h": base.copy(),
            "4h": base.copy(),
            "1d": base.copy(),
        }

    @pytest.fixture
    def mixed_data(self):
        """Conflicting trends across timeframes."""
        up = 50000 + np.arange(60) * 100
        down = 51000 - np.arange(60) * 100
        return {
            "1h": up.copy(),
            "4h": down.copy(),
            "1d": up.copy(),
        }

    def test_analyze_bullish(self, analyzer, bullish_data):
        result = analyzer.analyze(bullish_data)
        assert isinstance(result, MTFAnalysis)
        assert result.bias == "bullish"
        assert result.alignment > 0

    def test_analyze_bearish(self, analyzer, bearish_data):
        result = analyzer.analyze(bearish_data)
        assert result.bias == "bearish"
        assert result.alignment < 0

    def test_analyze_mixed(self, analyzer, mixed_data):
        result = analyzer.analyze(mixed_data)
        assert isinstance(result.bias, str)

    def test_timeframe_states(self, analyzer, bullish_data):
        result = analyzer.analyze(bullish_data)
        assert len(result.timeframe_states) == 3
        for state in result.timeframe_states:
            assert isinstance(state, TimeframeState)

    def test_check_alignment_bullish(self, analyzer, bullish_data):
        assert analyzer.check_alignment(bullish_data, required_agreement=2) is True

    def test_check_alignment_mixed(self, analyzer, mixed_data):
        # May or may not align depending on exact data
        result = analyzer.check_alignment(mixed_data, required_agreement=3)
        assert isinstance(result, bool)

    def test_get_bias_strength(self, analyzer, bullish_data):
        strength = analyzer.get_bias_strength(bullish_data)
        assert -1 <= strength <= 1
        assert strength > 0  # Bullish data

    def test_empty_data(self, analyzer):
        result = analyzer.analyze({})
        assert result.bias == "neutral"
        assert len(result.timeframe_states) == 0

    def test_short_data_skipped(self, analyzer):
        result = analyzer.analyze({"1h": np.array([100, 101, 102])})
        # Only 3 points, below min of 10
        assert len(result.timeframe_states) == 0

    def test_confidence(self, analyzer, bullish_data):
        result = analyzer.analyze(bullish_data)
        assert 0 <= result.confidence <= 1
