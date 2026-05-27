"""Tests for sentiment analysis."""
from __future__ import annotations

import pytest
import numpy as np
from datetime import datetime, timezone

from tradingbot.alternative.sentiment import (
    SentimentAnalyzer,
    SentimentResult,
    SentimentSignal,
)


class TestSentimentAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return SentimentAnalyzer()

    def test_score_text_bullish(self, analyzer):
        result = analyzer.score_text("BTC is bullish and about to moon pump rally")
        assert result.score > 0
        assert result.magnitude > 0

    def test_score_text_bearish(self, analyzer):
        result = analyzer.score_text("Market crash dump panic bearish sell")
        assert result.score < 0

    def test_score_text_neutral(self, analyzer):
        result = analyzer.score_text("The weather is nice today")
        assert result.score == 0

    def test_score_batch(self, analyzer):
        texts = [
            "Bullish moon pump",
            "Bearish crash dump",
            "Buy accumulate hodl",
        ]
        result = analyzer.score_batch(texts, source="test")
        assert isinstance(result, SentimentResult)
        assert result.source == "test"

    def test_aggregate_sentiment(self, analyzer):
        results = [
            SentimentResult(score=0.5),
            SentimentResult(score=-0.3),
            SentimentResult(score=0.8),
        ]
        agg = analyzer.aggregate_sentiment(results)
        assert agg == pytest.approx(1.0 / 3, abs=0.01)

    def test_aggregate_with_weights(self, analyzer):
        results = [
            SentimentResult(score=0.8),
            SentimentResult(score=-0.2),
        ]
        weights = [0.8, 0.2]
        agg = analyzer.aggregate_sentiment(results, weights)
        assert agg > 0.5

    def test_sentiment_momentum(self, analyzer):
        for _ in range(5):
            analyzer.score_text("bullish moon")
        for _ in range(5):
            analyzer.score_text("bearish crash")
        momentum = analyzer.sentiment_momentum(window=5)
        assert isinstance(momentum, float)

    def test_detect_divergence(self, analyzer):
        price_returns = np.array([0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02])
        sentiment_scores = np.array([-0.3, -0.2, -0.4, -0.1, -0.3, -0.2, -0.4, -0.1, -0.3, -0.2])
        for s in sentiment_scores:
            if s > 0:
                analyzer.score_text("bullish moon")
            else:
                analyzer.score_text("bearish crash")
        div = analyzer.detect_divergence(price_returns, sentiment_scores, window=5)
        assert div == True

    def test_fear_greed_index(self, analyzer):
        fg = analyzer.fear_greed_index(volatility=0.02, momentum=0.01, sentiment=0.5)
        assert 0 <= fg <= 100

    def test_fear_greed_extreme_fear(self, analyzer):
        fg = analyzer.fear_greed_index(volatility=0.10, momentum=-0.05, sentiment=-0.8)
        assert fg < 40

    def test_fear_greed_extreme_greed(self, analyzer):
        fg = analyzer.fear_greed_index(volatility=0.005, momentum=0.05, sentiment=0.8)
        assert fg > 60

    def test_generate_signal_bullish(self, analyzer):
        # Need 10+ entries for momentum to be non-zero
        for _ in range(15):
            analyzer.score_text("bullish moon rally")
        signal = analyzer.generate_signal(0.5)
        assert signal.direction == "bullish"
        assert signal.strength > 0

    def test_generate_signal_bearish(self, analyzer):
        for _ in range(15):
            analyzer.score_text("bearish crash dump")
        signal = analyzer.generate_signal(-0.5)
        assert signal.direction == "bearish"

    def test_generate_signal_neutral(self, analyzer):
        signal = analyzer.generate_signal(0.0)
        assert signal.direction == "neutral"

    def test_get_recent_sentiment(self, analyzer):
        analyzer.score_text("bullish")
        recent = analyzer.get_recent_sentiment(hours=1)
        assert len(recent) >= 1

    def test_empty_aggregate(self, analyzer):
        assert analyzer.aggregate_sentiment([]) == 0.0
