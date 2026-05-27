"""Tests for sentiment analysis and on-chain data."""
from __future__ import annotations

import pytest

from tradingbot.data.alternative.sentiment import SentimentAnalyzer, SentimentScore
from tradingbot.data.alternative.on_chain import OnChainDataProvider, OnChainMetrics


class TestSentimentAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return SentimentAnalyzer()

    def test_bullish_text(self, analyzer):
        result = analyzer.analyze_text("BTC is looking very bullish, expecting a massive pump soon!")
        assert result.score > 0
        assert result.bullish_signals > 0

    def test_bearish_text(self, analyzer):
        result = analyzer.analyze_text("Market looking bearish, expecting a crash and dump")
        assert result.score < 0
        assert result.bearish_signals > 0

    def test_neutral_text(self, analyzer):
        result = analyzer.analyze_text("The weather is nice today")
        assert result.score == 0
        assert result.magnitude == 0

    def test_batch_analysis(self, analyzer):
        texts = [
            "Very bullish on BTC!",
            "Moon incoming!",
            "Bear market is over, time to buy",
        ]
        result = analyzer.analyze_batch(texts, "test")
        assert result.score > 0
        assert result.metadata["n_texts"] == 3

    def test_aggregate_sentiment(self, analyzer):
        analyzer.analyze_text("Bullish!", "twitter")
        analyzer.analyze_text("Moon!", "reddit")
        agg = analyzer.get_aggregate_sentiment()
        assert isinstance(agg, SentimentScore)

    def test_sentiment_shift(self, analyzer):
        # No data
        shift = analyzer.get_sentiment_shift()
        assert shift == 0.0

    def test_negation_handling(self, analyzer):
        result = analyzer.analyze_text("Not bullish at all, this is not going to moon")
        # Negation should flip some sentiment
        assert result.score <= 0


class TestOnChainDataProvider:
    @pytest.fixture
    def provider(self):
        return OnChainDataProvider()

    @pytest.mark.asyncio
    async def test_synthetic_metrics(self, provider):
        metrics = provider._generate_synthetic_metrics("BTC")
        assert isinstance(metrics, OnChainMetrics)
        assert metrics.symbol == "BTC"
        assert metrics.tx_count_24h > 0

    def test_signal_computation(self, provider):
        metrics = OnChainMetrics(
            symbol="BTC",
            timestamp=None,
            exchange_inflow_usd=1e8,
            exchange_outflow_usd=2e8,
            whale_sentiment=0.5,
            accumulation_score=0.3,
            tx_count_24h=300000,
        )
        signals = provider.compute_signals(metrics)
        assert "exchange_flow" in signals
        assert signals["exchange_flow"] > 0  # More outflow = bullish
        assert signals["whale_sentiment"] == 0.5
