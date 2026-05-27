"""Sentiment Analysis for Trading.

Implements:
- Text sentiment scoring (lexicon-based)
- Aggregated sentiment from multiple sources
- Sentiment momentum and divergence signals
- Fear & Greed index estimation
- Sentiment-based trade signals
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Crypto-specific sentiment lexicon
POSITIVE_WORDS = {
    "bullish", "moon", "pump", "breakout", "surge", "rally", "buy",
    "accumulate", "hodl", "long", "support", "higher", "gain", "profit",
    "adoption", "partnership", "upgrade", "milestone", "record", "growth",
    "optimistic", "recovery", "bounce", "momentum", "inflow",
}

NEGATIVE_WORDS = {
    "bearish", "dump", "crash", "sell", "short", "resistance", "lower",
    "loss", "fear", "panic", "liquidation", "hack", "scam", "rug",
    "ban", "regulation", "warning", "decline", "outflow", "capitulation",
    "correction", "overbought", "bubble", "fraud", "collapse",
}


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    score: float = 0.0  # -1 (bearish) to +1 (bullish)
    magnitude: float = 0.0  # Strength of sentiment
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_data: dict = field(default_factory=dict)


@dataclass
class SentimentSignal:
    """Trading signal derived from sentiment."""
    direction: str = ""  # bullish, bearish, neutral
    strength: float = 0.0
    confidence: float = 0.0
    sentiment_score: float = 0.0
    sentiment_momentum: float = 0.0
    divergence: bool = False  # Price vs sentiment divergence


class SentimentAnalyzer:
    """Sentiment analysis engine.

    Scores text sentiment and aggregates across sources
    to generate trading signals.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.lookback_hours = config.get("lookback_hours", 24)
        self.min_sources = config.get("min_sources", 3)
        self._history: list[SentimentResult] = []

    def score_text(self, text: str, source: str = "text") -> SentimentResult:
        """Score sentiment of a single text."""
        words = set(re.findall(r'\w+', text.lower()))
        pos = len(words & POSITIVE_WORDS)
        neg = len(words & NEGATIVE_WORDS)
        total = pos + neg

        if total == 0:
            score = 0.0
        else:
            score = (pos - neg) / total

        magnitude = total / max(len(words), 1)

        result = SentimentResult(
            score=score,
            magnitude=magnitude,
            source=source,
        )
        self._history.append(result)
        return result

    def score_batch(
        self,
        texts: list[str],
        source: str = "batch",
    ) -> SentimentResult:
        """Score sentiment of multiple texts and aggregate."""
        if not texts:
            return SentimentResult()

        scores = [self.score_text(t, source).score for t in texts]
        avg_score = np.mean(scores)
        magnitude = np.std(scores) if len(scores) > 1 else 0

        return SentimentResult(
            score=avg_score,
            magnitude=magnitude,
            source=source,
        )

    def aggregate_sentiment(
        self,
        results: list[SentimentResult],
        weights: Optional[list[float]] = None,
    ) -> float:
        """Aggregate sentiment from multiple sources with optional weights."""
        if not results:
            return 0.0

        scores = [r.score for r in results]

        if weights and len(weights) == len(scores):
            total_weight = sum(weights)
            if total_weight > 0:
                return sum(s * w for s, w in zip(scores, weights)) / total_weight

        return float(np.mean(scores))

    def sentiment_momentum(self, window: int = 10) -> float:
        """Compute sentiment momentum (change in sentiment over time)."""
        if len(self._history) < window:
            return 0.0

        recent = [r.score for r in self._history[-window:]]
        older = [r.score for r in self._history[-window * 2:-window]] if len(self._history) >= window * 2 else [0]

        return float(np.mean(recent) - np.mean(older))

    def detect_divergence(
        self,
        price_returns: np.ndarray,
        sentiment_scores: np.ndarray,
        window: int = 10,
    ) -> bool:
        """Detect price-sentiment divergence.

        Bearish divergence: price up, sentiment down
        Bullish divergence: price down, sentiment up
        """
        n = min(len(price_returns), len(sentiment_scores))
        if n < window:
            return False

        price_trend = np.mean(price_returns[-window:])
        sent_trend = np.mean(sentiment_scores[-window:])

        # Divergence: opposite signs with magnitude
        return (price_trend > 0 and sent_trend < -0.1) or (price_trend < 0 and sent_trend > 0.1)

    def fear_greed_index(
        self,
        volatility: float,
        momentum: float,
        sentiment: float,
        dominance: float = 0.5,
        volume: float = 0.0,
    ) -> float:
        """Estimate fear & greed index (0 = extreme fear, 100 = extreme greed).

        Combines multiple inputs:
        - Volatility (inverted — high vol = fear)
        - Momentum (positive = greed)
        - Sentiment (positive = greed)
        - Market dominance
        - Volume momentum
        """
        # Normalize each component to 0-1
        vol_score = max(0, min(1, 1 - volatility * 10))  # Invert: low vol = greed
        mom_score = max(0, min(1, (momentum + 0.05) / 0.10))  # Center around 0
        sent_score = max(0, min(1, (sentiment + 1) / 2))  # -1 to 1 → 0 to 1
        dom_score = dominance
        vol_mom = max(0, min(1, (volume + 1) / 2))

        # Weighted average
        index = (
            vol_score * 0.25
            + mom_score * 0.25
            + sent_score * 0.30
            + dom_score * 0.10
            + vol_mom * 0.10
        )

        return float(index * 100)

    def generate_signal(
        self,
        current_sentiment: float,
        price_returns: Optional[np.ndarray] = None,
    ) -> SentimentSignal:
        """Generate trading signal from sentiment analysis."""
        momentum = self.sentiment_momentum()

        # Check divergence
        divergence = False
        if price_returns is not None and len(self._history) >= 10:
            sent_scores = np.array([r.score for r in self._history[-len(price_returns):]])
            if len(sent_scores) == len(price_returns):
                divergence = self.detect_divergence(price_returns, sent_scores)

        # Signal logic
        if current_sentiment > 0.3 and momentum > 0:
            direction = "bullish"
            strength = min(1.0, current_sentiment)
            confidence = min(1.0, 0.5 + abs(momentum))
        elif current_sentiment < -0.3 and momentum < 0:
            direction = "bearish"
            strength = min(1.0, abs(current_sentiment))
            confidence = min(1.0, 0.5 + abs(momentum))
        else:
            direction = "neutral"
            strength = 0.0
            confidence = 0.3

        # Reduce confidence on divergence
        if divergence:
            confidence *= 0.7

        return SentimentSignal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            sentiment_score=current_sentiment,
            sentiment_momentum=momentum,
            divergence=divergence,
        )

    def get_recent_sentiment(self, hours: int = 24) -> list[SentimentResult]:
        """Get sentiment results from the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [r for r in self._history if r.timestamp >= cutoff]
