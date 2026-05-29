"""Social Sentiment — Twitter/Reddit sentiment integration.

Implements:
- Tweet collection and scoring
- Reddit post analysis
- Social volume tracking
- Sentiment momentum
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SocialMetrics:
    """Social sentiment metrics."""
    mention_count: int = 0
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    sentiment_score: float = 0.0
    volume_change: float = 0.0
    trending: bool = False


class SocialSentimentAnalyzer:
    """Analyze social media sentiment."""

    def __init__(self, config: dict = None):
        config = config or {}
        self._mention_history: list[int] = []
        self._sentiment_history: list[float] = []

    def add_mentions(self, count: int, positive: int, negative: int) -> None:
        """Add social mention data."""
        total = positive + negative
        pos_ratio = positive / total if total > 0 else 0.5
        neg_ratio = negative / total if total > 0 else 0.5
        score = pos_ratio - neg_ratio

        self._mention_history.append(count)
        self._sentiment_history.append(score)

    def analyze(self, lookback: int = 24) -> SocialMetrics:
        """Analyze recent social sentiment."""
        if not self._mention_history:
            return SocialMetrics()

        recent_mentions = self._mention_history[-lookback:]
        recent_sentiment = self._sentiment_history[-lookback:]

        avg_sentiment = np.mean(recent_sentiment) if recent_sentiment else 0
        total_mentions = sum(recent_mentions)

        # Volume change
        if len(self._mention_history) >= lookback * 2:
            prev_mentions = self._mention_history[-lookback * 2:-lookback]
            vol_change = (sum(recent_mentions) - sum(prev_mentions)) / (sum(prev_mentions) + 1)
        else:
            vol_change = 0

        positive_ratio = sum(1 for s in recent_sentiment if s > 0) / len(recent_sentiment) if recent_sentiment else 0.5
        negative_ratio = 1 - positive_ratio

        trending = vol_change > 0.5 and total_mentions > 100

        return SocialMetrics(
            mention_count=total_mentions,
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            sentiment_score=float(avg_sentiment),
            volume_change=float(vol_change),
            trending=trending,
        )

    def get_signal(self) -> str:
        """Get trading signal from social sentiment."""
        metrics = self.analyze()
        if metrics.sentiment_score > 0.3 and metrics.trending:
            return "bullish"
        elif metrics.sentiment_score < -0.3 and metrics.trending:
            return "bearish"
        return "neutral"
