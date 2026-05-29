"""News API — real-time news feed integration.

Implements:
- News article collection
- Sentiment scoring
- Event detection
- News-based signals
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """A news article."""
    title: str = ""
    summary: str = ""
    source: str = ""
    url: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sentiment: float = 0.0
    relevance: float = 0.0
    symbols: list[str] = field(default_factory=list)
    event_type: str = ""  # listing, hack, regulation, partnership, etc.


@dataclass
class NewsSignal:
    """News-based trading signal."""
    signal: str = ""
    confidence: float = 0.0
    event_type: str = ""
    headline: str = ""


class NewsAPIAnalyzer:
    """Analyze news for trading signals."""

    def __init__(self, config: dict = None):
        config = config or {}
        self._articles: list[NewsArticle] = []
        self._max_articles = config.get("max_articles", 10000)

    def add_article(self, article: NewsArticle) -> None:
        self._articles.append(article)
        if len(self._articles) > self._max_articles:
            self._articles = self._articles[-self._max_articles:]

    def add(self, title: str, sentiment: float = 0, event_type: str = "", symbols: list[str] = None) -> None:
        self.add_article(NewsArticle(
            title=title, sentiment=sentiment, event_type=event_type, symbols=symbols or [],
        ))

    def get_recent(self, hours: float = 24) -> list[NewsArticle]:
        cutoff = datetime.utcnow().timestamp() - hours * 3600
        return [a for a in self._articles if a.timestamp.timestamp() > cutoff]

    def analyze(self, hours: float = 24) -> NewsSignal:
        """Analyze recent news."""
        recent = self.get_recent(hours)
        if not recent:
            return NewsSignal(signal="neutral")

        avg_sentiment = np.mean([a.sentiment for a in recent])

        # Event detection
        event_types = [a.event_type for a in recent if a.event_type]
        most_common_event = max(set(event_types), key=event_types.count) if event_types else ""

        if avg_sentiment > 0.3:
            signal = "bullish"
        elif avg_sentiment < -0.3:
            signal = "bearish"
        else:
            signal = "neutral"

        # High-impact events
        if most_common_event in ("listing", "partnership"):
            signal = "bullish"
        elif most_common_event in ("hack", "regulation", "ban"):
            signal = "bearish"

        return NewsSignal(
            signal=signal,
            confidence=min(1.0, abs(avg_sentiment)),
            event_type=most_common_event,
            headline=recent[0].title if recent else "",
        )

    def get_event_count(self, hours: float = 24) -> dict[str, int]:
        """Count events by type."""
        recent = self.get_recent(hours)
        counts = {}
        for a in recent:
            if a.event_type:
                counts[a.event_type] = counts.get(a.event_type, 0) + 1
        return counts
