"""Sentiment Analysis — Market sentiment from social media and news.

Provides sentiment scoring from various sources including Twitter/X,
Reddit, and news feeds. Uses simple NLP heuristics (no external API
required for basic functionality).
"""
from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Crypto-specific sentiment lexicon
_BULLISH_WORDS = {
    "moon", "bullish", "pump", "long", "buy", "accumulate", "hodl",
    "breakout", "rally", "surge", "ath", "all time high", "rocket",
    "undervalued", "cheap", "dip buy", "bottom", "reversal", "golden cross",
    "bull run", "uptrend", "higher highs", "higher lows", "support",
    "bounce", "recovery", "strong", "demand", "inflow", "whale buying",
}

_BEARISH_WORDS = {
    "bearish", "dump", "crash", "short", "sell", "overvalued", "bubble",
    "death cross", "correction", "capitulation", "fear", "panic",
    "resistance", "rejection", "lower highs", "lower lows", "breakdown",
    "liquidation", "outflow", "whale selling", "rug pull", "scam",
    "bear market", "downtrend", "weak", "supply", "fud",
}

_INTENSIFIERS = {
    "very": 1.5, "extremely": 2.0, "massive": 1.8, "huge": 1.7,
    "insane": 2.0, "absolutely": 1.8, "definitely": 1.3, "certainly": 1.3,
}

_NEGATORS = {"not", "no", "never", "don't", "doesn't", "won't", "isn't", "aren't"}


@dataclass
class SentimentScore:
    """Sentiment score for a text or collection of texts."""
    score: float  # -1.0 (very bearish) to 1.0 (very bullish)
    magnitude: float  # 0.0 to 1.0, strength of sentiment
    bullish_signals: int
    bearish_signals: int
    source: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)


class SentimentAnalyzer:
    """Analyze market sentiment from text using lexicon-based approach.

    Simple but effective for crypto markets. Can be extended with
    transformer models for better accuracy.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.decay_hours = cfg.get("decay_hours", 24)
        self._history: deque[SentimentScore] = deque(maxlen=10000)

    def analyze_text(self, text: str, source: str = "unknown") -> SentimentScore:
        """Analyze sentiment of a single text."""
        words = set(re.findall(r'\b\w+\b', text.lower()))
        bigrams = set()
        tokens = text.lower().split()
        for i in range(len(tokens) - 1):
            bigrams.add(f"{tokens[i]} {tokens[i+1]}")

        all_tokens = words | bigrams

        # Check for negation
        has_negation = bool(words & _NEGATORS)

        bullish_count = 0
        bearish_count = 0

        for token in all_tokens:
            # Check intensifiers
            intensity = 1.0
            for intensifier, mult in _INTENSIFIERS.items():
                if intensifier in text.lower():
                    intensity = mult
                    break

            if token in _BULLISH_WORDS:
                bullish_count += intensity
            if token in _BEARISH_WORDS:
                bearish_count += intensity

        # Apply negation flip
        if has_negation:
            bullish_count, bearish_count = bearish_count * 0.5, bullish_count * 0.5

        total = bullish_count + bearish_count
        if total == 0:
            score = 0.0
            magnitude = 0.0
        else:
            score = (bullish_count - bearish_count) / total
            magnitude = min(1.0, total / 10)

        result = SentimentScore(
            score=score,
            magnitude=magnitude,
            bullish_signals=int(bullish_count),
            bearish_signals=int(bearish_count),
            source=source,
        )

        self._history.append(result)
        return result

    def analyze_batch(self, texts: list[str], source: str = "batch") -> SentimentScore:
        """Analyze sentiment of multiple texts and aggregate."""
        if not texts:
            return SentimentScore(0.0, 0.0, 0, 0, source)

        scores = [self.analyze_text(t, source) for t in texts]

        # Weighted average by magnitude
        total_mag = sum(s.magnitude for s in scores)
        if total_mag == 0:
            avg_score = 0.0
        else:
            avg_score = sum(s.score * s.magnitude for s in scores) / total_mag

        return SentimentScore(
            score=avg_score,
            magnitude=total_mag / len(scores),
            bullish_signals=sum(s.bullish_signals for s in scores),
            bearish_signals=sum(s.bearish_signals for s in scores),
            source=source,
            metadata={"n_texts": len(scores)},
        )

    def get_aggregate_sentiment(self, hours: Optional[int] = None) -> SentimentScore:
        """Get aggregate sentiment from recent history."""
        hours = hours or self.decay_hours
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - hours * 3600

        recent = [s for s in self._history if s.timestamp.timestamp() > cutoff]
        if not recent:
            return SentimentScore(0.0, 0.0, 0, 0, "aggregate")

        # Time-weighted average (more recent = higher weight)
        weights = []
        for s in recent:
            age_hours = (now - s.timestamp).total_seconds() / 3600
            weight = max(0.1, 1.0 - age_hours / hours) * s.magnitude
            weights.append(weight)

        total_weight = sum(weights)
        if total_weight == 0:
            return SentimentScore(0.0, 0.0, 0, 0, "aggregate")

        weighted_score = sum(s.score * w for s, w in zip(recent, weights)) / total_weight

        return SentimentScore(
            score=weighted_score,
            magnitude=min(1.0, total_weight / len(recent)),
            bullish_signals=sum(s.bullish_signals for s in recent),
            bearish_signals=sum(s.bearish_signals for s in recent),
            source="aggregate",
            metadata={"n_signals": len(recent), "hours": hours},
        )

    def get_sentiment_shift(self, lookback_hours: int = 4) -> float:
        """Detect sentiment shift (change in recent vs older sentiment)."""
        now = datetime.now(timezone.utc)
        recent_cutoff = now.timestamp() - lookback_hours * 3600
        old_cutoff = now.timestamp() - lookback_hours * 2 * 3600

        recent = [s for s in self._history if s.timestamp.timestamp() > recent_cutoff]
        older = [s for s in self._history if old_cutoff < s.timestamp.timestamp() <= recent_cutoff]

        if not recent or not older:
            return 0.0

        recent_avg = sum(s.score for s in recent) / len(recent)
        older_avg = sum(s.score for s in older) / len(older)

        return recent_avg - older_avg
