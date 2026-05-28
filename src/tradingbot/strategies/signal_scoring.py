"""Signal Scoring and Ranking.

Implements:
- Multi-factor signal scoring
- Signal ranking by expected value
- Signal filtering by confidence threshold
- Historical signal accuracy tracking
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScoredSignal:
    """A signal with a computed score."""
    signal_id: str = ""
    symbol: str = ""
    side: str = ""
    raw_strength: float = 0.0
    raw_confidence: float = 0.0
    technical_score: float = 0.0
    momentum_score: float = 0.0
    volume_score: float = 0.0
    regime_score: float = 0.0
    final_score: float = 0.0
    expected_value: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class SignalAccuracy:
    """Historical accuracy of a signal source."""
    source: str = ""
    total_signals: int = 0
    correct_signals: int = 0
    accuracy: float = 0.0
    avg_return: float = 0.0


class SignalScorer:
    """Multi-factor signal scoring engine."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.min_score = config.get("min_score", 0.3)
        self.weights = config.get("weights", {
            "technical": 0.30,
            "momentum": 0.25,
            "volume": 0.20,
            "regime": 0.25,
        })
        self._accuracy_history: dict[str, list[float]] = {}

    def score_signal(
        self,
        signal_id: str,
        symbol: str,
        side: str,
        strength: float,
        confidence: float,
        technical: float = 0.0,
        momentum: float = 0.0,
        volume: float = 0.0,
        regime: float = 0.0,
    ) -> ScoredSignal:
        """Compute composite score for a signal."""
        w = self.weights
        final = (
            technical * w["technical"]
            + momentum * w["momentum"]
            + volume * w["volume"]
            + regime * w["regime"]
        )

        # Adjust by raw confidence
        final = final * confidence

        expected_value = final * strength

        return ScoredSignal(
            signal_id=signal_id,
            symbol=symbol,
            side=side,
            raw_strength=strength,
            raw_confidence=confidence,
            technical_score=technical,
            momentum_score=momentum,
            volume_score=volume,
            regime_score=regime,
            final_score=final,
            expected_value=expected_value,
        )

    def rank_signals(self, signals: list[ScoredSignal]) -> list[ScoredSignal]:
        """Rank signals by final score."""
        return sorted(signals, key=lambda s: s.final_score, reverse=True)

    def filter_signals(
        self,
        signals: list[ScoredSignal],
        min_score: float | None = None,
        max_signals: int = 10,
    ) -> list[ScoredSignal]:
        """Filter and limit signals."""
        threshold = min_score if min_score is not None else self.min_score
        filtered = [s for s in signals if s.final_score >= threshold]
        return self.rank_signals(filtered)[:max_signals]

    def record_outcome(
        self,
        source: str,
        was_correct: bool,
        pnl: float,
    ) -> None:
        """Record signal outcome for accuracy tracking."""
        if source not in self._accuracy_history:
            self._accuracy_history[source] = []
        self._accuracy_history[source].append(1.0 if was_correct else 0.0)

    def get_accuracy(self, source: str) -> SignalAccuracy:
        """Get historical accuracy for a signal source."""
        outcomes = self._accuracy_history.get(source, [])
        if not outcomes:
            return SignalAccuracy(source=source)

        return SignalAccuracy(
            source=source,
            total_signals=len(outcomes),
            correct_signals=int(sum(outcomes)),
            accuracy=np.mean(outcomes),
        )

    def get_all_accuracies(self) -> list[SignalAccuracy]:
        """Get accuracy for all signal sources."""
        return [self.get_accuracy(s) for s in self._accuracy_history]
