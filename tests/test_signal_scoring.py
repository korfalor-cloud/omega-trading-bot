"""Tests for signal scoring."""
from __future__ import annotations

import pytest

from tradingbot.strategies.signal_scoring import (
    ScoredSignal,
    SignalAccuracy,
    SignalScorer,
)


class TestSignalScorer:
    @pytest.fixture
    def scorer(self):
        return SignalScorer()

    def test_score_signal(self, scorer):
        sig = scorer.score_signal(
            signal_id="s1", symbol="BTC/USDT", side="buy",
            strength=0.8, confidence=0.7,
            technical=0.9, momentum=0.8, volume=0.6, regime=0.7,
        )
        assert sig.final_score > 0
        assert sig.expected_value > 0
        assert sig.symbol == "BTC/USDT"

    def test_rank_signals(self, scorer):
        signals = [
            scorer.score_signal("s1", "BTC/USDT", "buy", 0.5, 0.5, technical=0.3),
            scorer.score_signal("s2", "ETH/USDT", "buy", 0.8, 0.9, technical=0.9),
            scorer.score_signal("s3", "SOL/USDT", "buy", 0.6, 0.7, technical=0.5),
        ]
        ranked = scorer.rank_signals(signals)
        assert ranked[0].signal_id == "s2"

    def test_filter_signals(self, scorer):
        signals = [
            scorer.score_signal("s1", "BTC/USDT", "buy", 0.5, 0.5, technical=0.1, momentum=0.1, volume=0.1, regime=0.1),
            scorer.score_signal("s2", "ETH/USDT", "buy", 0.8, 0.9, technical=0.9, momentum=0.8, volume=0.7, regime=0.8),
        ]
        filtered = scorer.filter_signals(signals, min_score=0.3)
        assert len(filtered) == 1
        assert filtered[0].signal_id == "s2"

    def test_filter_max_signals(self, scorer):
        signals = [
            scorer.score_signal(f"s{i}", "BTC/USDT", "buy", 0.8, 0.9, technical=0.9, momentum=0.8, volume=0.7, regime=0.8)
            for i in range(20)
        ]
        filtered = scorer.filter_signals(signals, max_signals=5)
        assert len(filtered) == 5

    def test_record_outcome(self, scorer):
        scorer.record_outcome("trend", True, 100)
        scorer.record_outcome("trend", True, 50)
        scorer.record_outcome("trend", False, -30)
        acc = scorer.get_accuracy("trend")
        assert acc.accuracy == pytest.approx(2 / 3, abs=0.01)
        assert acc.total_signals == 3

    def test_accuracy_empty(self, scorer):
        acc = scorer.get_accuracy("nonexistent")
        assert acc.total_signals == 0
        assert acc.accuracy == 0.0

    def test_all_accuracies(self, scorer):
        scorer.record_outcome("a", True, 100)
        scorer.record_outcome("b", False, -50)
        accs = scorer.get_all_accuracies()
        assert len(accs) == 2
