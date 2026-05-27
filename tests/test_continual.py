"""Tests for continual learning and drift detection."""
from __future__ import annotations

import pytest

from tradingbot.continual.online_learner import (
    ConceptDriftDetector,
    ContinualLearner,
    PerformanceMonitor,
)


class TestConceptDriftDetector:
    def test_no_drift_stable_data(self):
        detector = ConceptDriftDetector({"min_samples": 20})
        for _ in range(100):
            result = detector.update(0.001)
        assert not result.detected

    def test_drift_detection_mean_shift(self):
        detector = ConceptDriftDetector({"min_samples": 10, "ph_threshold": 1.0, "ph_delta": 0.0001})
        # Stable period with small positive values
        for _ in range(50):
            detector.update(0.001)
        # Large shift to negative
        for _ in range(50):
            result = detector.update(-0.5)
        # Statistic should be significant after large shift
        assert result.statistic > 0

    def test_reset(self):
        detector = ConceptDriftDetector()
        detector.update(1.0)
        detector.update(2.0)
        detector.reset()
        assert detector._n == 0


class TestPerformanceMonitor:
    def test_rolling_sharpe(self):
        monitor = PerformanceMonitor(window=50)
        # Positive returns
        for _ in range(50):
            monitor.update(0.001, 100000)
        assert monitor.get_rolling_sharpe() > 0

    def test_rolling_win_rate(self):
        monitor = PerformanceMonitor()
        for _ in range(30):
            monitor.update(0.01, 100000)  # Win
        for _ in range(20):
            monitor.update(-0.01, 100000)  # Loss
        wr = monitor.get_rolling_win_rate()
        assert abs(wr - 0.6) < 0.01

    def test_max_drawdown(self):
        monitor = PerformanceMonitor()
        monitor.update(0.0, 100000)
        monitor.update(0.0, 110000)
        monitor.update(0.0, 90000)
        dd = monitor.get_rolling_max_drawdown()
        assert dd > 0.1

    def test_degradation_detection(self):
        monitor = PerformanceMonitor()
        # Negative returns
        for _ in range(50):
            monitor.update(-0.005, 100000)
        assert monitor.is_degrading()


class TestContinualLearner:
    def test_no_adaptation_stable(self):
        learner = ContinualLearner()
        for _ in range(100):
            action = learner.observe(0.001, 100000)
        assert action is None  # No adaptation needed

    def test_adaptation_on_degradation(self):
        learner = ContinualLearner({"performance_window": 30, "adjust_threshold": 0.3})
        actions = []
        for _ in range(100):
            action = learner.observe(-0.01, 90000)
            if action:
                actions.append(action)
        # Should trigger adaptation
        assert len(actions) > 0

    def test_status(self):
        learner = ContinualLearner()
        learner.observe(0.01, 100000)
        status = learner.get_status()
        assert "rolling_sharpe" in status
        assert "total_adaptations" in status
