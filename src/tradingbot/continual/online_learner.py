"""Continual / Online Learning — Adapt strategies to changing markets.

Detects concept drift and triggers model retraining or strategy
parameter adjustment when market dynamics shift.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftDetection:
    """Result of concept drift detection."""
    detected: bool
    method: str
    statistic: float
    threshold: float
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)


@dataclass
class AdaptationAction:
    """Action to take when drift is detected."""
    action_type: str  # retrain, adjust_params, replace_strategy, increase_diversity
    reason: str
    urgency: float  # 0-1
    parameters: dict = field(default_factory=dict)


class ConceptDriftDetector:
    """Detect concept drift using multiple statistical methods.

    Methods:
    - Page-Hinkley test: detects changes in mean
    - ADWIN: adaptive windowing for distribution changes
    - Performance monitoring: tracks rolling accuracy/returns
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.window_size = cfg.get("window_size", 200)
        self.ph_delta = cfg.get("ph_delta", 0.005)
        self.ph_threshold = cfg.get("ph_threshold", 50.0)
        self.adwin_delta = cfg.get("adwin_delta", 0.002)
        self.min_samples = cfg.get("min_samples", 50)

        self._values: deque = deque(maxlen=self.window_size * 2)
        self._ph_sum = 0.0
        self._ph_min = float("inf")
        self._running_mean = 0.0
        self._n = 0

    def update(self, value: float) -> DriftDetection:
        """Add a new observation and check for drift."""
        self._values.append(value)
        self._n += 1

        # Update running mean
        old_mean = self._running_mean
        self._running_mean += (value - self._running_mean) / self._n

        # Page-Hinkley test
        self._ph_sum += value - self._running_mean - self.ph_delta
        self._ph_min = min(self._ph_min, self._ph_sum)
        ph_stat = self._ph_sum - self._ph_min

        ph_drift = ph_stat > self.ph_threshold

        # ADWIN check
        adwin_drift, adwin_stat = self._check_adwin()

        # Combined detection
        detected = ph_drift or adwin_drift
        method = "page_hinkley" if ph_drift else ("adwin" if adwin_drift else "none")
        confidence = min(1.0, max(ph_stat / self.ph_threshold, adwin_stat / self.adwin_delta if adwin_stat else 0))

        return DriftDetection(
            detected=detected,
            method=method,
            statistic=max(ph_stat, adwin_stat or 0),
            threshold=self.ph_threshold if ph_drift else self.adwin_delta,
            confidence=confidence,
        )

    def _check_adwin(self) -> tuple[bool, Optional[float]]:
        """ADWIN: check if splitting the window reveals distribution change."""
        values = list(self._values)
        if len(values) < self.min_samples * 2:
            return False, None

        n = len(values)
        max_stat = 0.0
        best_split = None

        # Check splits at various points
        for split in range(self.min_samples, n - self.min_samples, max(1, n // 20)):
            left = values[:split]
            right = values[split:]

            mean_left = np.mean(left)
            mean_right = np.mean(right)
            n_left = len(left)
            n_right = len(right)

            # Harmonic mean of window sizes
            m = 1.0 / (1.0 / n_left + 1.0 / n_right)

            # Test statistic
            diff = abs(mean_left - mean_right)
            stat = diff * math.sqrt(m / 2)

            if stat > max_stat:
                max_stat = stat
                best_split = split

        # Compare against threshold (epsilon cut)
        threshold = math.sqrt(1.0 / (2.0 * self.min_samples) * math.log(2.0 / self.adwin_delta))

        return max_stat > threshold, max_stat

    def reset(self) -> None:
        """Reset detector state."""
        self._values.clear()
        self._ph_sum = 0.0
        self._ph_min = float("inf")
        self._running_mean = 0.0
        self._n = 0


class PerformanceMonitor:
    """Track rolling strategy performance metrics."""

    def __init__(self, window: int = 100):
        self.window = window
        self._returns: deque = deque(maxlen=window)
        self._equity_curve: deque = deque(maxlen=window)

    def update(self, trade_return: float, equity: float) -> None:
        self._returns.append(trade_return)
        self._equity_curve.append(equity)

    def get_rolling_sharpe(self) -> float:
        if len(self._returns) < 10:
            return 0.0
        returns = list(self._returns)
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(var) if var > 0 else 1e-10
        return mean / std * math.sqrt(252)

    def get_rolling_win_rate(self) -> float:
        if not self._returns:
            return 0.0
        wins = sum(1 for r in self._returns if r > 0)
        return wins / len(self._returns)

    def get_rolling_max_drawdown(self) -> float:
        equity = list(self._equity_curve)
        if len(equity) < 2:
            return 0.0
        peak = equity[0]
        max_dd = 0.0
        for eq in equity:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd

    def is_degrading(self, threshold_sharpe: float = 0.5) -> bool:
        """Check if performance is degrading below threshold."""
        return self.get_rolling_sharpe() < threshold_sharpe


class ContinualLearner:
    """Orchestrates continual learning and strategy adaptation.

    Combines drift detection with performance monitoring to decide
    when and how to adapt strategies.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.drift_detector = ConceptDriftDetector(cfg.get("drift", {}))
        self.performance_monitor = PerformanceMonitor(cfg.get("performance_window", 100))
        self.retrain_threshold = cfg.get("retrain_threshold", 0.7)
        self.adjust_threshold = cfg.get("adjust_threshold", 0.5)
        self._adaptation_history: list[AdaptationAction] = []

    def observe(self, trade_return: float, equity: float, features: Optional[dict] = None) -> Optional[AdaptationAction]:
        """Observe a new trade result and decide if adaptation is needed."""
        # Update monitors
        self.performance_monitor.update(trade_return, equity)

        # Check drift on returns
        drift = self.drift_detector.update(trade_return)

        # Check performance degradation
        perf_degraded = self.performance_monitor.is_degrading()

        # Decide action
        action = self._decide_action(drift, perf_degraded, features)

        if action:
            self._adaptation_history.append(action)
            logger.warning(f"Continual learning triggered: {action.action_type} — {action.reason}")

        return action

    def _decide_action(
        self,
        drift: DriftDetection,
        perf_degraded: bool,
        features: Optional[dict],
    ) -> Optional[AdaptationAction]:
        """Decide what adaptation action to take."""

        # High confidence drift + performance degradation → retrain
        if drift.detected and drift.confidence > self.retrain_threshold and perf_degraded:
            return AdaptationAction(
                action_type="retrain",
                reason=f"Concept drift detected ({drift.method}, confidence={drift.confidence:.2f}) with performance degradation",
                urgency=drift.confidence,
                parameters={"drift_method": drift.method, "drift_stat": drift.statistic},
            )

        # Moderate drift → adjust parameters
        if drift.detected and drift.confidence > self.adjust_threshold:
            return AdaptationAction(
                action_type="adjust_params",
                reason=f"Moderate drift detected ({drift.method})",
                urgency=drift.confidence * 0.7,
                parameters={"drift_method": drift.method},
            )

        # Performance degrading without clear drift → increase diversity
        if perf_degraded and not drift.detected:
            return AdaptationAction(
                action_type="increase_diversity",
                reason="Performance degrading without clear drift signal",
                urgency=0.4,
                parameters={"rolling_sharpe": self.performance_monitor.get_rolling_sharpe()},
            )

        return None

    def get_status(self) -> dict:
        """Get current continual learning status."""
        return {
            "rolling_sharpe": self.performance_monitor.get_rolling_sharpe(),
            "rolling_win_rate": self.performance_monitor.get_rolling_win_rate(),
            "rolling_max_drawdown": self.performance_monitor.get_rolling_max_drawdown(),
            "total_adaptations": len(self._adaptation_history),
            "last_adaptation": self._adaptation_history[-1].action_type if self._adaptation_history else None,
        }
