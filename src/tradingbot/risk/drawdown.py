"""Drawdown Analysis and Protection.

Implements:
- Real-time drawdown monitoring
- Drawdown duration tracking
- Recovery analysis
- Circuit breaker integration
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DrawdownState:
    """Current drawdown state."""
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    drawdown_duration: int = 0
    max_drawdown_duration: int = 0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    is_in_drawdown: bool = False
    recovery_pct: float = 0.0


class DrawdownMonitor:
    """Real-time drawdown monitoring and protection."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.max_allowed_dd = config.get("max_drawdown", 0.15)
        self.warning_dd = config.get("warning_drawdown", 0.10)
        self._peak_equity = 0.0
        self._current_equity = 0.0
        self._dd_duration = 0
        self._max_dd_duration = 0
        self._max_dd = 0.0
        self._equity_history: list[float] = []

    def update(self, equity: float) -> DrawdownState:
        """Update with new equity value."""
        self._current_equity = equity
        self._equity_history.append(equity)

        if equity > self._peak_equity:
            self._peak_equity = equity
            self._dd_duration = 0

        dd = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0
        self._max_dd = max(self._max_dd, dd)

        if dd > 0:
            self._dd_duration += 1
            self._max_dd_duration = max(self._max_dd_duration, self._dd_duration)

        recovery = 1 - dd if dd > 0 else 1.0

        return DrawdownState(
            current_drawdown=dd,
            max_drawdown=self._max_dd,
            drawdown_duration=self._dd_duration,
            max_drawdown_duration=self._max_dd_duration,
            peak_equity=self._peak_equity,
            current_equity=equity,
            is_in_drawdown=dd > 0,
            recovery_pct=recovery,
        )

    def is_circuit_breaker(self, equity: float) -> bool:
        """Check if drawdown exceeds max allowed."""
        dd = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0
        return dd >= self.max_allowed_dd

    def is_warning(self, equity: float) -> bool:
        """Check if drawdown exceeds warning level."""
        dd = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0
        return dd >= self.warning_dd

    def get_drawdown_series(self) -> np.ndarray:
        """Get historical drawdown series."""
        if not self._equity_history:
            return np.array([])
        equity = np.array(self._equity_history)
        peak = np.maximum.accumulate(equity)
        return (peak - equity) / peak

    def analyze_drawdowns(self) -> dict:
        """Analyze all drawdown periods."""
        dd_series = self.get_drawdown_series()
        if len(dd_series) == 0:
            return {"n_drawdowns": 0}

        in_dd = dd_series > 0
        starts = []
        ends = []
        max_dds = []

        current_start = None
        current_max = 0

        for i, is_dd in enumerate(in_dd):
            if is_dd and current_start is None:
                current_start = i
                current_max = dd_series[i]
            elif is_dd:
                current_max = max(current_max, dd_series[i])
            elif not is_dd and current_start is not None:
                starts.append(current_start)
                ends.append(i)
                max_dds.append(current_max)
                current_start = None
                current_max = 0

        if current_start is not None:
            starts.append(current_start)
            ends.append(len(dd_series))
            max_dds.append(current_max)

        durations = [e - s for s, e in zip(starts, ends)]

        return {
            "n_drawdowns": len(starts),
            "avg_duration": float(np.mean(durations)) if durations else 0,
            "max_duration": int(max(durations)) if durations else 0,
            "avg_depth": float(np.mean(max_dds)) if max_dds else 0,
            "max_depth": float(max(max_dds)) if max_dds else 0,
        }

    def reset(self) -> None:
        """Reset peak tracking."""
        self._peak_equity = self._current_equity
        self._dd_duration = 0
