"""Tests for drawdown monitor."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.risk.drawdown import DrawdownMonitor, DrawdownState


class TestDrawdownMonitor:
    @pytest.fixture
    def monitor(self):
        return DrawdownMonitor(config={"max_drawdown": 0.15, "warning_drawdown": 0.10})

    def test_initial_state(self, monitor):
        state = monitor.update(100000)
        assert state.current_drawdown == 0
        assert state.peak_equity == 100000

    def test_no_drawdown_rising(self, monitor):
        monitor.update(100000)
        monitor.update(105000)
        state = monitor.update(110000)
        assert state.current_drawdown == 0
        assert state.peak_equity == 110000

    def test_drawdown(self, monitor):
        monitor.update(100000)
        monitor.update(105000)
        state = monitor.update(95000)
        assert state.current_drawdown > 0
        assert state.peak_equity == 105000
        assert state.is_in_drawdown is True

    def test_max_drawdown(self, monitor):
        monitor.update(100000)
        monitor.update(90000)  # 10% DD
        monitor.update(95000)  # Recovery
        monitor.update(85000)  # ~15% DD from peak
        state = monitor.update(80000)
        assert state.max_drawdown > 0.15

    def test_drawdown_duration(self, monitor):
        monitor.update(100000)
        for _ in range(5):
            state = monitor.update(90000)
        assert state.drawdown_duration == 5

    def test_recovery(self, monitor):
        monitor.update(100000)
        monitor.update(90000)
        state = monitor.update(101000)  # Exceed peak to reset
        assert state.current_drawdown == 0
        assert state.drawdown_duration == 0

    def test_circuit_breaker(self, monitor):
        monitor.update(100000)
        assert monitor.is_circuit_breaker(84000) is True  # 16% DD
        assert monitor.is_circuit_breaker(90000) is False  # 10% DD

    def test_warning(self, monitor):
        monitor.update(100000)
        assert monitor.is_warning(89000) is True  # 11% DD
        assert monitor.is_warning(95000) is False  # 5% DD

    def test_drawdown_series(self, monitor):
        monitor.update(100000)
        monitor.update(105000)
        monitor.update(95000)
        monitor.update(100000)
        series = monitor.get_drawdown_series()
        assert len(series) == 4
        assert series[2] > 0  # Has drawdown

    def test_analyze_drawdowns(self, monitor):
        for eq in [100000, 105000, 95000, 100000, 110000, 100000]:
            monitor.update(eq)
        result = monitor.analyze_drawdowns()
        assert result["n_drawdowns"] >= 1

    def test_reset(self, monitor):
        monitor.update(100000)
        monitor.update(80000)
        monitor.reset()
        state = monitor.update(80000)
        assert state.current_drawdown == 0
        assert state.peak_equity == 80000

    def test_state_fields(self, monitor):
        state = monitor.update(100000)
        assert isinstance(state, DrawdownState)
        assert hasattr(state, "recovery_pct")
        assert hasattr(state, "max_drawdown_duration")
