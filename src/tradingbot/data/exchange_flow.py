"""Exchange Flow — exchange inflow/outflow tracking.

Implements:
- Exchange inflow monitoring
- Exchange outflow monitoring
- Net flow calculation
- Flow-based signals
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FlowState:
    """Exchange flow state."""
    inflow: float = 0.0
    outflow: float = 0.0
    netflow: float = 0.0
    cumulative_flow: float = 0.0
    signal: str = ""


class ExchangeFlowAnalyzer:
    """Analyze exchange inflow/outflow."""

    def __init__(self, config: dict = None):
        config = config or {}
        self._inflow_history: list[float] = []
        self._outflow_history: list[float] = []

    def update(self, inflow: float, outflow: float) -> None:
        self._inflow_history.append(inflow)
        self._outflow_history.append(outflow)

    def analyze(self, lookback: int = 24) -> FlowState:
        """Analyze recent flow."""
        if not self._inflow_history:
            return FlowState()

        inflow = sum(self._inflow_history[-lookback:])
        outflow = sum(self._outflow_history[-lookback:])
        netflow = inflow - outflow
        cumulative = sum(self._inflow_history) - sum(self._outflow_history)

        # Signal
        if netflow > 0:
            signal = "bearish"  # More inflow = selling pressure
        elif netflow < 0:
            signal = "bullish"  # More outflow = holding
        else:
            signal = "neutral"

        return FlowState(
            inflow=inflow,
            outflow=outflow,
            netflow=netflow,
            cumulative_flow=cumulative,
            signal=signal,
        )

    def get_trend(self, lookback: int = 7) -> str:
        """Get flow trend."""
        if len(self._inflow_history) < lookback * 2:
            return "neutral"

        recent_net = sum(self._inflow_history[-lookback:]) - sum(self._outflow_history[-lookback:])
        prev_net = sum(self._inflow_history[-lookback * 2:-lookback]) - sum(self._outflow_history[-lookback * 2:-lookback])

        if recent_net > prev_net * 1.2:
            return "increasing_inflow"
        elif recent_net < prev_net * 0.8:
            return "increasing_outflow"
        return "stable"
