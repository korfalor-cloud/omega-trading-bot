"""On-Chain Analytics.

Implements:
- Whale transaction detection
- Exchange flow analysis (inflows/outflows)
- Active address metrics
- Network value to transactions (NVT) ratio
- Miner revenue and hash rate signals
- Supply distribution analysis
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OnChainSignal:
    """On-chain derived signal."""
    metric: str = ""
    value: float = 0.0
    z_score: float = 0.0
    signal: str = ""  # bullish, bearish, neutral
    strength: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class WhaleAlert:
    """Detected whale transaction."""
    amount: float = 0.0
    direction: str = ""  # to_exchange, from_exchange, wallet_transfer
    signal: str = ""  # bullish, bearish
    timestamp: datetime = field(default_factory=datetime.utcnow)


class OnChainAnalyzer:
    """On-chain data analysis for trading signals.

    Analyzes blockchain metrics to generate signals about
    smart money movement and network health.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.whale_threshold_usd = config.get("whale_threshold_usd", 1_000_000)
        self.lookback_days = config.get("lookback_days", 30)
        self._exchange_flow_history: list[float] = []
        self._active_address_history: list[float] = []
        self._nvt_history: list[float] = []

    def analyze_exchange_flows(
        self,
        inflows: np.ndarray,
        outflows: np.ndarray,
    ) -> OnChainSignal:
        """Analyze exchange inflows/outflows.

        Net inflow → bearish (selling pressure)
        Net outflow → bullish (accumulation)
        """
        net_flow = np.sum(outflows) - np.sum(inflows)
        self._exchange_flow_history.append(net_flow)

        if len(self._exchange_flow_history) < 10:
            z_score = 0.0
        else:
            mean = np.mean(self._exchange_flow_history)
            std = np.std(self._exchange_flow_history)
            z_score = (net_flow - mean) / std if std > 0 else 0

        if z_score > 1.5:
            signal = "bullish"
            strength = min(1.0, z_score / 3)
        elif z_score < -1.5:
            signal = "bearish"
            strength = min(1.0, abs(z_score) / 3)
        else:
            signal = "neutral"
            strength = 0.0

        return OnChainSignal(
            metric="exchange_flow",
            value=net_flow,
            z_score=z_score,
            signal=signal,
            strength=strength,
        )

    def detect_whale_activity(
        self,
        transactions: list[dict],
    ) -> list[WhaleAlert]:
        """Detect whale transactions and classify direction."""
        alerts = []
        for tx in transactions:
            amount = tx.get("amount_usd", 0)
            if amount < self.whale_threshold_usd:
                continue

            from_addr = tx.get("from", "")
            to_addr = tx.get("to", "")
            is_from_exchange = "exchange" in from_addr.lower()
            is_to_exchange = "exchange" in to_addr.lower()

            if is_to_exchange:
                direction = "to_exchange"
                signal = "bearish"  # Whales sending to exchange = likely selling
            elif is_from_exchange:
                direction = "from_exchange"
                signal = "bullish"  # Withdrawing from exchange = accumulation
            else:
                direction = "wallet_transfer"
                signal = "neutral"

            alerts.append(WhaleAlert(
                amount=amount,
                direction=direction,
                signal=signal,
                timestamp=tx.get("timestamp", datetime.utcnow()),
            ))

        return alerts

    def nvt_ratio(
        self,
        market_cap: float,
        daily_transaction_volume: float,
    ) -> OnChainSignal:
        """Network Value to Transactions ratio.

        High NVT → overvalued (price exceeds utility)
        Low NVT → undervalued
        """
        nvt = market_cap / daily_transaction_volume if daily_transaction_volume > 0 else 0
        self._nvt_history.append(nvt)

        if len(self._nvt_history) < 10:
            z_score = 0.0
        else:
            mean = np.mean(self._nvt_history)
            std = np.std(self._nvt_history)
            z_score = (nvt - mean) / std if std > 0 else 0

        if z_score > 1.5:
            signal = "bearish"  # Overvalued
            strength = min(1.0, z_score / 3)
        elif z_score < -1.5:
            signal = "bullish"  # Undervalued
            strength = min(1.0, abs(z_score) / 3)
        else:
            signal = "neutral"
            strength = 0.0

        return OnChainSignal(
            metric="nvt_ratio",
            value=nvt,
            z_score=z_score,
            signal=signal,
            strength=strength,
        )

    def active_address_signal(
        self,
        active_addresses: np.ndarray,
    ) -> OnChainSignal:
        """Analyze active address trends.

        Rising addresses → bullish (network growth)
        Falling addresses → bearish (network decline)
        """
        if len(active_addresses) < 2:
            return OnChainSignal(metric="active_addresses")

        ma_short = np.mean(active_addresses[-7:])
        ma_long = np.mean(active_addresses[-30:]) if len(active_addresses) >= 30 else np.mean(active_addresses)

        self._active_address_history.append(float(active_addresses[-1]))

        momentum = (ma_short - ma_long) / ma_long if ma_long > 0 else 0

        if momentum > 0.1:
            signal = "bullish"
            strength = min(1.0, momentum * 5)
        elif momentum < -0.1:
            signal = "bearish"
            strength = min(1.0, abs(momentum) * 5)
        else:
            signal = "neutral"
            strength = 0.0

        return OnChainSignal(
            metric="active_addresses",
            value=float(active_addresses[-1]),
            z_score=momentum,
            signal=signal,
            strength=strength,
        )

    def hash_rate_signal(
        self,
        hash_rate: np.ndarray,
    ) -> OnChainSignal:
        """Analyze hash rate trends.

        Rising hash rate → bullish (miner confidence)
        Dropping hash rate → bearish (miner capitulation)
        """
        if len(hash_rate) < 14:
            return OnChainSignal(metric="hash_rate")

        ma_7 = np.mean(hash_rate[-7:])
        ma_30 = np.mean(hash_rate[-30:]) if len(hash_rate) >= 30 else np.mean(hash_rate)
        change = (ma_7 - ma_30) / ma_30 if ma_30 > 0 else 0

        if change > 0.05:
            signal = "bullish"
            strength = min(1.0, change * 10)
        elif change < -0.05:
            signal = "bearish"
            strength = min(1.0, abs(change) * 10)
        else:
            signal = "neutral"
            strength = 0.0

        return OnChainSignal(
            metric="hash_rate",
            value=float(hash_rate[-1]),
            z_score=change,
            signal=signal,
            strength=strength,
        )

    def aggregate_signals(self, signals: list[OnChainSignal]) -> dict:
        """Aggregate multiple on-chain signals into a composite view."""
        if not signals:
            return {"composite_score": 0, "direction": "neutral", "signals": []}

        scores = []
        for s in signals:
            if s.signal == "bullish":
                scores.append(s.strength)
            elif s.signal == "bearish":
                scores.append(-s.strength)
            else:
                scores.append(0)

        composite = np.mean(scores) if scores else 0

        if composite > 0.2:
            direction = "bullish"
        elif composite < -0.2:
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "composite_score": float(composite),
            "direction": direction,
            "n_signals": len(signals),
            "signals": [
                {"metric": s.metric, "signal": s.signal, "strength": s.strength}
                for s in signals
            ],
        }
