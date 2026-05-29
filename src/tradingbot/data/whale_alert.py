"""Whale Alert — large transaction monitoring.

Implements:
- Large transaction detection
- Whale movement tracking
- Accumulation/distribution signals
- Whale behavior patterns
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WhaleTransaction:
    """A whale transaction."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    amount: float = 0.0
    value_usd: float = 0.0
    from_address: str = ""
    to_address: str = ""
    tx_type: str = ""  # exchange_deposit, exchange_withdrawal, whale_transfer


@dataclass
class WhaleSignal:
    """Whale activity signal."""
    signal: str = ""  # accumulation, distribution, neutral
    confidence: float = 0.0
    recent_volume: float = 0.0
    avg_transaction: float = 0.0


class WhaleAlertAnalyzer:
    """Analyze whale transactions."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.min_value_usd = config.get("min_value_usd", 100000)
        self._transactions: list[WhaleTransaction] = []
        self._max_history = config.get("max_history", 10000)

    def add_transaction(self, tx: WhaleTransaction) -> None:
        if tx.value_usd >= self.min_value_usd:
            self._transactions.append(tx)
            if len(self._transactions) > self._max_history:
                self._transactions = self._transactions[-self._max_history:]

    def add(self, symbol: str, amount: float, value_usd: float, tx_type: str = "transfer") -> None:
        self.add_transaction(WhaleTransaction(
            symbol=symbol, amount=amount, value_usd=value_usd, tx_type=tx_type,
        ))

    def get_recent(self, hours: float = 24) -> list[WhaleTransaction]:
        cutoff = datetime.utcnow().timestamp() - hours * 3600
        return [tx for tx in self._transactions if tx.timestamp.timestamp() > cutoff]

    def analyze(self, hours: float = 24) -> WhaleSignal:
        """Analyze recent whale activity."""
        recent = self.get_recent(hours)
        if not recent:
            return WhaleSignal(signal="neutral")

        deposits = [tx for tx in recent if tx.tx_type == "exchange_deposit"]
        withdrawals = [tx for tx in recent if tx.tx_type == "exchange_withdrawal"]

        deposit_vol = sum(tx.value_usd for tx in deposits)
        withdrawal_vol = sum(tx.value_usd for tx in withdrawals)

        if withdrawal_vol > deposit_vol * 1.5:
            signal = "accumulation"
            confidence = min(1.0, withdrawal_vol / (deposit_vol + 1))
        elif deposit_vol > withdrawal_vol * 1.5:
            signal = "distribution"
            confidence = min(1.0, deposit_vol / (withdrawal_vol + 1))
        else:
            signal = "neutral"
            confidence = 0.5

        return WhaleSignal(
            signal=signal,
            confidence=confidence,
            recent_volume=sum(tx.value_usd for tx in recent),
            avg_transaction=np.mean([tx.value_usd for tx in recent]) if recent else 0,
        )
