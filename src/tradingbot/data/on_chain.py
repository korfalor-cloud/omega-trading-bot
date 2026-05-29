"""On-Chain Analytics — blockchain data analysis.

Implements:
- Whale transaction detection
- Exchange flow monitoring
- NVT ratio calculation
- MVRV ratio
- Active addresses
- Mining metrics
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OnChainMetrics:
    """On-chain analytics metrics."""
    nvt_ratio: float = 0.0
    mvrv_ratio: float = 0.0
    active_addresses: int = 0
    exchange_inflow: float = 0.0
    exchange_outflow: float = 0.0
    exchange_netflow: float = 0.0
    whale_transactions: int = 0
    sopr: float = 0.0
    puell_multiple: float = 0.0
    stock_to_flow: float = 0.0


class OnChainAnalyzer:
    """On-chain data analysis engine."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.whale_threshold = config.get("whale_threshold", 100)  # BTC
        self._price_history: list[float] = []
        self._volume_history: list[float] = []
        self._tx_volume_history: list[float] = []

    def update(self, price: float, volume: float, tx_volume: float = 0) -> None:
        """Update with new data."""
        self._price_history.append(price)
        self._volume_history.append(volume)
        self._tx_volume_history.append(tx_volume or volume * price)

    def compute_nvt(self, lookback: int = 30) -> float:
        """Network Value to Transactions ratio."""
        if len(self._price_history) < lookback or len(self._tx_volume_history) < lookback:
            return 0

        market_cap = np.mean(self._price_history[-lookback:]) * 21e6  # BTC supply proxy
        tx_vol = np.mean(self._tx_volume_history[-lookback:])

        return market_cap / tx_vol if tx_vol > 0 else 0

    def compute_mvrv(self, lookback: int = 365) -> float:
        """Market Value to Realized Value."""
        if len(self._price_history) < lookback:
            return 1.0

        current = self._price_history[-1]
        realized = np.mean(self._price_history[-lookback:])
        return current / realized if realized > 0 else 1.0

    def compute_sopr(self) -> float:
        """Spent Output Profit Ratio."""
        if len(self._price_history) < 2:
            return 1.0
        return self._price_history[-1] / self._price_history[-2]

    def compute_puell_multiple(self, lookback: int = 365) -> float:
        """Puell Multiple (mining revenue / 365d avg)."""
        if len(self._price_history) < lookback:
            return 1.0

        daily_revenue = self._price_history[-1] * 6.25 * 144  # BTC block reward * blocks/day
        avg_revenue = np.mean(self._price_history[-lookback:]) * 6.25 * 144

        return daily_revenue / avg_revenue if avg_revenue > 0 else 1.0

    def compute_stock_to_flow(self, current_supply: float = 19.5e6) -> float:
        """Stock-to-Flow model."""
        annual_production = 328500  # ~900 BTC/day * 365
        return current_supply / annual_production if annual_production > 0 else 0

    def detect_whale_activity(self, transactions: list[dict]) -> list[dict]:
        """Detect whale transactions."""
        return [tx for tx in transactions if tx.get("amount", 0) >= self.whale_threshold]

    def compute_exchange_flow(self, inflows: list[float], outflows: list[float]) -> dict:
        """Compute exchange flow metrics."""
        return {
            "inflow": sum(inflows),
            "outflow": sum(outflows),
            "netflow": sum(inflows) - sum(outflows),
            "signal": "bearish" if sum(inflows) > sum(outflows) else "bullish",
        }

    def get_metrics(self) -> OnChainMetrics:
        """Get all on-chain metrics."""
        return OnChainMetrics(
            nvt_ratio=self.compute_nvt(),
            mvrv_ratio=self.compute_mvrv(),
            active_addresses=0,
            exchange_netflow=0,
            sopr=self.compute_sopr(),
            puell_multiple=self.compute_puell_multiple(),
            stock_to_flow=self.compute_stock_to_flow(),
        )

    def get_signal(self) -> str:
        """Get overall on-chain signal."""
        nvt = self.compute_nvt()
        mvrv = self.compute_mvrv()

        if nvt > 100 and mvrv > 2:
            return "bearish"
        elif nvt < 50 and mvrv < 1:
            return "bullish"
        return "neutral"
