"""Sector Exposure — crypto sector risk management.

Implements:
- Sector classification
- Sector exposure tracking
- Sector concentration limits
- Cross-sector correlation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SectorExposure:
    """Sector exposure report."""
    sector: str = ""
    exposure_usd: float = 0.0
    exposure_pct: float = 0.0
    n_positions: int = 0
    risk_level: str = "low"


# Default sector mapping
SECTOR_MAP = {
    "BTC": "L1", "ETH": "L1", "SOL": "L1", "ADA": "L1", "DOT": "L1",
    "UNI": "DeFi", "AAVE": "DeFi", "MKR": "DeFi", "COMP": "DeFi", "SNX": "DeFi",
    "MATIC": "L2", "ARB": "L2", "OP": "L2",
    "DOGE": "Meme", "SHIB": "Meme", "PEPE": "Meme",
    "LINK": "Oracle", "AVAX": "L1", "NEAR": "L1",
}


class SectorExposureManager:
    """Manage sector-level exposure."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.max_sector_pct = config.get("max_sector_pct", 0.40)
        self._positions: dict[str, float] = {}

    def update_position(self, symbol: str, notional: float) -> None:
        self._positions[symbol] = notional

    def get_sector(self, symbol: str) -> str:
        base = symbol.split("/")[0].upper()
        return SECTOR_MAP.get(base, "Other")

    def analyze(self) -> list[SectorExposure]:
        """Analyze sector exposure."""
        sector_values: dict[str, float] = {}
        sector_counts: dict[str, int] = {}

        for symbol, notional in self._positions.items():
            sector = self.get_sector(symbol)
            sector_values[sector] = sector_values.get(sector, 0) + abs(notional)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        total = sum(sector_values.values())
        results = []

        for sector, value in sector_values.items():
            pct = value / total if total > 0 else 0
            risk = "high" if pct > self.max_sector_pct else "medium" if pct > self.max_sector_pct * 0.7 else "low"
            results.append(SectorExposure(
                sector=sector,
                exposure_usd=value,
                exposure_pct=pct,
                n_positions=sector_counts.get(sector, 0),
                risk_level=risk,
            ))

        return sorted(results, key=lambda x: x.exposure_pct, reverse=True)

    def get_concentration_risk(self) -> float:
        """Get sector concentration risk (HHI)."""
        total = sum(abs(v) for v in self._positions.values())
        if total == 0:
            return 0

        sector_values: dict[str, float] = {}
        for symbol, notional in self._positions.items():
            sector = self.get_sector(symbol)
            sector_values[sector] = sector_values.get(sector, 0) + abs(notional)

        weights = [v / total for v in sector_values.values()]
        return sum(w ** 2 for w in weights)

    def should_rebalance(self) -> bool:
        """Check if sector exposure needs rebalancing."""
        for exposure in self.analyze():
            if exposure.exposure_pct > self.max_sector_pct:
                return True
        return False
